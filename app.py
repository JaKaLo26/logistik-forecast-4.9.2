# app.py
# Logistik Forecast 4.9.5

from __future__ import annotations

from html import escape
import hashlib
import json
import math
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from io import StringIO

import gradio as gr
import pandas as pd
import spaces
from uuid import uuid4
from src.ml_training import train_experiment
from src.synthetic_training import generate_training_data
from dotenv import load_dotenv

load_dotenv()

from src.capacity import normalize_orders, summarize_orders
from src.clustering import cluster_orders, DRIVER_DEFAULTS, plan_day
from src.geocoding import Geocoder
from src.routing import OSRMRouter
from src.maps import build_map
from src.forecast import (
    calculate_tour_forecast,
)
from src.tour_progress import loading_plan
from src.training_logger import log_forecast, training_logger_status

TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "").strip()


# ============================================================
# HUGGING FACE ZEROGPU COMPATIBILITY
# ============================================================

@spaces.GPU(duration=1)
def zerogpu_healthcheck():
    """Minimal ZeroGPU function required by Hugging Face ZeroGPU."""
    return "ok"

COLORS = [
    "#2563eb", "#7c3aed", "#0891b2", "#dc2626", "#ea580c",
    "#16a34a", "#9333ea", "#0f766e", "#be123c", "#a16207",
]

SMALL = {"class": "14 t", "pallet_capacity": 18, "payload_kg": 6000}
LARGE = {"class": "40 t", "pallet_capacity": 33, "payload_kg": 24000}


def _read_json(payload: str) -> pd.DataFrame:
    if not payload or str(payload).strip() in {"", "[]", "null"}:
        return pd.DataFrame()
    return pd.read_json(StringIO(payload), orient="records", dtype=False)


def _json_load(payload, default):
    try:
        return json.loads(payload) if payload else default
    except Exception:
        return default


def _candidate_id(oid, hit, index):
    """Stabile interne ID; der sichtbare Adresstext bleibt reine Anzeige."""
    source = "|".join([
        str(oid),
        str(hit.get("provider", "")),
        str(hit.get("lat", "")),
        str(hit.get("lon", "")),
        str(hit.get("display_name", "")),
        str(index),
    ])
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _prepare_candidates(oid, hits):
    prepared = []
    for index, hit in enumerate(hits or []):
        if not isinstance(hit, dict):
            continue
        candidate = dict(hit)
        candidate["candidate_id"] = (
            str(candidate.get("candidate_id") or "")
            or _candidate_id(oid, candidate, index)
        )
        prepared.append(candidate)
    return prepared


def _candidate_choices(hits):
    return [
        (str(hit["display_name"]), str(hit["candidate_id"]))
        for hit in hits
        if hit.get("display_name") and hit.get("candidate_id")
    ]


def _ensure_geo_dtypes(full):
    """Koordinaten und Confidence bleiben auch nach JSON-Roundtrips float."""
    for column in ("lat", "lon", "confidence"):
        if column not in full.columns:
            full[column] = float("nan")
        full[column] = pd.to_numeric(
            full[column],
            errors="coerce",
        ).astype(float)
    return full


def _normalize_column_name(name):
    value = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "kundenname": "kunde",
        "straße": "strasse",
        "str.": "strasse",
        "gewicht": "warengewicht_kg",
        "gewicht_kg": "warengewicht_kg",
        "servicezeit": "service_min",
        "servicezeit_min": "service_min",
        "entladezeit": "service_min",
        "entladezeit_min": "service_min",
        "palettenanzahl": "paletten",
        "anzahl_paletten": "paletten",
        "auftragsnummer": "auftrag",
        "auftrag_id": "auftrag",
        "postleitzahl": "plz",
        "stadt": "ort",
    }
    return aliases.get(value, value)


def make_fleet(n_small=3, n_large=3):
    rows = []
    idx = 0

    for i in range(1, int(n_small) + 1):
        rows.append({
            "vehicle_id": f"LKW-K{i:02d}",
            **SMALL,
            "color": COLORS[idx % len(COLORS)],
            "available": True,
            **DRIVER_DEFAULTS,
        })
        idx += 1

    for i in range(1, int(n_large) + 1):
        rows.append({
            "vehicle_id": f"LKW-G{i:02d}",
            **LARGE,
            "color": COLORS[idx % len(COLORS)],
            "available": True,
            **DRIVER_DEFAULTS,
        })
        idx += 1

    return pd.DataFrame(rows)


def update_fleet(n_small, n_large):
    df = make_fleet(max(0, int(n_small or 0)), max(0, int(n_large or 0)))
    if df.empty:
        return [], "⚠️ Noch keine Touren geplant."

    return (
        df.to_dict("records"),
        f"**{len(df)} Fahrzeuge · {int(df.pallet_capacity.sum())} "
        f"Palettenplätze · {int(df.payload_kg.sum()):,} kg Nutzlast**",
    )


def vehicle_parameters(vehicle_row):
    if "40" in str(vehicle_row.get("class", "")):
        return {
            "vehicle_weight_kg": 40000,
            "vehicle_height_m": 4.0,
            "vehicle_width_m": 2.55,
            "vehicle_length_m": 16.5,
            "vehicle_max_speed_kmh": 80,
            "vehicle_commercial": True,
        }

    return {
        "vehicle_weight_kg": 14000,
        "vehicle_height_m": 4.0,
        "vehicle_width_m": 2.55,
        "vehicle_length_m": 10.0,
        "vehicle_max_speed_kmh": 80,
        "vehicle_commercial": True,
    }


def read_csv(file):
    if not file:
        raise gr.Error("Bitte CSV auswählen.")

    try:
        try:
            df = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
        except UnicodeDecodeError:
            df = pd.read_csv(file, sep=None, engine="python", encoding="cp1252", dtype=str)

        if df.empty:
            raise gr.Error("CSV ist leer.")

        df.columns = [_normalize_column_name(c) for c in df.columns]
        df = normalize_orders(df)
        summary = summarize_orders(df)

        return (
            df,
            f"✅ **{summary['auftraege']} Aufträge · {summary['paletten']} Paletten · "
            f"{summary['gewicht_kg']:,} kg**",
            df.to_json(orient="records", force_ascii=False),
        )

    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"CSV konnte nicht importiert werden: {type(exc).__name__}: {exc}")


def geocode_all(orders_json):
    orders = _read_json(orders_json)
    if orders.empty:
        raise gr.Error("Bitte zuerst CSV importieren.")

    geocoder = Geocoder(TIMEOUT)
    full_rows = []
    candidates = {}
    failures = 0

    yield (
        pd.DataFrame(columns=["auftrag", "kunde", "adresse", "status", "treffer", "sicherheit"]),
        "[]",
        "{}",
        f"**Adressprüfung gestartet: 0 / {len(orders)} Adressen.** Die Adressdienste werden abgefragt.",
    )

    for _, row in orders.iterrows():
        oid = str(row["auftrag"])
        error = ""
        try:
            hits = _prepare_candidates(
                oid,
                geocoder.search(row["adresse"], 5),
            )
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            hits = []
            error = str(exc)
            failures += 1
        candidates[oid] = hits
        best = hits[0] if hits else None

        if best:
            confidence = float(best.get("confidence", 0))
            status = "MANUELL PRÜFEN"
        else:
            confidence = 0.0
            status = "MANUELL PRÜFEN"

        full_rows.append({
            "auftrag": oid,
            "kunde": str(row["kunde"]),
            "eingabe": str(row["adresse"]),
            "treffer": best.get("display_name", "") if best else "",
            "confidence": confidence,
            "status": status,
            "lat": best.get("lat") if best else None,
            "lon": best.get("lon") if best else None,
            "provider": best.get("provider", "") if best else "",
            "geocoding_error": error,
        })

        full = pd.DataFrame(full_rows)
        progress = f"**{len(full_rows)} / {len(orders)} Adressen bearbeitet.** "
        if failures:
            progress += (
                f"⚠️ Adressdienst bei {failures} Adresse(n) fehlgeschlagen. "
                "Diese Adressen bitte unten einzeln erneut suchen. "
            )
        yield (
            address_visible(full),
            full.to_json(orient="records", force_ascii=False),
            json.dumps(candidates, ensure_ascii=False),
            progress + address_status(full),
        )


def address_visible(full):
    if full.empty:
        return pd.DataFrame()

    return pd.DataFrame({
        "auftrag": full["auftrag"].astype(str),
        "kunde": full["kunde"].astype(str),
        "adresse": full["eingabe"].astype(str),
        "status": full["status"].astype(str),
        "treffer": full["treffer"].fillna("").astype(str),
        "sicherheit": (
            pd.to_numeric(full["confidence"], errors="coerce")
            .fillna(0).mul(100).round().astype(int).astype(str) + " %"
        ),
    })


def address_status(full):
    if full.empty:
        return "Noch keine Adressen geprüft."

    open_count = int((full["status"] == "MANUELL PRÜFEN").sum())
    return (
        f"**{len(full)} Adressen geprüft · "
        f"✅ {len(full) - open_count} bestätigt · "
        f"⚠️ {open_count} offen**"
    )


def _next_open(full):
    rows = full[full["status"].astype(str) == "MANUELL PRÜFEN"]
    return None if rows.empty else rows.iloc[0]


def prepare_review(address_state_json, candidates_json):
    full = _read_json(address_state_json)
    row = _next_open(full)

    if row is None:
        return (
            "",
            "[]",
            gr.update(value="", interactive=False),
            "✅ Alle Adressen sind eindeutig.",
            gr.update(choices=[], value=None, interactive=False),
        )

    oid = str(row["auftrag"])
    hits = _prepare_candidates(
        oid,
        _json_load(candidates_json, {}).get(oid, []),
    )
    choices = _candidate_choices(hits)

    return (
        oid,
        json.dumps(hits, ensure_ascii=False),
        gr.update(value=str(row["eingabe"]), interactive=True),
        f"### Jetzt prüfen\n**{row['kunde']}**  \n{row['eingabe']}",
        gr.update(
            choices=choices,
            value=None,
            interactive=bool(choices),
        ),
    )


def search_review_address(
    address_state_json,
    candidates_json,
    oid,
    address,
):
    full = _ensure_geo_dtypes(_read_json(address_state_json))
    address = str(address or "").strip()

    if full.empty or not oid:
        raise gr.Error("Keine offene Adresse ausgewählt.")
    if not address:
        raise gr.Error("Bitte eine Adresse eingeben.")

    mask = full["auftrag"].astype(str) == str(oid)
    if not mask.any():
        raise gr.Error("Der ausgewählte Auftrag wurde nicht gefunden.")

    try:
        hits = _prepare_candidates(
            str(oid),
            Geocoder(TIMEOUT).search(address, 5),
        )
    except Exception as exc:
        raise gr.Error(
            f"Adresssuche fehlgeschlagen: {type(exc).__name__}: {exc}"
        )

    candidates = _json_load(candidates_json, {})
    candidates[str(oid)] = hits
    full.loc[mask, "eingabe"] = address
    full.loc[mask, "status"] = "MANUELL PRÜFEN"

    if hits:
        best = hits[0]
        full.loc[mask, "treffer"] = best.get("display_name", "")
        full.loc[mask, "lat"] = best.get("lat")
        full.loc[mask, "lon"] = best.get("lon")
        full.loc[mask, "confidence"] = best.get("confidence", 0)
        full.loc[mask, "provider"] = best.get("provider", "")
        message = (
            f"### Treffer für die korrigierte Adresse\n"
            f"**{full.loc[mask, 'kunde'].iloc[0]}**  \n{address}"
        )
        status = f"✅ {len(hits)} Adressvorschlag/-vorschläge gefunden."
    else:
        full.loc[mask, "treffer"] = ""
        full.loc[mask, ["lat", "lon"]] = None
        full.loc[mask, "confidence"] = 0.0
        full.loc[mask, "provider"] = ""
        message = (
            "### Keine Treffer\n"
            f"**{full.loc[mask, 'kunde'].iloc[0]}**  \n"
            "Adresse oben korrigieren und erneut suchen."
        )
        status = "⚠️ Keine Vorschläge gefunden."

    choices = _candidate_choices(hits)
    return (
        address_visible(full),
        full.to_json(orient="records", force_ascii=False),
        json.dumps(candidates, ensure_ascii=False),
        json.dumps(hits, ensure_ascii=False),
        message,
        gr.update(
            choices=choices,
            value=None,
            interactive=bool(choices),
        ),
        status,
    )


def confirm_review(address_state_json, candidates_json, oid, hits_json, selected):
    full = _ensure_geo_dtypes(_read_json(address_state_json))

    if not oid:
        raise gr.Error("Keine offene Adresse ausgewählt.")
    if not selected:
        raise gr.Error("Bitte Adresse auswählen.")

    hits = _prepare_candidates(str(oid), _json_load(hits_json, []))
    chosen = next(
        (
            hit for hit in hits
            if str(hit.get("candidate_id")) == str(selected)
        ),
        None,
    )

    # Kompatibler Fallback für einen noch geöffneten alten Browser-State.
    if chosen is None:
        chosen = next(
            (
                hit for hit in hits
                if str(hit.get("display_name")) == str(selected)
            ),
            None,
        )

    if chosen is None or chosen.get("lat") is None or chosen.get("lon") is None:
        raise gr.Error("Vorschlag konnte nicht übernommen werden.")

    mask = full["auftrag"].astype(str) == str(oid)
    full.loc[mask, "treffer"] = chosen["display_name"]
    full.loc[mask, "lat"] = chosen["lat"]
    full.loc[mask, "lon"] = chosen["lon"]
    full.loc[mask, "confidence"] = chosen.get("confidence", 0)
    full.loc[mask, "provider"] = chosen.get("provider", "")
    full.loc[mask, "status"] = "MANUELL BESTÄTIGT"

    next_row = _next_open(full)

    if next_row is None:
        next_oid = ""
        next_hits = "[]"
        next_address = gr.update(value="", interactive=False)
        info = "✅ **Adressprüfung abgeschlossen.**"
        choice = gr.update(choices=[], value=None, interactive=False)
    else:
        next_oid = str(next_row["auftrag"])
        hits2 = _prepare_candidates(
            next_oid,
            _json_load(candidates_json, {}).get(next_oid, []),
        )
        choices = _candidate_choices(hits2)

        next_hits = json.dumps(hits2, ensure_ascii=False)
        next_address = gr.update(
            value=str(next_row["eingabe"]),
            interactive=True,
        )
        info = f"### Jetzt prüfen\n**{next_row['kunde']}**  \n{next_row['eingabe']}"
        choice = gr.update(
            choices=choices,
            value=None,
            interactive=bool(choices),
        )

    return (
        address_visible(full),
        full.to_json(orient="records", force_ascii=False),
        address_status(full),
        f"✅ Bestätigt: **{chosen['display_name']}**",
        next_oid,
        next_hits,
        next_address,
        info,
        choice,
    )


def save_addresses(address_state_json, orders_json):
    full = _read_json(address_state_json)
    orders = _read_json(orders_json)

    if full.empty:
        raise gr.Error("Keine Adressprüfung vorhanden.")

    if not full["status"].eq("MANUELL BESTÄTIGT").all():
        raise gr.Error("Es sind noch Adressen offen.")

    merged = orders.merge(
        full[["auftrag", "treffer", "lat", "lon"]],
        on="auftrag",
        how="left",
    )

    if merged[["lat", "lon"]].isna().any().any():
        raise gr.Error("Mindestens eine Adresse hat keine Koordinaten.")

    return (
        merged.to_json(orient="records", force_ascii=False),
        "✅ Adressen abgeschlossen. Weiter zur Flotte und Clusterbildung.",
    )


def search_depot(address):
    address = str(address or "").strip()

    if not address:
        raise gr.Error("Depotadresse eingeben.")

    hits = Geocoder(TIMEOUT).search(address, 5)

    if not hits:
        return (
            "[]",
            gr.update(choices=[], value=None, interactive=False),
            "⚠️ Kein Depot-Treffer.",
        )

    choices = [hit["display_name"] for hit in hits]

    return (
        json.dumps(hits, ensure_ascii=False),
        gr.update(choices=choices, value=choices[0], interactive=True),
        "Depotvorschlag gefunden – bitte bestätigen.",
    )


def confirm_depot(input_address, hits_json, selected):
    hits = _json_load(hits_json, [])
    chosen = next((h for h in hits if h.get("display_name") == selected), None)

    if not chosen:
        raise gr.Error("Depotvorschlag auswählen.")

    state = {
        "address_input": input_address,
        "display_name": chosen["display_name"],
        "lat": chosen["lat"],
        "lon": chosen["lon"],
        "confidence": chosen.get("confidence", 0),
        "provider": chosen.get("provider", ""),
    }

    return (
        json.dumps(state, ensure_ascii=False),
        f"✅ **Depot:** {chosen['display_name']}",
    )


def create_clusters(geo_json, depot_json):
    orders = _read_json(geo_json)
    if orders.empty:
        raise gr.Error("Adressprüfung zuerst abschließen.")
    if not depot_json:
        raise gr.Error("Depot zuerst bestätigen.")

    depot = _json_load(depot_json, {})
    depot_coord = (float(depot["lat"]), float(depot["lon"]))

    vehicles = make_fleet(0, len(orders))
    assignments, summary, warnings = cluster_orders(
        orders,
        vehicles,
        depot_coord
    )

    planned_fleet = []
    small_index = large_index = 0
    for row in summary.to_dict("records"):
        fits_small = (row["paletten"] <= SMALL["pallet_capacity"]
                      and row["gewicht_kg"] <= SMALL["payload_kg"])
        if fits_small:
            small_index += 1
            vehicle_id, spec = f"LKW-K{small_index:02d}", SMALL
        else:
            large_index += 1
            vehicle_id, spec = f"LKW-G{large_index:02d}", LARGE
        planned_fleet.append({
            "vehicle_id": vehicle_id, **spec, "available": True,
            "color": COLORS[len(planned_fleet) % len(COLORS)], **DRIVER_DEFAULTS,
        })
    if planned_fleet:
        id_map = dict(zip(summary["vehicle_id"], (v["vehicle_id"] for v in planned_fleet)))
        assignments["vehicle_id"] = assignments["vehicle_id"].replace(id_map)
        summary["vehicle_id"] = summary["vehicle_id"].map(id_map)
        for key, load, capacity in (
            ("paletten_auslastung_pct", "paletten", "pallet_capacity"),
            ("gewicht_auslastung_pct", "gewicht_kg", "payload_kg"),
        ):
            summary[key] = [round(100 * amount / v[capacity], 1)
                            for amount, v in zip(summary[load], planned_fleet)]

    unassigned = assignments["vehicle_id"].eq("NICHT ZUGEWIESEN")
    if unassigned.any():
        count = int(unassigned.sum())
        msg = (
            f"❌ **Fehler: {count} von {len(orders)} Aufträgen konnten keinem Lkw "
            "zugewiesen werden.** Prüfe Kapazität, Nutzlast und Fahrerzeitbudgets. "
            "Die Routenoptimierung ist erst bei vollständiger Zuordnung möglich."
        )
    else:
        msg = (
            f"✅ {len(planned_fleet)} Lkw eingeplant ({small_index} × 14 t, {large_index} × 40 t). "
            "Fahrzeit, Service, Pausen und Rückweg wurden eingeplant."
        )

    if warnings:
        msg += "\n\n⚠️ " + " | ".join(warnings)

    visible = assignments[
        [
            "cluster_id",
            "vehicle_id",
            "auftrag",
            "kunde",
            "adresse",
            "paletten",
            "gesamtgewicht_kg",
        ]
    ].copy()

    return (
        visible,
        summary,
        "[]" if unassigned.any() else assignments.to_json(orient="records", force_ascii=False),
        msg,
        "[]",
        planned_fleet if not unassigned.any() else [],
    )


def optimize_routes(cluster_json, vehicle_table, depot_json):
    clustered = _read_json(cluster_json)
    vehicles = pd.DataFrame(vehicle_table)

    if clustered.empty or clustered["vehicle_id"].eq("NICHT ZUGEWIESEN").any():
        raise gr.Error("Zuerst alle Aufträge einem Lkw zuweisen.")

    depot = _json_load(depot_json, {})
    depot_coord = (float(depot["lat"]), float(depot["lon"]))

    router = OSRMRouter(
        os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org"),
        TIMEOUT,
    )

    route_rows = []
    routes = []
    debug = []

    active = clustered[clustered["vehicle_id"] != "NICHT ZUGEWIESEN"]

    for vid, group in active.groupby("vehicle_id"):
        vehicle = vehicles[vehicles["vehicle_id"].astype(str) == str(vid)]

        if vehicle.empty:
            continue

        vehicle_row = vehicle.iloc[0]
        stops = [row.to_dict() for _, row in group.iterrows()]

        try:
            stops = sorted(stops, key=lambda stop: stop.get("cluster_sequence", 0))
            result = router.route([depot_coord] + [(float(s["lat"]), float(s["lon"])) for s in stops] + [depot_coord])
            result["ordered_stops"] = stops
            result["optimizer"] = "street-time-feasible-insertion"
            if len(result.get("legs", [])) != len(stops) + 1:
                raise ValueError("Unvollständige Fahrzeitdaten für Zeitprüfung.")
            driver_plan = plan_day([leg["duration"] / 60 for leg in result["legs"]],
                                   [float(s.get("service_min", 0)) for s in stops], vehicle_row)
            if not driver_plan["feasible"]:
                raise ValueError("Aktuelle Straßenroute überschreitet Fahrerbudget. Regionen erneut bilden.")
        except Exception as exc:
            debug.append({"vehicle_id": vid, "error": str(exc)})
            continue

        ordered = result["ordered_stops"]

        for seq, stop in enumerate(ordered, 1):
            stop = dict(stop)
            route_rows.append({
                "vehicle_id": vid,
                "cluster_id": stop.get("cluster_id", ""),
                "stopp_nr": seq,
                "ladereihenfolge": len(ordered) - seq + 1,
                "auftrag": stop.get("auftrag", ""),
                "kunde": stop.get("kunde", ""),
                "adresse": stop.get("adresse", ""),
                "paletten": int(stop.get("paletten", 0)),
                "gewicht_kg": int(stop.get("gesamtgewicht_kg", 0)),
                "service_min": float(stop.get("service_min", 0)),
            })

        routes.append({
            "vehicle_id": vid,
            "cluster_id": str(group.iloc[0]["cluster_id"]),
            "vehicle_class": str(vehicle_row.get("class", "")),
            "color": vehicle_row.get("color", "#2563eb"),
            "geometry": result["geometry"],
            "legs": result.get("legs", []),
            "stops": [
                {
                    "auftrag": stop.get("auftrag", ""),
                    "kunde": stop.get("kunde", ""),
                    "adresse": stop.get("adresse", ""),
                    "lat": float(stop["lat"]),
                    "lon": float(stop["lon"]),
                    "paletten": int(stop.get("paletten", 0)),
                    "gesamtgewicht_kg": int(stop.get("gesamtgewicht_kg", 0)),
                    "service_min": float(stop.get("service_min", 0)),
                }
                for stop in ordered
            ],
            "distance_m": float(result["distance_m"]),
            "duration_s": float(result["duration_s"]),
            "optimizer": result.get("optimizer", "osrm-trip"),
            "driver_plan": driver_plan,
        })

        debug.append({
            "vehicle_id": vid,
            "optimizer": result.get("optimizer"),
            "distance_m": result["distance_m"],
            "duration_s": result["duration_s"],
            "osrm_legs": len(result.get("legs", [])),
            "optimized_stop_count": len(ordered),
        })

    if not routes:
        raise gr.Error("Keine Route konnte optimiert werden.")

    choices = [route["vehicle_id"] for route in routes]

    return (
        pd.DataFrame(route_rows),
        gr.update(choices=choices, value=choices[0], interactive=True),
        build_map(routes[:1], depot),
        json.dumps(routes, ensure_ascii=False),
        json.dumps(debug, ensure_ascii=False, indent=2, default=str),
        (
            "✅ Straßenroute und Fahrerbudget geprüft; Cluster-Reihenfolge beibehalten. "
            "Pausen sind Zeitreserven ohne geprüfte Parkplätze. Forecast-Ankünfte enthalten diese Pausen noch nicht. "

        ),
    )


def render_route(routes_json, selected, depot_json):
    routes = _json_load(routes_json, [])
    depot = _json_load(depot_json, {})

    if selected and selected != "Alle Touren":
        routes = [
            route
            for route in routes
            if str(route["vehicle_id"]) == str(selected)
        ]

    return build_map(routes, depot)


TOUR_TIMEZONE = ZoneInfo("Europe/Berlin")


def current_tour_date():
    return datetime.now(TOUR_TIMEZONE).strftime("%d.%m.%y")


def normalize_start_time(date, time):
    date = (date or "").strip() or current_tour_date()
    time = (time or "").strip()
    if not time:
        raise gr.Error("Bitte eine Startzeit eingeben, z. B. 07:00.")
    for date_format in ("%d.%m.%y", "%d.%m.%Y"):
        try:
            parsed = datetime.strptime(f"{date} {time}", f"{date_format} %H:%M")
            return parsed.replace(tzinfo=TOUR_TIMEZONE).isoformat()
        except ValueError:
            continue
    raise gr.Error("Bitte Datum und Uhrzeit wie 03.11.26 und 07:00 eingeben.")


def render_forecast_card(vehicle_id, driver_status, segments, expanded=False):
    def minutes(value):
        return f"{value / 60:.0f} min" if value is not None else "nicht verfügbar"

    fields = (
        ("OSRM", "osrm_baseline_s"),
        ("TTLive= Ist-Zeit", "tomtom_live_s"),
        ("TomTom historisch", "tomtom_historic_s"),
        ("Eigener Forecast", "model_forecast_s"),
    )
    opening = " open" if expanded else ""
    parts = [
        f'<details{opening} style="padding:12px;border:1px solid #64748b;border-radius:12px;margin-bottom:8px">',
        f"<summary><strong>Lkw {escape(str(vehicle_id))}</strong> · {escape(driver_status)}</summary>",
    ]
    for index, segment in enumerate(segments, 1):
        origin = "Depot" if index == 1 else f"Stopp {index - 1}"
        destination = "Depot" if segment["to_stop_id"] == "DEPOT" else f"Stopp {index}"
        name = escape(str(segment.get("to_stop_name") or destination))
        predicted = minutes(segment.get("model_forecast_s"))
        parts.append(f'<details style="padding:8px;margin-left:8px"><summary>{origin} → {destination} · {name} · Forecast {predicted}</summary>')
        parts.append('<div style="display:flex;flex-wrap:wrap;gap:12px;padding:8px">')
        for label, key in fields:
            value = segment.get(key)
            if key.startswith("tomtom_") and not segment.get("tomtom_success"):
                value = None
            parts.append(f"<span>{label}: <strong>{minutes(value)}</strong></span>")
        if destination != "Depot":
            parts.append(f"<span>Service am Ziel: <strong>{minutes(segment.get('planned_service_s'))}</strong></span>")
        parts.append("</div></details>")
    parts.append("</details>")
    return "".join(parts)


def calculate_all_forecasts(
    routes_json,
    depot_json,
    vehicle_table,
    start_date,
    start_time,
    ml_bundle=None
):
    routes = _json_load(routes_json, [])
    depot = _json_load(depot_json, {})
    vehicles = pd.DataFrame(vehicle_table)

    if not routes:
        raise gr.Error("Zuerst Routen optimieren.")
    if not depot:
        raise gr.Error("Depot fehlt.")
    if not TOMTOM_API_KEY:
        raise gr.Error("TOMTOM_API_KEY fehlt im Hugging-Face-Space.")

    normalized_start = normalize_start_time(start_date, start_time)

    forecasts = {}
    tour_cards = {}
    deviations = []
    debug = []
    logging_results = []

    for route in routes:
        vehicle_id = str(route["vehicle_id"])

        vehicle_rows = vehicles[
            vehicles["vehicle_id"].astype(str) == vehicle_id
        ]

        params = (
            {}
            if vehicle_rows.empty
            else vehicle_parameters(vehicle_rows.iloc[0])
        )

        tour_id = str(uuid4())

        try:
            forecast = calculate_tour_forecast(
                route=route,
                depot=depot,
                start_time=normalized_start,
                ml_bundle=ml_bundle,
                api_key=TOMTOM_API_KEY,
                timeout=TIMEOUT,
                tour_id=tour_id,
                forecast_version=1,
                vehicle_parameters=params,
            )
        except Exception as exc:
            debug.append({
                "vehicle_id": vehicle_id,
                "status": "forecast_failed",
                "error": str(exc),
            })
            continue

        forecasts[vehicle_id] = forecast
        segments = forecast["segments"]
        limits = ("remaining_drive_min", "remaining_week_drive_min",
                  "remaining_fortnight_drive_min")
        missing_live = [
            f"{segment.get('from_stop_name', 'Start')} → {segment.get('to_stop_name', 'Ziel')} "
            f"({segment.get('tomtom_error') or 'kein gültiger Live-Wert'})"
            for segment in segments
            if not (
                segment.get("tomtom_success")
                and isinstance(segment.get("tomtom_live_s"), (int, float))
                and math.isfinite(segment["tomtom_live_s"])
                and segment["tomtom_live_s"] >= 0
            )
        ]
        live_available = bool(segments) and not missing_live
        if live_available:
            live_seconds = sum(segment["tomtom_live_s"] for segment in segments)
            forecast_seconds = sum(segment["model_forecast_s"] for segment in segments)
            driving = live_seconds / 60
            delta = round((forecast_seconds - live_seconds) / 60)
            difference = f"Abweichung eigener Forecast zu TTLive: {delta:+d} min" if delta else "Abweichung eigener Forecast zu TTLive: 0 min"
        else:
            difference = "Abweichung nicht verfügbar (TTLive unvollständig)."

        if vehicle_rows.empty:
            driver_status = "Fahrerbudget fehlt."
        else:
            driver = vehicle_rows.iloc[0]
            budgets = [float(driver.get(key, DRIVER_DEFAULTS[key])) for key in limits]
            budget = min(540, *budgets) if all(math.isfinite(v) and v >= 0 for v in budgets) else None
            if budget is None:
                driver_status = "Fahrerbudget ungültig."
            elif not live_available:
                driver_status = "⛔ Live-Prüfung fehlgeschlagen. " + ("; ".join(missing_live) or "Keine Abschnitte vorhanden.")
            elif driving > budget:
                driver_status = (
                    f"⛔ Lenkzeitbudget um {math.ceil(driving - budget)} min überschritten. "
                    f"TTLive: {driving:.0f} von {budget:.0f} min inklusive Rückfahrt. Tour neu planen."
                )
            else:
                driver_status = (
                    f"{budget - driving:.0f} min Restlenkzeit "
                    f"nach der Tour (TTLive: {driving:.0f} von {budget:.0f} min inklusive Rückfahrt)."
                )

        deviations.append(f"- **{vehicle_id}** · {difference}")
        tour_cards[vehicle_id] = render_forecast_card(vehicle_id, driver_status, segments, expanded=True)

        try:
            log_result = log_forecast(forecast)
            logging_results.append({
                "vehicle_id": vehicle_id,
                "result": log_result,
            })
        except Exception as exc:
            logging_results.append({
                "vehicle_id": vehicle_id,
                "error": str(exc),
            })

        debug.append({
            "vehicle_id": vehicle_id,
            "tour_id": forecast.get("tour_id"),
            "forecast_version": forecast.get("forecast_version"),
            "segments": forecast.get("segments"),
        })

    if not forecasts:
        raise gr.Error("Kein Forecast konnte berechnet werden.")

    combined_debug = {
        "forecast": debug,
        "logging": logging_results,
        "logger_status": training_logger_status(),
    }

    return (
        tour_cards,
        "\n".join(deviations),
        json.dumps(combined_debug, ensure_ascii=False, indent=2, default=str),
        f"Forecast für {len(forecasts)} von {len(routes)} Touren berechnet.",
    )


def logger_status_markdown():
    status = training_logger_status()

    if status.get("remote_logging_ready"):
        return (
            "🟢 **Trainingsspeicher bereit**  \n"
            f"Dataset: `{status.get('hf_dataset_repo')}`"
        )

    problems = []

    if not status.get("hf_token_configured"):
        problems.append("HF_TOKEN fehlt")

    if not status.get("dataset_configured"):
        problems.append("HF_DATASET_REPO fehlt")

    return (
        "🟠 **Remote-Training-Logging noch nicht bereit.**  \n"
        + " · ".join(problems)
        + "  \nLokaler Fallback bleibt aktiv."
    )


CSS = """
html, body { background: #eef5fa !important; color-scheme: light; }
.gradio-container {
    min-height: 100vh; max-width: 1440px !important; margin: auto;
    padding: 32px 28px !important;
    background: radial-gradient(ellipse at 8% 0%, #d9f5ee 0%, transparent 45%),
                radial-gradient(ellipse at 95% 12%, #e0e5ff 0%, transparent 42%),
                linear-gradient(145deg, #f5fafc, #edf3fa) !important;
}
.gradio-container > .main { background: transparent !important; }
#app-hero {
    padding: 32px; margin-bottom: 18px; border: 1px solid white;
    border-radius: 28px; background: linear-gradient(120deg, #ffffffdd, #ffffff80);
    box-shadow: 0 12px 45px #355b8010, inset 0 1px 0 white;
}
.hero-top { display: flex; justify-content: space-between; gap: 16px; align-items: center; }
.hero-brand { font-size: 12px; font-weight: 750; letter-spacing: .16em; color: #426078; }
.hero-badge { border: 1px solid #b7dbd3; border-radius: 99px; padding: 6px 12px; color: #246656; background: #e7f7f0; font-size: 12px; }
#app-hero h1 { font-size: clamp(28px, 4vw, 46px); line-height: 1.15; letter-spacing: -.04em; color: #19364d; margin: 24px 0 12px; font-weight: 750; }
#app-hero p { color: #50677c; max-width: 660px; font-size: 16px; line-height: 1.65; margin: 0; }
.hero-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 24px; }
.hero-pills span { padding: 8px 13px; border: 1px solid #ffffff; border-radius: 99px; background: #ffffffb8; color: #3e5b70; font-size: 12px; }
#workflow-tabs > .tab-nav {
    display: flex; flex-wrap: wrap; gap: 6px; padding: 8px;
    background: #ffffffb8; border: 1px solid white; border-radius: 20px;
    box-shadow: 0 4px 18px #355b8009; margin-bottom: 18px;
}
#workflow-tabs > .tab-nav button {
    flex: 1 1 auto; min-height: 46px; border: 0; border-radius: 13px;
    color: #476176; font-size: 13px; padding: 10px 14px; transition: background .18s ease;
}
#workflow-tabs > .tab-nav button[aria-selected="true"] {
    background: #235e83 !important; color: white !important; box-shadow: 0 4px 12px #235e8326;
}
#workflow-tabs > .tabitem {
    padding: 24px !important; border: 1px solid white !important; border-radius: 24px !important;
    background: #ffffffa6 !important; box-shadow: 0 12px 36px #355b800b;
}
.gradio-container .block { border-radius: 16px !important; }
.gradio-container button { min-height: 44px; }
.gradio-container button.primary { box-shadow: 0 5px 15px #236d9220; }
.gradio-container :is(button, input, textarea, select):focus-visible {
    outline: 3px solid #247ca5 !important; outline-offset: 3px;
}
.step-intro { margin-bottom: 12px; }
.step-intro h2 { color: #19364d; letter-spacing: -.025em; }
.step-intro p { color: #50677c; }
#tour-map iframe { width: 100% !important; min-height: 480px; height: 62vh; border: 0; border-radius: 20px; }
#tour-map { min-height: 240px; }
@supports (backdrop-filter: blur(16px)) {
    #app-hero, #workflow-tabs > .tab-nav { backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); }
}
/* Prevent wide Gradio rows, dataframes and embedded content from clipping mobile view. */
.gradio-container, .gradio-container * { box-sizing: border-box; }
.gradio-container { width: 100% !important; max-width: 100vw !important; min-width: 0 !important; }
.gradio-container :is(.gr-row, .gr-column, .gr-group, .gr-form, .gr-panel, .block, .wrap, .contain) { min-width: 0 !important; max-width: 100%; }
.gradio-container :is(.table-wrap, .dataframe, .overflow-x-auto) { max-width: 100%; overflow-x: auto !important; -webkit-overflow-scrolling: touch; }
.gradio-container :is(img, iframe, svg) { max-width: 100%; }
@media(max-width:768px) {
    .gradio-container :is(.gr-row, .gradio-row) { flex-wrap: wrap !important; }
    .gradio-container :is(.gr-column, .gradio-column) { min-width: 0 !important; flex: 1 1 100% !important; width: 100% !important; }
    #workflow-tabs > .tab-nav { min-width: 0; width: 100%; }
    #workflow-tabs > .tab-nav button { min-width: 0; overflow-wrap: anywhere; }
    #workflow-tabs > .tabitem { min-width: 0; max-width: 100%; }
    #tour-map, #tour-map iframe { width: 100% !important; max-width: 100% !important; }
}
@media(max-width:768px) {
    .gradio-container { padding: 16px 10px !important; }
    #app-hero { padding: 22px 18px; border-radius: 22px; }
    .hero-top { flex-wrap: wrap; gap: 10px; }
    #workflow-tabs > .tab-nav { display: grid; grid-template-columns: 1fr 1fr; }
    #workflow-tabs > .tab-nav button { text-align: left; font-size: 12px; padding: 10px; }
    #workflow-tabs > .tabitem { padding: 14px !important; }
    #tour-map iframe { min-height: 360px; height: 55vh; }
}
@media(prefers-reduced-motion: reduce) {
    .gradio-container *, .gradio-container *::before, .gradio-container *::after { transition: none !important; animation: none !important; }
}
"""



# Keep the requested light palette even when the host uses dark mode.
THEME = gr.themes.Soft(
    primary_hue="sky", secondary_hue="teal", neutral_hue="slate",
    font=["ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=["ui-monospace", "monospace"], radius_size="lg",
)
_light_tokens = {
    "body_background_fill": "#eef5fa", "body_text_color": "#243e53",
    "body_text_color_subdued": "#52697b",
    "background_fill_primary": "#ffffff", "background_fill_secondary": "#f0f6fa",
    "block_background_fill": "#ffffffd9", "block_border_color": "#dbe7ee",
    "block_label_background_fill": "#f4f8fb", "block_label_text_color": "#38566c",
    "block_title_text_color": "#243e53", "block_info_text_color": "#52697b",
    "input_background_fill": "#ffffff", "input_border_color": "#cbdde7",
    "input_placeholder_color": "#657b8c", "input_background_fill_focus": "#ffffff",
    "border_color_primary": "#dbe7ee", "border_color_accent": "#277b9e",
    "panel_background_fill": "#f4f8fb", "panel_border_color": "#dbe7ee",
    "button_primary_background_fill": "#235e83", "button_primary_background_fill_hover": "#194c6d",
    "button_primary_text_color": "#ffffff", "button_primary_border_color": "#235e83",
    "button_secondary_background_fill": "#edf5f9", "button_secondary_background_fill_hover": "#e0eef5",
    "button_secondary_text_color": "#294e66", "button_secondary_border_color": "#cfdee8",
    "table_text_color": "#243e53", "table_even_background_fill": "#f3f8fb",
    "table_odd_background_fill": "#ffffff", "table_border_color": "#dbe7ee",
    "code_background_fill": "#f1f6fa", "accordion_text_color": "#38566c",
}
THEME = THEME.set(**_light_tokens, **{key + "_dark": value for key, value in _light_tokens.items()})

with gr.Blocks(
    title="Logistik Forecast 4.9.5",
    css=CSS,
    theme=THEME
) as demo:

    gr.HTML("""
        <div class="hero-top"><span class="hero-brand">LOGISTIK / FORECAST</span><span class="hero-badge">AI Workflow · Testversion</span></div>
        <h1>Touren planen.<br>Ankünfte besser einschätzen.</h1>
        <p>Von der ersten Adresse bis zum datenbasierten Vergleich.
        Plane deine Flotte, aktualisiere Prognosen und prüfe die Ergebnisse.</p>
        <div class="hero-pills"><span>01 — Planung</span><span>02 — Forecast &amp; Reforecast</span><span>03 — Training &amp; Vergleich</span></div>
    """, elem_id="app-hero")

    ml_state = gr.State(None)
    orders_state = gr.State("[]")
    address_state = gr.State("[]")
    candidates_state = gr.State("{}")
    geo_state = gr.State("[]")
    depot_hits_state = gr.State("[]")
    depot_state = gr.State("")
    cluster_state = gr.State("[]")
    routes_state = gr.State("[]")
    selected_order_state = gr.State("")
    selected_hits_state = gr.State("[]")

    with gr.Tabs(elem_id="workflow-tabs"):

        with gr.Tab("1 · Aufträge & Adressen"):
            gr.Markdown("## Aufträge & Adressen\nCSV auswählen und importieren. Wähle für jede Adresse einen Vorschlag und bestätige ihn.", elem_classes=["step-intro"])
            upload = gr.File(
                label="CSV-Datei",
                file_types=[".csv"],
                type="filepath"
            )
            import_btn = gr.Button("CSV importieren", variant="primary")
            order_summary = gr.Markdown()
            address_status_md = gr.Markdown()
            finish_status = gr.Markdown()
            with gr.Accordion("Aufträge und Prüfergebnisse anzeigen", open=False):
                orders_table = gr.Dataframe(
                    label="Aufträge",
                    interactive=False,
                    wrap=True
                )
                address_table = gr.Dataframe(
                    headers=[
                        "auftrag", "kunde", "adresse",
                        "status", "treffer", "sicherheit"
                    ],
                    label="Adressprüfung",
                    interactive=False,
                    wrap=True,
                )
            with gr.Group(visible=False) as address_review:
                review_info = gr.Markdown("Noch keine Prüfung gestartet.")
                review_address = gr.Textbox(
                    label="Adresse korrigieren oder ergänzen",
                    placeholder="Straße Hausnummer, PLZ Ort",
                    interactive=False,
                )
                search_review_btn = gr.Button(
                    "Korrigierte Adresse erneut suchen"
                )
                address_choice = gr.Radio(
                    choices=[],
                    label="Welche Adresse ist richtig?",
                    interactive=False
                )
                confirm_review_btn = gr.Button(
                    "Adresse bestätigen",
                    variant="primary"
                )
                confirm_status = gr.Markdown()


        with gr.Tab("2 · Depot, Flotte & Regionen"):
            gr.Markdown("## Depot & Flotte\nBestätige das Depot. Die App ermittelt aus den Aufträgen die Tourgruppen und die benötigten Lkw.", elem_classes=["step-intro"])
            depot_input = gr.Textbox(
                label="Depotadresse",
                placeholder="z. B. Mercedesstraße 1, 70372 Stuttgart"
            )
            depot_search_btn = gr.Button("Depot automatisch prüfen")
            depot_choice = gr.Radio(
                choices=[],
                label="Gefundene Depotadresse",
                interactive=False
            )
            depot_search_status = gr.Markdown()
            depot_confirm_btn = gr.Button(
                "Depot bestätigen",
                variant="primary"
            )
            depot_status = gr.Markdown()

            vehicle_table = gr.State([])
            gr.Markdown("### Eingeplante Fahrzeuge")

            @gr.render(inputs=vehicle_table)
            def render_vehicle_cards(fleet):
                if not fleet:
                    gr.Markdown("Noch keine Touren geplant.")
                for vehicle in fleet:
                    vehicle_id = vehicle["vehicle_id"]
                    with gr.Group():
                        gr.Markdown(
                            f"**{vehicle_id} · {vehicle['class']}**  \n"
                            f"Palettenkapazität: **{vehicle['pallet_capacity']}** · "
                            f"Maximale Zuladung: **{vehicle['payload_kg']} kg** · "
                            f"Verbleibende Lenkzeit: **{vehicle['remaining_drive_min']} min**"
                        )

            cluster_btn = gr.Button(
                "Tourgruppen und Lkw ermitteln",
                variant="primary"
            )
            cluster_status = gr.Markdown()
            cluster_summary = gr.Dataframe(
                label="Regionen / LKW-Auslastung",
                wrap=True
            )
            cluster_assignments = gr.Dataframe(
                label="Aufträge nach Region",
                wrap=True
            )

        with gr.Tab("3 · Routen & Forecast"):
            gr.Markdown("## Deine Touren im Überblick\nBerechne die Straßenrouten und den Forecast. Wähle anschließend einen Lkw aus.", elem_classes=["step-intro"])
            gr.Markdown(
                "🟢 **TomTom API konfiguriert**"
                if TOMTOM_API_KEY
                else "🔴 **TOMTOM_API_KEY fehlt**"
            )
            logger_status_ui = gr.Markdown(logger_status_markdown())
            with gr.Row():
                start_date_input = gr.Textbox(label="Tourdatum", value=current_tour_date, placeholder="03.11.26")
                start_time_input = gr.Textbox(label="Startzeit", placeholder="07:00")
            optimize_btn = gr.Button(
                "Routen und Forecast berechnen",
                variant="primary"
            )
            optimize_status = gr.Markdown()
            optimized_stops = gr.Dataframe(
                label="Optimierte Stopp-Reihenfolge",
                wrap=True
            )
            route_selector = gr.Dropdown(
                choices=[],
                label="Tour anzeigen",
                interactive=False
            )
            loading_table = gr.Dataframe(label="Ladeplan für den ausgewählten LKW", interactive=False)
            map_html = gr.HTML(elem_id="tour-map")
            with gr.Accordion("Technische Details", open=False):
                routing_debug = gr.Code(
                    label="Routing-Debug",
                    language="json"
                )

            forecast_status = gr.Markdown()
            forecast_cards = gr.HTML()
            forecast_by_vehicle = gr.State({})
            with gr.Accordion("Technische Details", open=False):
                traffic_debug = gr.Code(
                    label="Forecast / Logging Debug",
                    language="json"
                )

        with gr.Tab("4 · IHK: Training & Vergleich"):
            gr.Markdown("## Abweichung je Lkw")
            live_deviations = gr.Markdown("Berechne zuerst einen Forecast auf Seite 3. Der Vergleich umfasst die gesamte Tour inklusive Rückfahrt.")
            gr.Markdown("## Planbasis gegen ML prüfen\nAuftrags-CSV für den Offline-Test oder vorbereitete Segment-CSV hochladen. Training, Validierung und Test verwenden getrennte Touren.")
            synthetic_seed = gr.Number(label="Zufallsstartwert für Auftrags-CSV", value=42, precision=0, minimum=0)
            synthetic_count = gr.Number(label="Anzahl synthetischer Touren", value=30, precision=0, minimum=5, maximum=500)
            training_csv = gr.File(label="Trainingsdaten (CSV; Schema siehe README)", file_types=[".csv"], type="filepath")
            train_button = gr.Button("Trainieren, vergleichen und für diese Sitzung laden", variant="primary")
            training_status = gr.Markdown("Noch kein trainiertes Modell aktiv; Forecast nutzt die bestehende Heuristik.")
            training_metrics = gr.Dataframe(label="Fehler auf zurückgehaltenen Touren", interactive=False)
            training_details = gr.Dataframe(label="Plan, ML und Referenz je Testsegment", interactive=False)
            training_manifest = gr.Code(label="Modell- und Split-Nachweis", language="json")

    def run_training(file, seed, tour_count):
        if not file:
            raise gr.Error("Bitte Trainings-CSV hochladen.")
        try:
            frame = pd.read_csv(file, sep=None, engine="python", dtype={"plz": str})
            generator = None
            if "auftrag" in frame.columns and "prediction_created_at" not in frame.columns:
                frame, generator = generate_training_data(frame, seed=int(seed), tour_count=int(tour_count))
            bundle, metrics, details = train_experiment(frame)
            if generator:
                bundle["report"]["generator"] = generator
        except Exception as exc:
            raise gr.Error(f"Training abgewiesen: {exc}")
        report = bundle["report"]
        gain = metrics.iloc[0].MAE_min - metrics.iloc[1].MAE_min
        status = f"Modell aktiv für neue Forecasts dieser Sitzung. Herkunft: **{report['data_origin']}**. MAE-Differenz Plan minus ML: **{gain:.2f} min**. Positive Werte sprechen in diesem Test für ML; keine allgemeine Gütegarantie."
        if report['data_origin'] != 'real':
            status += " **Nur synthetischer/simulierter Funktionsnachweis.**"
        return bundle, status, metrics, details, json.dumps(report, ensure_ascii=False, indent=2)

    train_button.click(run_training, [training_csv, synthetic_seed, synthetic_count], [ml_state, training_status, training_metrics, training_details, training_manifest])

    def import_orders(file):
        return (*read_csv(file), "[]", gr.update(visible=False), "Adressen werden geprüft …")

    import_event = import_btn.click(
        import_orders,
        upload,
        [orders_table, order_summary, orders_state, geo_state, address_review, finish_status]
    )

    geocode_event = import_event.success(
        geocode_all,
        orders_state,
        [
            address_table,
            address_state,
            candidates_state,
            address_status_md
        ],
    )

    geocode_event.success(
        prepare_review,
        [address_state, candidates_state],
        [
            selected_order_state,
            selected_hits_state,
            review_address,
            review_info,
            address_choice
        ],
    )

    search_review_btn.click(
        search_review_address,
        [
            address_state,
            candidates_state,
            selected_order_state,
            review_address,
        ],
        [
            address_table,
            address_state,
            candidates_state,
            selected_hits_state,
            review_info,
            address_choice,
            confirm_status,
        ],
    )

    confirm_event = confirm_review_btn.click(
        confirm_review,
        [
            address_state,
            candidates_state,
            selected_order_state,
            selected_hits_state,
            address_choice
        ],
        [
            address_table,
            address_state,
            address_status_md,
            confirm_status,
            selected_order_state,
            selected_hits_state,
            review_address,
            review_info,
            address_choice,
        ],
    )

    def update_address_readiness(addresses, orders):
        full = _read_json(addresses)
        pending = not full.empty and full["status"].eq("MANUELL PRÜFEN").any()
        if full.empty or pending:
            return "[]", "", gr.update(visible=bool(pending))
        geocoded, status = save_addresses(addresses, orders)
        return geocoded, status, gr.update(visible=False)

    for event in (geocode_event, confirm_event):
        event.success(
            update_address_readiness,
            [address_state, orders_state],
            [geo_state, finish_status, address_review],
        )

    depot_search_btn.click(
        search_depot,
        depot_input,
        [
            depot_hits_state,
            depot_choice,
            depot_search_status
        ],
    )

    depot_confirm_btn.click(
        confirm_depot,
        [
            depot_input,
            depot_hits_state,
            depot_choice
        ],
        [
            depot_state,
            depot_status
        ],
    )

    cluster_btn.click(
        create_clusters,
        [
            geo_state,
            depot_state
        ],
        [
            cluster_assignments,
            cluster_summary,
            cluster_state,
            cluster_status,
            routes_state,
            vehicle_table,
        ],
    )

    clear_results = optimize_btn.click(
        lambda: ({}, "", "", "", "[]"),
        outputs=[forecast_by_vehicle, forecast_cards, live_deviations, forecast_status, routes_state],
    )
    route_event = clear_results.success(
        optimize_routes,
        [
            cluster_state,
            vehicle_table,
            depot_state
        ],
        [
            optimized_stops,
            route_selector,
            map_html,
            routes_state,
            routing_debug,
            optimize_status,
        ],
    )

    route_selector.change(
        render_route,
        [
            routes_state,
            route_selector,
            depot_state
        ],
        map_html,
    )

    def show_loading_plan(routes_json, vehicle_id):
        route = next((route for route in _json_load(routes_json, [])
                      if str(route.get("vehicle_id")) == str(vehicle_id)), None)
        return pd.DataFrame(loading_plan(route["stops"]) if route else [])

    route_selector.change(show_loading_plan, [routes_state, route_selector], loading_table)

    forecast_event = route_event.success(
        calculate_all_forecasts,
        [
            routes_state,
            depot_state,
            vehicle_table,
            start_date_input,
            start_time_input,
            ml_state
        ],
        [forecast_by_vehicle, live_deviations, traffic_debug, forecast_status],
    )

    def show_vehicle_forecast(cards, vehicle_id):
        return cards.get(vehicle_id, "Für diesen Lkw liegt kein Forecast vor.") if vehicle_id else ""

    forecast_inputs = [forecast_by_vehicle, route_selector]
    forecast_event.success(show_vehicle_forecast, forecast_inputs, forecast_cards)
    route_selector.change(show_vehicle_forecast, forecast_inputs, forecast_cards)


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )


