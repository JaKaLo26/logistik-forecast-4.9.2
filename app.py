# app.py
# Logistik Forecast 4.9.5

from __future__ import annotations

import json
import os
from datetime import datetime
from io import StringIO

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.capacity import normalize_orders, summarize_orders
from src.clustering import cluster_orders
from src.geocoding import Geocoder
from src.routing import OSRMRouter
from src.maps import build_map
from src.forecast import (
    calculate_tour_forecast,
    recalculate_tour_from_stop,
    segment_rows,
    tour_summary,
)
from src.training_logger import log_forecast, training_logger_status

TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "").strip()

COLORS = [
    "#2563eb", "#7c3aed", "#0891b2", "#dc2626", "#ea580c",
    "#16a34a", "#9333ea", "#0f766e", "#be123c", "#a16207",
]

SMALL = {"class": "14 t", "pallet_capacity": 18, "payload_kg": 6000}
LARGE = {"class": "40 t", "pallet_capacity": 33, "payload_kg": 24000}


def _read_json(payload: str) -> pd.DataFrame:
    if not payload or str(payload).strip() in {"", "[]", "null"}:
        return pd.DataFrame()
    return pd.read_json(StringIO(payload), orient="records")


def _json_load(payload, default):
    try:
        return json.loads(payload) if payload else default
    except Exception:
        return default


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
        })
        idx += 1

    for i in range(1, int(n_large) + 1):
        rows.append({
            "vehicle_id": f"LKW-G{i:02d}",
            **LARGE,
            "color": COLORS[idx % len(COLORS)],
            "available": True,
        })
        idx += 1

    return pd.DataFrame(rows)


def update_fleet(n_small, n_large):
    df = make_fleet(max(0, int(n_small or 0)), max(0, int(n_large or 0)))
    if df.empty:
        return df, "⚠️ Keine Fahrzeuge konfiguriert."

    return (
        df,
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
            df = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(file, sep=None, engine="python", encoding="cp1252")

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

    for _, row in orders.iterrows():
        hits = geocoder.search(row["adresse"], 5)
        oid = str(row["auftrag"])
        candidates[oid] = hits
        best = hits[0] if hits else None

        if best:
            confidence = float(best.get("confidence", 0))
            status = "OK" if confidence >= 0.72 else "MANUELL PRÜFEN"
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
        })

    full = pd.DataFrame(full_rows)

    return (
        address_visible(full),
        full.to_json(orient="records", force_ascii=False),
        json.dumps(candidates, ensure_ascii=False),
        address_status(full),
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
        f"✅ {len(full) - open_count} automatisch/bestätigt · "
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
            "✅ Alle Adressen sind eindeutig.",
            gr.update(choices=[], value=None, interactive=False),
        )

    oid = str(row["auftrag"])
    hits = _json_load(candidates_json, {}).get(oid, [])
    choices = [h.get("display_name", "") for h in hits if h.get("display_name")]
    current = str(row.get("treffer", "") or "")

    if current and current not in choices:
        choices.insert(0, current)

    return (
        oid,
        json.dumps(hits, ensure_ascii=False),
        f"### Jetzt prüfen\n**{row['kunde']}**  \n{row['eingabe']}",
        gr.update(
            choices=choices,
            value=choices[0] if choices else None,
            interactive=bool(choices),
        ),
    )


def confirm_review(address_state_json, candidates_json, oid, hits_json, selected):
    full = _read_json(address_state_json)

    if not oid:
        raise gr.Error("Keine offene Adresse ausgewählt.")
    if not selected:
        raise gr.Error("Bitte Adresse auswählen.")

    hits = _json_load(hits_json, [])
    chosen = next((h for h in hits if h.get("display_name") == selected), None)

    if chosen is None:
        current_rows = full[full["auftrag"].astype(str) == str(oid)]
        if not current_rows.empty:
            current = current_rows.iloc[0]
            if str(current.get("treffer", "")) == str(selected):
                chosen = {
                    "display_name": selected,
                    "lat": current["lat"],
                    "lon": current["lon"],
                    "confidence": current["confidence"],
                    "provider": current["provider"],
                }

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
        info = "✅ **Adressprüfung abgeschlossen.**"
        choice = gr.update(choices=[], value=None, interactive=False)
    else:
        next_oid = str(next_row["auftrag"])
        hits2 = _json_load(candidates_json, {}).get(next_oid, [])
        choices = [h.get("display_name", "") for h in hits2 if h.get("display_name")]
        current2 = str(next_row.get("treffer", "") or "")

        if current2 and current2 not in choices:
            choices.insert(0, current2)

        next_hits = json.dumps(hits2, ensure_ascii=False)
        info = f"### Jetzt prüfen\n**{next_row['kunde']}**  \n{next_row['eingabe']}"
        choice = gr.update(
            choices=choices,
            value=choices[0] if choices else None,
            interactive=bool(choices),
        )

    return (
        address_visible(full),
        full.to_json(orient="records", force_ascii=False),
        address_status(full),
        f"✅ Bestätigt: **{selected}**",
        next_oid,
        next_hits,
        info,
        choice,
    )


def save_addresses(address_state_json, orders_json):
    full = _read_json(address_state_json)
    orders = _read_json(orders_json)

    if full.empty:
        raise gr.Error("Keine Adressprüfung vorhanden.")

    if (full["status"] == "MANUELL PRÜFEN").any():
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


def create_clusters(geo_json, vehicle_table, depot_json):
    orders = _read_json(geo_json)
    vehicles = pd.DataFrame(vehicle_table)

    if orders.empty:
        raise gr.Error("Adressprüfung zuerst abschließen.")
    if vehicles.empty:
        raise gr.Error("Keine Fahrzeuge.")
    if not depot_json:
        raise gr.Error("Depot zuerst bestätigen.")

    depot = _json_load(depot_json, {})
    depot_coord = (float(depot["lat"]), float(depot["lon"]))

    assignments, summary, warnings = cluster_orders(
        orders,
        vehicles,
        depot_coord
    )

    msg = (
        "✅ Geografische Cluster erstellt. "
        "Jeder LKW erhält ein möglichst zusammenhängendes Liefergebiet."
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
        assignments.to_json(orient="records", force_ascii=False),
        msg,
    )


def optimize_routes(cluster_json, vehicle_table, depot_json):
    clustered = _read_json(cluster_json)
    vehicles = pd.DataFrame(vehicle_table)

    if clustered.empty:
        raise gr.Error("Zuerst Cluster bilden.")

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
            result = router.optimize_roundtrip(depot_coord, stops)
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

    choices = ["Alle Touren"] + [route["vehicle_id"] for route in routes]

    return (
        pd.DataFrame(route_rows),
        gr.update(choices=choices, value="Alle Touren", interactive=True),
        build_map(routes, depot),
        json.dumps(routes, ensure_ascii=False),
        json.dumps(debug, ensure_ascii=False, indent=2, default=str),
        (
            "✅ Stoppreihenfolge mit OSRM optimiert. "
            "Als Nächstes kann der segmentweise TomTom-Forecast berechnet werden."
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


def normalize_start_time(value):
    text = str(value or "").strip()

    if not text:
        return (
            datetime.now()
            .astimezone()
            .replace(second=0, microsecond=0)
            .isoformat()
        )

    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        raise gr.Error(
            "Startzeit bitte im ISO-Format eingeben, "
            "z. B. 2026-08-20T07:00:00+02:00"
        )

    if parsed.tzinfo is None:
        parsed = parsed.astimezone()

    return parsed.replace(second=0, microsecond=0).isoformat()


def calculate_all_forecasts(
    routes_json,
    depot_json,
    vehicle_table,
    start_time
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

    normalized_start = normalize_start_time(start_time)

    forecasts = {}
    summary_rows = []
    segment_table_rows = []
    debug = []
    logging_results = []
    vehicle_choices = []

    for route in routes:
        vehicle_id = str(route["vehicle_id"])
        vehicle_choices.append(vehicle_id)

        vehicle_rows = vehicles[
            vehicles["vehicle_id"].astype(str) == vehicle_id
        ]

        params = (
            {}
            if vehicle_rows.empty
            else vehicle_parameters(vehicle_rows.iloc[0])
        )

        tour_id = (
            f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
            f"-{vehicle_id}"
        )

        try:
            forecast = calculate_tour_forecast(
                route=route,
                depot=depot,
                start_time=normalized_start,
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
        summary_rows.append(tour_summary(forecast))

        for row in segment_rows(forecast):
            row["vehicle_id"] = vehicle_id
            row["forecast_version"] = forecast.get("forecast_version")
            segment_table_rows.append(row)

  