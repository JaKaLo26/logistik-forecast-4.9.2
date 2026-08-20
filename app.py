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

    summary_df = pd.DataFrame(summary_rows)
    segments_df = pd.DataFrame(segment_table_rows)
    markdown = build_forecast_markdown(summary_df)

    first_vehicle = vehicle_choices[0] if vehicle_choices else None
    stop_choices = forecast_stop_choices(
        forecasts.get(first_vehicle, {})
    )

    combined_debug = {
        "forecast": debug,
        "logging": logging_results,
        "logger_status": training_logger_status(),
    }

    return (
        summary_df,
        segments_df,
        markdown,
        json.dumps(forecasts, ensure_ascii=False, default=str),
        json.dumps(combined_debug, ensure_ascii=False, indent=2, default=str),
        gr.update(
            choices=vehicle_choices,
            value=first_vehicle,
            interactive=bool(vehicle_choices)
        ),
        gr.update(
            choices=stop_choices,
            value=stop_choices[0] if stop_choices else None,
            interactive=bool(stop_choices)
        ),
        "✅ Segment-Forecast berechnet und Trainingslogger ausgeführt.",
    )


def build_forecast_markdown(summary_df):
    if summary_df is None or summary_df.empty:
        return "Noch kein Forecast vorhanden."

    lines = [
        "## Dynamischer Tour-Forecast 4.9.5",
        "",
        (
            "Die Fahrzeit wird **zwischen jedem Stopp separat** berechnet. "
            "Servicezeiten verschieben die Abfahrtszeit des nächsten Segments."
        ),
        "",
    ]

    for _, row in summary_df.iterrows():
        ist_value = row.get("ist_fahrzeit_min")
        ist_text = "noch offen" if pd.isna(ist_value) else f"{ist_value} min"

        lines += [
            f"### 🚚 {row['vehicle_id']} · Forecast V{row['forecast_version']}",
            f"- Stopps: **{row['stopps']}**",
            f"- Distanz: **{row['distanz_km']} km**",
            "",
            "**Fahrzeitvergleich**",
            f"- 1 · OSRM Basis: **{row['osrm_basis_min']} min**",
            f"- 2 · TomTom Live: **{row['tomtom_live_min']} min**",
            f"- 3 · TomTom Historisch: **{row['tomtom_historisch_min']} min**",
            f"- 4 · Eigener Forecast: **{row['forecast_fahrzeit_min']} min**",
            f"- 5 · IST: **{ist_text}**",
            "",
            f"- Service geplant: **{row['service_geplant_min']} min**",
            (
                "- Service aktuell verwendet: "
                f"**{row.get('service_verwendet_min', row['service_geplant_min'])} min**"
            ),
            f"- Forecast Gesamt: **{row['forecast_gesamt_min']} min**",
            "",
        ]

    return "\n".join(lines)


def forecast_stop_choices(forecast):
    if not forecast:
        return []

    choices = []
    seen = set()

    for segment in forecast.get("segments") or []:
        stop_id = segment.get("to_stop_id")

        if (
            not stop_id
            or stop_id == "DEPOT"
            or stop_id in seen
        ):
            continue

        seen.add(stop_id)
        choices.append(str(stop_id))

    return choices


def update_reforecast_stops(forecasts_json, vehicle_id):
    forecasts = _json_load(forecasts_json, {})
    forecast = forecasts.get(str(vehicle_id), {})
    choices = forecast_stop_choices(forecast)

    return gr.update(
        choices=choices,
        value=choices[0] if choices else None,
        interactive=bool(choices)
    )


def reforecast_from_stop(
    forecasts_json,
    routes_json,
    depot_json,
    vehicle_table,
    vehicle_id,
    stop_id,
    actual_service_min,
    actual_departure
):
    forecasts = _json_load(forecasts_json, {})
    routes = _json_load(routes_json, [])
    depot = _json_load(depot_json, {})
    vehicles = pd.DataFrame(vehicle_table)

    if not vehicle_id:
        raise gr.Error("Bitte LKW auswählen.")
    if not stop_id:
        raise gr.Error("Bitte Stopp auswählen.")

    previous = forecasts.get(str(vehicle_id))

    if not previous:
        raise gr.Error("Für diesen LKW existiert kein Forecast.")

    route = next(
        (
            route
            for route in routes
            if str(route.get("vehicle_id")) == str(vehicle_id)
        ),
        None
    )

    if not route:
        raise gr.Error("Route des LKW wurde nicht gefunden.")

    vehicle_rows = vehicles[
        vehicles["vehicle_id"].astype(str) == str(vehicle_id)
    ]

    params = (
        {}
        if vehicle_rows.empty
        else vehicle_parameters(vehicle_rows.iloc[0])
    )

    service_seconds = None

    if (
        actual_service_min is not None
        and str(actual_service_min).strip() != ""
    ):
        service_seconds = max(
            0,
            int(round(float(actual_service_min) * 60))
        )

    departure_value = str(actual_departure or "").strip() or None

    try:
        new_forecast = recalculate_tour_from_stop(
            previous_forecast=previous,
            route=route,
            depot=depot,
            from_stop_id=str(stop_id),
            new_departure_time=departure_value,
            actual_service_s=service_seconds,
            api_key=TOMTOM_API_KEY,
            timeout=TIMEOUT,
            vehicle_parameters=params,
        )
    except Exception as exc:
        raise gr.Error(
            "Neuberechnung fehlgeschlagen: "
            f"{type(exc).__name__}: {exc}"
        )

    forecasts[str(vehicle_id)] = new_forecast

    try:
        log_result = log_forecast(new_forecast)
    except Exception as exc:
        log_result = {"error": str(exc)}

    summary_rows = []
    segment_table_rows = []

    for vid, forecast in forecasts.items():
        summary_rows.append(tour_summary(forecast))

        for row in segment_rows(forecast):
            row["vehicle_id"] = vid
            row["forecast_version"] = forecast.get("forecast_version")
            segment_table_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    segments_df = pd.DataFrame(segment_table_rows)

    debug = {
        "reforecast_vehicle": vehicle_id,
        "reforecast_from_stop": stop_id,
        "new_forecast_version": new_forecast.get("forecast_version"),
        "logging": log_result,
        "forecast": new_forecast,
    }

    return (
        summary_df,
        segments_df,
        build_forecast_markdown(summary_df),
        json.dumps(forecasts, ensure_ascii=False, default=str),
        json.dumps(debug, ensure_ascii=False, indent=2, default=str),
        (
            f"✅ **{vehicle_id} ab Stopp {stop_id} neu berechnet. "
            f"Forecast-Version {new_forecast.get('forecast_version')} gespeichert.**"
        ),
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
html, body { background: #0f1117 !important; }
.gradio-container, .gradio-container > .main {
    background: #0f1117 !important;
    color: #f3f4f6 !important;
}
.gradio-container { min-height: 100vh; }
.step-title {
    font-size: 1.35rem;
    font-weight: 750;
    margin-bottom: .35rem;
}
#tour-map iframe {
    width: 100% !important;
    min-height: 560px !important;
    height: 62vh !important;
    border-radius: 12px;
}
#tour-map { min-height: 560px; }
@media(max-width:768px) {
    .gradio-container {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    #tour-map iframe {
        min-height: 500px !important;
        height: 58vh !important;
    }
}
"""


with gr.Blocks(
    title="Logistik Forecast 4.9.5",
    css=CSS
) as demo:

    gr.Markdown(
        "# Logistik Forecast 4.9.5\n"
        "**Adressen → Flotte → Cluster → OSRM → "
        "TomTom Segment-Forecast → Reforecast → ML-Dataset**"
    )

    orders_state = gr.State("[]")
    address_state = gr.State("[]")
    candidates_state = gr.State("{}")
    geo_state = gr.State("[]")
    depot_hits_state = gr.State("[]")
    depot_state = gr.State("")
    cluster_state = gr.State("[]")
    routes_state = gr.State("[]")
    forecasts_state = gr.State("{}")
    selected_order_state = gr.State("")
    selected_hits_state = gr.State("[]")

    with gr.Tabs():

        with gr.Tab("1 · Aufträge & Adressen"):
            upload = gr.File(
                label="CSV-Datei",
                file_types=[".csv"],
                type="filepath"
            )
            import_btn = gr.Button("CSV importieren", variant="primary")
            order_summary = gr.Markdown()
            orders_table = gr.Dataframe(
                label="Aufträge",
                interactive=False,
                wrap=True
            )
            geocode_btn = gr.Button("Adressen automatisch prüfen")
            address_status_md = gr.Markdown()
            address_table = gr.Dataframe(
                headers=[
                    "auftrag", "kunde", "adresse",
                    "status", "treffer", "sicherheit"
                ],
                label="Adressprüfung",
                interactive=False,
                wrap=True,
            )
            review_info = gr.Markdown("Noch keine Prüfung gestartet.")
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
            finish_addresses_btn = gr.Button("Adressprüfung abschließen")
            finish_status = gr.Markdown()

        with gr.Tab("2 · Depot, Flotte & Regionen"):
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

            with gr.Row():
                small_count = gr.Number(
                    label="14-t-LKW",
                    value=3,
                    minimum=0,
                    precision=0
                )
                large_count = gr.Number(
                    label="40-t-LKW",
                    value=3,
                    minimum=0,
                    precision=0
                )
                fleet_btn = gr.Button("Flotte aktualisieren")

            fleet_status = gr.Markdown()
            vehicle_table = gr.Dataframe(
                value=make_fleet(3, 3),
                label="Verfügbare Fahrzeuge",
                interactive=True,
                wrap=True,
            )
            cluster_btn = gr.Button(
                "Regionen bilden & LKW zuweisen",
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

        with gr.Tab("3 · Routenoptimierung"):
            optimize_btn = gr.Button(
                "LKW-Routen optimieren",
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
            map_html = gr.HTML(elem_id="tour-map")
            routing_debug = gr.Code(
                label="Routing-Debug",
                language="json"
            )

        with gr.Tab("4 · Dynamischer Forecast"):
            gr.Markdown(
                "🟢 **TomTom API konfiguriert**"
                if TOMTOM_API_KEY
                else "🔴 **TOMTOM_API_KEY fehlt**"
            )
            logger_status_ui = gr.Markdown(logger_status_markdown())
            start_time_input = gr.Textbox(
                label="Tourstart",
                placeholder="2026-08-20T07:00:00+02:00"
            )
            forecast_btn = gr.Button(
                "Segment-Forecast berechnen",
                variant="primary"
            )
            forecast_status = gr.Markdown()
            forecast_md = gr.Markdown()
            forecast_table = gr.Dataframe(
                label="Tourübersicht – 5 Zeitkategorien",
                wrap=True
            )
            segment_table = gr.Dataframe(
                label="Forecast je Streckenabschnitt",
                wrap=True
            )
            traffic_debug = gr.Code(
                label="Forecast / Logging Debug",
                language="json"
            )

        with gr.Tab("5 · Reforecast ab Stopp"):
            reforecast_vehicle = gr.Dropdown(
                choices=[],
                label="LKW",
                interactive=False
            )
            reforecast_stop = gr.Dropdown(
                choices=[],
                label="Neuberechnung ab Stopp",
                interactive=False
            )
            actual_service_min = gr.Number(
                label="Tatsächliche Servicezeit an diesem Stopp (Minuten)",
                minimum=0,
                precision=0,
            )
            actual_departure_input = gr.Textbox(
                label="Tatsächliche / neue Abfahrtszeit",
                placeholder="2026-08-20T10:35:00+02:00"
            )
            reforecast_btn = gr.Button(
                "Ab diesem Stopp neu berechnen",
                variant="primary"
            )
            reforecast_status = gr.Markdown()
            reforecast_summary = gr.Dataframe(
                label="Aktuelle Forecast-Versionen",
                wrap=True
            )
            reforecast_segments = gr.Dataframe(
                label="Aktuelle Segmente",
                wrap=True
            )
            reforecast_md = gr.Markdown()
            reforecast_debug = gr.Code(
                label="Reforecast-Debug",
                language="json"
            )

    import_btn.click(
        read_csv,
        upload,
        [orders_table, order_summary, orders_state]
    )

    geocode_event = geocode_btn.click(
        geocode_all,
        orders_state,
        [
            address_table,
            address_state,
            candidates_state,
            address_status_md
        ],
    )

    geocode_event.then(
        prepare_review,
        [address_state, candidates_state],
        [
            selected_order_state,
            selected_hits_state,
            review_info,
            address_choice
        ],
    )

    confirm_review_btn.click(
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
            review_info,
            address_choice,
        ],
    )

    finish_addresses_btn.click(
        save_addresses,
        [address_state, orders_state],
        [geo_state, finish_status],
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

    fleet_btn.click(
        update_fleet,
        [
            small_count,
            large_count
        ],
        [
            vehicle_table,
            fleet_status
        ],
    )

    cluster_btn.click(
        create_clusters,
        [
            geo_state,
            vehicle_table,
            depot_state
        ],
        [
            cluster_assignments,
            cluster_summary,
            cluster_state,
            cluster_status
        ],
    )

    optimize_btn.click(
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

    forecast_btn.click(
        calculate_all_forecasts,
        [
            routes_state,
            depot_state,
            vehicle_table,
            start_time_input
        ],
        [
            forecast_table,
            segment_table,
            forecast_md,
            forecasts_state,
            traffic_debug,
            reforecast_vehicle,
            reforecast_stop,
            forecast_status,
        ],
    )

    reforecast_vehicle.change(
        update_reforecast_stops,
        [
            forecasts_state,
            reforecast_vehicle
        ],
        reforecast_stop,
    )

    reforecast_btn.click(
        reforecast_from_stop,
        [
            forecasts_state,
            routes_state,
            depot_state,
            vehicle_table,
            reforecast_vehicle,
            reforecast_stop,
            actual_service_min,
            actual_departure_input,
        ],
        [
            reforecast_summary,
            reforecast_segments,
            reforecast_md,
            forecasts_state,
            reforecast_debug,
            reforecast_status,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )
