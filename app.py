# app.py
# Logistik Forecast 4.9.5

from __future__ import annotations

import json
import os
from datetime import datetime
from io import StringIO

import gradio as gr
import pandas as pd
import spaces
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

from src.training_logger import (
    log_forecast,
    training_logger_status,
)


# ============================================================
# HUGGING FACE
# ============================================================

@spaces.GPU(duration=1)
def zerogpu_startup_check():
    return True


# ============================================================
# KONFIGURATION
# ============================================================

TIMEOUT = int(
    os.getenv(
        "REQUEST_TIMEOUT_SECONDS",
        "20"
    )
)

TOMTOM_API_KEY = os.getenv(
    "TOMTOM_API_KEY",
    ""
).strip()


# ============================================================
# FARBEN
# ============================================================

COLORS = [
    "#2563eb",
    "#7c3aed",
    "#0891b2",
    "#dc2626",
    "#ea580c",
    "#16a34a",
    "#9333ea",
    "#0f766e",
    "#be123c",
    "#a16207",
    "#1d4ed8",
    "#15803d",
    "#b91c1c",
    "#6d28d9",
    "#0369a1",
]


# ============================================================
# FAHRZEUGE
# ============================================================

SMALL = {
    "class": "14 t",
    "pallet_capacity": 18,
    "payload_kg": 6000,
}

LARGE = {
    "class": "40 t",
    "pallet_capacity": 33,
    "payload_kg": 24000,
}


# ============================================================
# JSON
# ============================================================

def _read_json(payload: str) -> pd.DataFrame:

    if (
        not payload
        or str(payload).strip()
        in {
            "",
            "[]",
            "null"
        }
    ):
        return pd.DataFrame()

    return pd.read_json(
        StringIO(payload),
        orient="records"
    )


def _json_load(
    payload,
    default
):

    try:

        if not payload:
            return default

        return json.loads(
            payload
        )

    except Exception:
        return default


# ============================================================
# CSV SPALTEN
# ============================================================

def _normalize_column_name(name):

    value = (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

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

    return aliases.get(
        value,
        value
    )


# ============================================================
# FLOTTE
# ============================================================

def make_fleet(
    n_small=3,
    n_large=3
):

    rows = []
    color_idx = 0

    for i in range(
        1,
        int(n_small) + 1
    ):

        rows.append({
            "vehicle_id": f"LKW-K{i:02d}",
            **SMALL,
            "color": COLORS[
                color_idx
                % len(COLORS)
            ],
            "available": True,
        })

        color_idx += 1

    for i in range(
        1,
        int(n_large) + 1
    ):

        rows.append({
            "vehicle_id": f"LKW-G{i:02d}",
            **LARGE,
            "color": COLORS[
                color_idx
                % len(COLORS)
            ],
            "available": True,
        })

        color_idx += 1

    return pd.DataFrame(
        rows
    )


def update_fleet(
    n_small,
    n_large
):

    df = make_fleet(
        max(
            0,
            int(
                n_small
                or 0
            )
        ),
        max(
            0,
            int(
                n_large
                or 0
            )
        )
    )

    if df.empty:

        return (
            df,
            "⚠️ Keine Fahrzeuge konfiguriert."
        )

    return (
        df,
        (
            f"**{len(df)} Fahrzeuge · "
            f"{int(df.pallet_capacity.sum())} "
            f"Palettenplätze · "
            f"{int(df.payload_kg.sum()):,} kg "
            f"Nutzlast**"
        )
    )


# ============================================================
# TOMTOM FAHRZEUGPARAMETER
# ============================================================

def vehicle_parameters(
    vehicle_row
):

    vehicle_class = str(
        vehicle_row.get(
            "class",
            ""
        )
    )

    if "40" in vehicle_class:

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


# ============================================================
# CSV
# ============================================================

def read_csv(file):

    if not file:

        raise gr.Error(
            "Bitte CSV auswählen."
        )

    try:

        try:

            df = pd.read_csv(
                file,
                sep=None,
                engine="python",
                encoding="utf-8-sig"
            )

        except UnicodeDecodeError:

            df = pd.read_csv(
                file,
                sep=None,
                engine="python",
                encoding="cp1252"
            )

        if df.empty:

            raise gr.Error(
                "CSV ist leer."
            )

        df.columns = [
            _normalize_column_name(
                column
            )
            for column
            in df.columns
        ]

        df = normalize_orders(
            df
        )

        summary = summarize_orders(
            df
        )

        return (
            df,
            (
                f"✅ **{summary['auftraege']} Aufträge · "
                f"{summary['paletten']} Paletten · "
                f"{summary['gewicht_kg']:,} kg**"
            ),
            df.to_json(
                orient="records",
                force_ascii=False
            ),
        )

    except gr.Error:
        raise

    except Exception as exc:

        raise gr.Error(
            "CSV konnte nicht importiert werden: "
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================
# ADRESSEN
# ============================================================

def geocode_all(
    orders_json
):

    orders = _read_json(
        orders_json
    )

    if orders.empty:

        raise gr.Error(
            "Bitte zuerst CSV importieren."
        )

    geocoder = Geocoder(
        TIMEOUT
    )

    full_rows = []
    candidates = {}

    for _, row in orders.iterrows():

        hits = geocoder.search(
            row["adresse"],
            5
        )

        oid = str(
            row["auftrag"]
        )

        candidates[
            oid
        ] = hits

        best = (
            hits[0]
            if hits
            else None
        )

        if best:

            confidence = float(
                best.get(
                    "confidence",
                    0
                )
            )

            status = (
                "OK"
                if confidence >= 0.72
                else "MANUELL PRÜFEN"
            )

        else:

            confidence = 0.0
            status = "MANUELL PRÜFEN"

        full_rows.append({
            "auftrag": oid,
            "kunde": str(
                row["kunde"]
            ),
            "eingabe": str(
                row["adresse"]
            ),
            "treffer": (
                best.get(
                    "display_name",
                    ""
                )
                if best
                else ""
            ),
            "confidence": confidence,
            "status": status,
            "lat": (
                best.get("lat")
                if best
                else None
            ),
            "lon": (
                best.get("lon")
                if best
                else None
            ),
            "provider": (
                best.get(
                    "provider",
                    ""
                )
                if best
                else ""
            ),
        })

    full = pd.DataFrame(
        full_rows
    )

    return (
        address_visible(
            full
        ),
        full.to_json(
            orient="records",
            force_ascii=False
        ),
        json.dumps(
            candidates,
            ensure_ascii=False
        ),
        address_status(
            full
        ),
    )


def address_visible(full):

    if full.empty:
        return pd.DataFrame()

    return pd.DataFrame({
        "auftrag":
            full["auftrag"].astype(str),

        "kunde":
            full["kunde"].astype(str),

        "adresse":
            full["eingabe"].astype(str),

        "status":
            full["status"].astype(str),

        "treffer":
            (
                full["treffer"]
                .fillna("")
                .astype(str)
            ),

        "sicherheit":
            (
                pd.to_numeric(
                    full["confidence"],
                    errors="coerce"
                )
                .fillna(0)
                .mul(100)
                .round()
                .astype(int)
                .astype(str)
                + " %"
            ),
    })


def address_status(full):

    if full.empty:

        return (
            "Noch keine Adressen geprüft."
        )

    open_count = int(
        (
            full["status"]
            == "MANUELL PRÜFEN"
        ).sum()
    )

    confirmed = (
        len(full)
        - open_count
    )

    return (
        f"**{len(full)} Adressen geprüft · "
        f"✅ {confirmed} automatisch/bestätigt · "
        f"⚠️ {open_count} offen**"
    )


def _next_open(full):

    rows = full[
        full["status"].astype(str)
        == "MANUELL PRÜFEN"
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def prepare_review(
    address_state_json,
    candidates_json
):

    full = _read_json(
        address_state_json
    )

    row = _next_open(
        full
    )

    if row is None:

        return (
            "",
            "[]",
            "✅ Alle Adressen sind eindeutig.",
            gr.update(
                choices=[],
                value=None,
                interactive=False
            ),
        )

    oid = str(
        row["auftrag"]
    )

    candidates = _json_load(
        candidates_json,
        {}
    )

    hits = candidates.get(
        oid,
        []
    )

    choices = [
        hit.get(
            "display_name",
            ""
        )
        for hit
        in hits
        if hit.get(
            "display_name"
        )
    ]

    current = str(
        row.get(
            "treffer",
            ""
        )
        or ""
    )

    if (
        current
        and current
        not in choices
    ):
        choices.insert(
            0,
            current
        )

    return (
        oid,
        json.dumps(
            hits,
            ensure_ascii=False
        ),
        (
            "### Jetzt prüfen\n"
            f"**{row['kunde']}**  \n"
            f"{row['eingabe']}"
        ),
        gr.update(
            choices=choices,
            value=(
                choices[0]
                if choices
                else None
            ),
            interactive=bool(
                choices
            ),
        ),
    )


def confirm_review(
    address_state_json,
    candidates_json,
    oid,
    hits_json,
    selected
):

    full = _read_json(
        address_state_json
    )

    if not oid:

        raise gr.Error(
            "Keine offene Adresse ausgewählt."
        )

    if not selected:

        raise gr.Error(
            "Bitte Adresse auswählen."
        )

    hits = _json_load(
        hits_json,
        []
    )

    chosen = next(
        (
            hit
            for hit in hits
            if hit.get(
                "display_name"
            )
            == selected
        ),
        None
    )

    if chosen is None:

        current_rows = full[
            full["auftrag"].astype(str)
            == str(oid)
        ]

        if not current_rows.empty:

            current = (
                current_rows.iloc[0]
            )

            if (
                str(
                    current.get(
                        "treffer",
                        ""
                    )
                )
                == str(
                    selected
                )
            ):

                chosen = {
                    "display_name": selected,
                    "lat": current["lat"],
                    "lon": current["lon"],
                    "confidence":
                        current["confidence"],
                    "provider":
                        current["provider"],
                }

    if (
        chosen is None
        or chosen.get("lat") is None
        or chosen.get("lon") is None
    ):

        raise gr.Error(
            "Vorschlag konnte nicht übernommen werden."
        )

    mask = (
        full["auftrag"].astype(str)
        == str(oid)
    )

    full.loc[
        mask,
        "treffer"
    ] = chosen[
        "display_name"
    ]

    full.loc[
        mask,
        "lat"
    ] = chosen[
        "lat"
    ]

    full.loc[
        mask,
        "lon"
    ] = chosen[
        "lon"
    ]

    full.loc[
        mask,
        "confidence"
    ] = chosen.get(
        "confidence",
        0
    )

    full.loc[
        mask,
        "provider"
    ] = chosen.get(
        "provider",
        ""
    )

    full.loc[
        mask,
        "status"
    ] = "MANUELL BESTÄTIGT"

    next_row = _next_open(
        full
    )

    if next_row is None:

        next_oid = ""
        next_hits = "[]"

        info = (
            "✅ **Adressprüfung abgeschlossen.**"
        )

        choice = gr.update(
            choices=[],
            value=None,
            interactive=False
        )

    else:

        next_oid = str(
            next_row["auftrag"]
        )

        all_candidates = _json_load(
            candidates_json,
            {}
        )

        hits2 = all_candidates.get(
            next_oid,
            []
        )

        choices = [
            hit.get(
                "display_name",
                ""
            )
            for hit
            in hits2
            if hit.get(
                "display_name"
            )
        ]

        current2 = str(
            next_row.get(
                "treffer",
                ""
            )
            or ""
        )

        if (
            current2
            and current2
            not in choices
        ):

            choices.insert(
                0,
                current2
            )

        next_hits = json.dumps(
            hits2,
            ensure_ascii=False
        )

        info = (
            "### Jetzt prüfen\n"
            f"**{next_row['kunde']}**  \n"
            f"{next_row['eingabe']}"
        )

        choice = gr.update(
            choices=choices,
            value=(
                choices[0]
                if choices
                else None
            ),
            interactive=bool(
                choices
            ),
        )

    return (
        address_visible(
            full
        ),
        full.to_json(
            orient="records",
            force_ascii=False
        ),
        address_status(
            full
        ),
        f"✅ Bestätigt: **{selected}**",
        next_oid,
        next_hits,
        info,
        choice,
    )


def save_addresses(
    address_state_json,
    orders_json
):

    full = _read_json(
        address_state_json
    )

    orders = _read_json(
        orders_json
    )

    if full.empty:

        raise gr.Error(
            "Keine Adressprüfung vorhanden."
        )

    if (
        full["status"]
        == "MANUELL PRÜFEN"
    ).any():

        raise gr.Error(
            "Es sind noch Adressen offen."
        )

    merged = orders.merge(
        full[
            [
                "auftrag",
                "treffer",
                "lat",
                "lon"
            ]
        ],
        on="auftrag",
        how="left",
    )

    if (
        merged[
            [
                "lat",
                "lon"
            ]
        ]
        .isna()
        .any()
        .any()
    ):

        raise gr.Error(
            "Mindestens eine Adresse hat keine Koordinaten."
        )

    return (
        merged.to_json(
            orient="records",
            force_ascii=False
        ),
        (
            "✅ Adressen abgeschlossen. "
            "Weiter zur Flotte und Clusterbildung."
        ),
    )


# ============================================================
# DEPOT
# ============================================================

def search_depot(address):

    address = str(
        address
        or ""
    ).strip()

    if not address:

        raise gr.Error(
            "Depotadresse eingeben."
        )

    hits = Geocoder(
        TIMEOUT
    ).search(
        address,
        5
    )

    if not hits:

        return (
            "[]",
            gr.update(
                choices=[],
                value=None,
                interactive=False
            ),
            "⚠️ Kein Depot-Treffer.",
        )

    choices = [
        hit["display_name"]
        for hit
        in hits
    ]

    return (
        json.dumps(
            hits,
            ensure_ascii=False
        ),
        gr.update(
            choices=choices,
            value=choices[0],
            interactive=True
        ),
        (
            "Depotvorschlag gefunden – "
            "bitte bestätigen."
        ),
    )


def confirm_depot(
    input_address,
    hits_json,
    selected
):

    hits = _json_load(
        hits_json,
        []
    )

    chosen = next(
        (
            hit
            for hit
            in hits
            if hit.get(
                "display_name"
            )
            == selected
        ),
        None
    )

    if not chosen:

        raise gr.Error(
            "Depotvorschlag auswählen."
        )

    state = {
        "address_input": input_address,
        "display_name":
            chosen["display_name"],
        "lat":
            chosen["lat"],
        "lon":
            chosen["lon"],
        "confidence":
            chosen.get(
                "confidence",
                0
            ),
        "provider":
            chosen.get(
                "provider",
                ""
            ),
    }

    return (
        json.dumps(
            state,
            ensure_ascii=False
        ),
# ============================================================
# FORECAST
# ============================================================

def calculate_all_forecasts(
    routes_json,
    depot_json,
    vehicle_table,
    start_time
):

    routes = _json_load(
        routes_json,
        []
    )

    depot = _json_load(
        depot_json,
        {}
    )

    vehicles = pd.DataFrame(
        vehicle_table
    )

    if not routes:

        raise gr.Error(
            "Zuerst Routen optimieren."
        )

    if not depot:

        raise gr.Error(
            "Depot fehlt."
        )

    if not TOMTOM_API_KEY:

        raise gr.Error(
            "TOMTOM_API_KEY fehlt im Hugging-Face-Space."
        )

    normalized_start = (
        normalize_start_time(
            start_time
        )
    )

    forecasts = {}
    summary_rows = []
    segment_table_rows = []
    debug = []
    logging_results = []
    vehicle_choices = []

    for route_index, route in enumerate(
        routes,
        1
    ):

        vehicle_id = str(
            route["vehicle_id"]
        )

        vehicle_choices.append(
            vehicle_id
        )

        vehicle_rows = vehicles[
            vehicles["vehicle_id"].astype(str)
            == vehicle_id
        ]

        if vehicle_rows.empty:

            params = {}

        else:

            params = (
                vehicle_parameters(
                    vehicle_rows.iloc[0]
                )
            )

        tour_id = (
            f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
            f"-{vehicle_id}"
        )

        try:

            forecast = (
                calculate_tour_forecast(
                    route=route,
                    depot=depot,
                    start_time=
                        normalized_start,
                    api_key=
                        TOMTOM_API_KEY,
                    timeout=
                        TIMEOUT,
                    tour_id=
                        tour_id,
                    forecast_version=
                        1,
                    vehicle_parameters=
                        params,
                )
            )

        except Exception as exc:

            debug.append({
                "vehicle_id":
                    vehicle_id,

                "status":
                    "forecast_failed",

                "error":
                    str(exc),
            })

            continue

        forecasts[
            vehicle_id
        ] = forecast

        summary = tour_summary(
            forecast
        )

        summary_rows.append(
            summary
        )

        rows = segment_rows(
            forecast
        )

        for row in rows:

            row[
                "vehicle_id"
            ] = vehicle_id

            row[
                "forecast_version"
            ] = forecast.get(
                "forecast_version"
            )

            segment_table_rows.append(
                row
            )

        try:

            log_result = log_forecast(
                forecast
            )

            logging_results.append({
                "vehicle_id":
                    vehicle_id,

                "result":
                    log_result,
            })

        except Exception as exc:

            logging_results.append({
                "vehicle_id":
                    vehicle_id,

                "error":
                    str(exc),
            })

        debug.append({
            "vehicle_id":
                vehicle_id,

            "tour_id":
                forecast.get(
                    "tour_id"
                ),

            "forecast_version":
                forecast.get(
                    "forecast_version"
                ),

            "segments":
                forecast.get(
                    "segments"
                ),
        })

    if not forecasts:

        raise gr.Error(
            "Kein Forecast konnte berechnet werden."
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    segments_df = pd.DataFrame(
        segment_table_rows
    )

    markdown = (
        build_forecast_markdown(
            summary_df
        )
    )

    first_vehicle = (
        vehicle_choices[0]
        if vehicle_choices
        else None
    )

    stop_choices = (
        forecast_stop_choices(
            forecasts.get(
                first_vehicle,
                {}
            )
        )
    )

    combined_debug = {
        "forecast": debug,
        "logging": logging_results,
        "logger_status":
            training_logger_status(),
    }

    return (
        summary_df,
        segments_df,
        markdown,

        json.dumps(
            forecasts,
            ensure_ascii=False,
            default=str
        ),

        json.dumps(
            combined_debug,
            ensure_ascii=False,
            indent=2,
            default=str
        ),

        gr.update(
            choices=
                vehicle_choices,
            value=
                first_vehicle,
            interactive=
                bool(
                    vehicle_choices
                )
        ),

        gr.update(
            choices=
                stop_choices,
            value=(
                stop_choices[0]
                if stop_choices
                else None
            ),
            interactive=
                bool(
                    stop_choices
                )
        ),

        (
            "✅ Segment-Forecast berechnet und "
            "Trainingslogger ausgeführt."
        ),
    )


# ============================================================
# FORECAST TEXT
# ============================================================

def build_forecast_markdown(
    summary_df
):

    if (
        summary_df is None
        or summary_df.empty
    ):

        return (
            "Noch kein Forecast vorhanden."
        )

    lines = [
        "## Dynamischer Tour-Forecast 4.9.5",
        "",
        (
            "Die Fahrzeit wird **zwischen jedem Stopp "
            "separat** berechnet. Servicezeiten verschieben "
            "die Abfahrtszeit des nächsten Segments."
        ),
        "",
    ]

    for _, row in summary_df.iterrows():

        ist_value = row.get(
            "ist_fahrzeit_min"
        )

        if pd.isna(
            ist_value
        ):
            ist_text = "noch offen"
        else:
            ist_text = (
                f"{ist_value} min"
            )

        lines += [
            (
                f"### 🚚 {row['vehicle_id']} "
                f"· Forecast V{row['forecast_version']}"
            ),

            (
                f"- Stopps: "
                f"**{row['stopps']}**"
            ),

            (
                f"- Distanz: "
                f"**{row['distanz_km']} km**"
            ),

            "",

            "**Fahrzeitvergleich**",

            (
                "- 1 · OSRM Basis: "
                f"**{row['osrm_basis_min']} min**"
            ),

            (
                "- 2 · TomTom Live: "
                f"**{row['tomtom_live_min']} min**"
            ),

            (
                "- 3 · TomTom Historisch: "
                f"**{row['tomtom_historisch_min']} min**"
            ),

            (
                "- 4 · Eigener Forecast: "
                f"**{row['forecast_fahrzeit_min']} min**"
            ),

            (
                "- 5 · IST: "
                f"**{ist_text}**"
            ),

            "",

            (
                "- Service geplant: "
                f"**{row['service_geplant_min']} min**"
            ),

            (
                "- Service aktuell verwendet: "
                f"**{row.get('service_verwendet_min', row['service_geplant_min'])} min**"
            ),

            (
                "- Forecast Gesamt: "
                f"**{row['forecast_gesamt_min']} min**"
            ),

            "",
        ]

    return "\n".join(
        lines
    )


# ============================================================
# REFORECAST STOPPS
# ============================================================

def forecast_stop_choices(
    forecast
):

    if not forecast:

        return []

    choices = []
    seen = set()

    for segment in (
        forecast.get(
            "segments"
        )
        or []
    ):

        stop_id = segment.get(
            "to_stop_id"
        )

        if (
            not stop_id
            or stop_id == "DEPOT"
            or stop_id in seen
        ):

            continue

        seen.add(
            stop_id
        )

        choices.append(
            str(
                stop_id
            )
        )

    return choices


def update_reforecast_stops(
    forecasts_json,
    vehicle_id
):

    forecasts = _json_load(
        forecasts_json,
        {}
    )

    forecast = forecasts.get(
        str(
            vehicle_id
        ),
        {}
    )

    choices = (
        forecast_stop_choices(
            forecast
        )
    )

    return gr.update(
        choices=choices,
        value=(
            choices[0]
            if choices
            else None
        ),
        interactive=bool(
            choices
        )
    )


# ============================================================
# REFORECAST
# ============================================================

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

    forecasts = _json_load(
        forecasts_json,
        {}
    )

    routes = _json_load(
        routes_json,
        []
    )

    depot = _json_load(
        depot_json,
        {}
    )

    vehicles = pd.DataFrame(
        vehicle_table
    )

    if not vehicle_id:

        raise gr.Error(
            "Bitte LKW auswählen."
        )

    if not stop_id:

        raise gr.Error(
            "Bitte Stopp auswählen."
        )

    previous = forecasts.get(
        str(
            vehicle_id
        )
    )

    if not previous:

        raise gr.Error(
            "Für diesen LKW existiert kein Forecast."
        )

    route = next(
        (
            route
            for route
            in routes
            if str(
                route.get(
                    "vehicle_id"
                )
            )
            == str(
                vehicle_id
            )
        ),
        None
    )

    if not route:

        raise gr.Error(
            "Route des LKW wurde nicht gefunden."
        )

    vehicle_rows = vehicles[
        vehicles["vehicle_id"].astype(str)
        == str(
            vehicle_id
        )
    ]

    if vehicle_rows.empty:

        params = {}

    else:

        params = vehicle_parameters(
            vehicle_rows.iloc[0]
        )

    service_seconds = None

    if (
        actual_service_min is not None
        and str(
            actual_service_min
        ).strip()
        != ""
    ):

        service_seconds = max(
            0,
            int(
                round(
                    float(
                        actual_service_min
                    )
                    * 60
                )
            )
        )

    departure_value = str(
        actual_departure
        or ""
    ).strip()

    if not departure_value:
        departure_value = None

    try:

        new_forecast = (
            recalculate_tour_from_stop(
                previous_forecast=
                    previous,

                route=
                    route,

                depot=
                    depot,

                from_stop_id=
                    str(
                        stop_id
                    ),

                new_departure_time=
                    departure_value,

                actual_service_s=
                    service_seconds,

                api_key=
                    TOMTOM_API_KEY,

                timeout=
                    TIMEOUT,

                vehicle_parameters=
                    params,
            )
        )

    except Exception as exc:

        raise gr.Error(
            "Neuberechnung fehlgeschlagen: "
            f"{type(exc).__name__}: {exc}"
        )

    forecasts[
        str(
            vehicle_id
        )
    ] = new_forecast

    try:

        log_result = log_forecast(
            new_forecast
        )

    except Exception as exc:

        log_result = {
            "error": str(exc)
        }

    summary_rows = []
    segment_table_rows = []

    for vid, forecast in forecasts.items():

        summary_rows.append(
            tour_summary(
                forecast
            )
        )

        rows = segment_rows(
            forecast
        )

        for row in rows:

            row[
                "vehicle_id"
            ] = vid

            row[
                "forecast_version"
            ] = forecast.get(
                "forecast_version"
            )

            segment_table_rows.append(
                row
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    segments_df = pd.DataFrame(
        segment_table_rows
    )

    debug = {
        "reforecast_vehicle":
            vehicle_id,

        "reforecast_from_stop":
            stop_id,

        "new_forecast_version":
            new_forecast.get(
                "forecast_version"
            ),

        "logging":
            log_result,

        "forecast":
            new_forecast,
    }

    return (
        summary_df,
        segments_df,

        build_forecast_markdown(
            summary_df
        ),

        json.dumps(
            forecasts,
            ensure_ascii=False,
            default=str
        ),

        json.dumps(
            debug,
            ensure_ascii=False,
            indent=2,
            default=str
        ),

        (
            f"✅ **{vehicle_id} ab Stopp {stop_id} "
            f"neu berechnet. Forecast-Version "
            f"{new_forecast.get('forecast_version')} gespeichert.**"
        ),
    )


# ============================================================
# LOGGER STATUS
# ============================================================

def logger_status_markdown():

    status = (
        training_logger_status()
    )

    if status.get(
        "remote_logging_ready"
    ):

        return (
            "🟢 **Trainingsspeicher bereit**  \n"
            f"Dataset: "
            f"`{status.get('hf_dataset_repo')}`"
        )

    problems = []

    if not status.get(
        "hf_token_configured"
    ):

        problems.append(
            "HF_TOKEN fehlt"
        )

    if not status.get(
        "dataset_configured"
    ):

        problems.append(
            "HF_DATASET_REPO fehlt"
        )

    return (
        "🟠 **Remote-Training-Logging noch nicht bereit.**  \n"
        + " · ".join(
            problems
        )
        + "  \nLokaler Fallback bleibt aktiv."
    )


# ============================================================
# CSS
# ============================================================

CSS = """
html, body {
    background: #0f1117 !important;
}

.gradio-container,
.gradio-container > .main {
    background: #0f1117 !important;
    color: #f3f4f6 !important;
}

.gradio-container {
    min-height: 100vh;
}

.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .prose h4,
.gradio-container label,
.gradio-container span {
    color: #f3f4f6;
}

.step-title {
    font-size: 1.35rem;
    font-weight: 750;
    margin-bottom: .35rem;
}

.process {
    padding: .8rem 1rem;
    border: 1px solid #374151;
    border-radius: 12px;
    margin: .3rem 0;
}

#tour-map iframe {
    width: 100% !important;
    min-height: 560px !important;
    height: 62vh !important;
    border-radius: 12px;
}

#tour-map {
    min-height: 560px;
}

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


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="Logistik Forecast 4.9.5",
    css=CSS
) as demo:

    gr.Markdown(
        "# Logistik Forecast 4.9.5\n"
        "**Adressen → Flotte → Cluster → OSRM → "
        "TomTom Segment-Forecast → Reforecast → ML-Dataset**"
    )

    # ========================================================
    # STATES
    # ========================================================

    orders_state = gr.State(
        "[]"
    )

    address_state = gr.State(
        "[]"
    )

    candidates_state = gr.State(
        "{}"
    )

    geo_state = gr.State(
        "[]"
    )

    depot_hits_state = gr.State(
        "[]"
    )

    depot_state = gr.State(
        ""
    )

    cluster_state = gr.State(
        "[]"
    )

    routes_state = gr.State(
        "[]"
    )

    forecasts_state = gr.State(
        "{}"
    )

    selected_order_state = gr.State(
        ""
    )

    selected_hits_state = gr.State(
        "[]"
    )

    # ========================================================
    # TABS
    # ========================================================

    with gr.Tabs():

        with gr.Tab(
            "1 · Aufträge & Adressen"
        ):

            gr.Markdown(
                '<div class="step-title">'
                '1. CSV importieren und Adressen prüfen'
                '</div>'
            )

            upload = gr.File(
                label="CSV-Datei",
                file_types=[
                    ".csv"
                ],
                type="filepath"
            )

            import_btn = gr.Button(
                "CSV importieren",
                variant="primary"
            )

            order_summary = gr.Markdown()

            orders_table = gr.Dataframe(
                label="Aufträge",
                interactive=False,
                wrap=True
            )

            geocode_btn = gr.Button(
                "Adressen automatisch prüfen"
            )

            address_status_md = gr.Markdown()

            address_table = gr.Dataframe(
                headers=[
                    "auftrag",
                    "kunde",
                    "adresse",
                    "status",
                    "treffer",
                    "sicherheit"
                ],
                label="Adressprüfung",
                interactive=False,
                wrap=True,
            )

            gr.Markdown(
                "### Nur unsichere Adressen"
            )

            review_info = gr.Markdown(
                "Noch keine Prüfung gestartet."
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

            finish_addresses_btn = gr.Button(
                "Adressprüfung abschließen"
            )

            finish_status = gr.Markdown()

        with gr.Tab(
            "2 · Depot, Flotte & Regionen"
        ):

            gr.Markdown(
                '<div class="step-title">'
                '2. Depot und Fahrzeugkapazität'
                '</div>'
            )

            depot_input = gr.Textbox(
                label="Depotadresse",
                placeholder=(
                    "z. B. Mercedesstraße 1, "
                    "70372 Stuttgart"
                )
            )

            depot_search_b