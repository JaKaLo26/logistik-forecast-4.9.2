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


# ============================================================
# UMGEBUNG
# ============================================================

load_dotenv()


# ============================================================
# PROJEKT-IMPORTS
# ============================================================

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
    """
    ZeroGPU-Kompatibilitätsfunktion.

    Routing, Forecast und TomTom laufen auf CPU.
    """
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
# CSV SPALTENNAMEN
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

        "kundenname":
            "kunde",

        "straße":
            "strasse",

        "str.":
            "strasse",

        "gewicht":
            "warengewicht_kg",

        "gewicht_kg":
            "warengewicht_kg",

        "servicezeit":
            "service_min",

        "servicezeit_min":
            "service_min",

        "entladezeit":
            "service_min",

        "entladezeit_min":
            "service_min",

        "palettenanzahl":
            "paletten",

        "anzahl_paletten":
            "paletten",

        "auftragsnummer":
            "auftrag",

        "auftrag_id":
            "auftrag",

        "postleitzahl":
            "plz",

        "stadt":
            "ort",
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

            "vehicle_id":
                f"LKW-K{i:02d}",

            **SMALL,

            "color":
                COLORS[
                    color_idx
                    % len(COLORS)
                ],

            "available":
                True,
        })

        color_idx += 1

    for i in range(
        1,
        int(n_large) + 1
    ):

        rows.append({

            "vehicle_id":
                f"LKW-G{i:02d}",

            **LARGE,

            "color":
                COLORS[
                    color_idx
                    % len(COLORS)
                ],

            "available":
                True,
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
# FAHRZEUGPARAMETER FÜR TOMTOM
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

    if (
        "40"
        in vehicle_class
    ):

        return {

            "vehicle_weight_kg":
                40000,

            "vehicle_height_m":
                4.0,

            "vehicle_width_m":
                2.55,

            "vehicle_length_m":
                16.5,

            "vehicle_max_speed_kmh":
                80,

            "vehicle_commercial":
                True,
        }

    return {

        "vehicle_weight_kg":
            14000,

        "vehicle_height_m":
            4.0,

        "vehicle_width_m":
            2.55,

        "vehicle_length_m":
            10.0,

        "vehicle_max_speed_kmh":
            80,

        "vehicle_commercial":
            True,
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

            "auftrag":
                oid,

            "kunde":
                str(
                    row["kunde"]
                ),

            "eingabe":
                str(
                    row["adresse"]
                ),

            "treffer":
                (
                    best.get(
                        "display_name",
                        ""
                    )
                    if best
                    else ""
                ),

            "confidence":
                confidence,

            "status":
                status,

            "lat":
                (
                    best.get(
                        "lat"
                    )
                    if best
                    else None
                ),

            "lon":
                (
                    best.get(
                        "lon"
                    )
                    if best
                    else None
                ),

            "provider":
                (
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
            full[
                "auftrag"
            ].astype(str),

        "kunde":
            full[
                "kunde"
            ].astype(str),

        "adresse":
            full[
                "eingabe"
            ].astype(str),

        "status":
            full[
                "status"
            ].astype(str),

        "treffer":
            (
                full[
                    "treffer"
                ]
                .fillna("")
                .astype(str)
            ),

        "sicherheit":
            (
                pd.to_numeric(
                    full[
                        "confidence"
                    ],
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
            full[
                "status"
            ]
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

        full[
            "status"
        ].astype(str)

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

            choices=
                choices,

            value=(
                choices[0]
                if choices
                else None
            ),

            interactive=
                bool(
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
            full[
                "auftrag"
            ].astype(str)
            == str(
                oid
            )
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

                    "display_name":
                        selected,

                    "lat":
                        current["lat"],

                    "lon":
                        current["lon"],

                    "confidence":
                        current[
                            "confidence"
                        ],

                    "provider":
                        current[
                            "provider"
                        ],
                }

    if (
        chosen is None
        or chosen.get(
            "lat"
        )
        is None
        or chosen.get(
            "lon"
        )
        is None
    ):

        raise gr.Error(
            "Vorschlag konnte nicht übernommen werden."
        )

    mask = (
        full[
            "auftrag"
        ].astype(str)
        == str(
            oid
        )
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
            next_row[
                "auftrag"
            ]
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

            choices=
                choices,

            value=(
                choices[0]
                if choices
                else None
            ),

            interactive=
                bool(
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
        full[
            "status"
        ]
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

        on=
            "auftrag",

        how=
            "left",
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

        