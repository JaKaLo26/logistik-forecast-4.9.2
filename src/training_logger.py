# src/training_logger.py

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import HfApi, hf_hub_download


# ============================================================
# KONFIGURATION
# ============================================================

HF_TOKEN = os.getenv(
    "HF_TOKEN",
    ""
).strip()

HF_DATASET_REPO = os.getenv(
    "HF_DATASET_REPO",
    ""
).strip()

LOCAL_DATA_DIR = Path(
    os.getenv(
        "LOCAL_TRAINING_DATA_DIR",
        "training_data"
    )
)


# ============================================================
# ZIELDATEIEN IM HF DATASET
# ============================================================

FORECAST_RUNS_FILE = (
    "forecasts/forecast_runs.jsonl"
)

SEGMENTS_FILE = (
    "forecasts/forecast_segments.jsonl"
)

ACTUAL_TOURS_FILE = (
    "actual/actual_tours.jsonl"
)

ACTUAL_SEGMENTS_FILE = (
    "actual/actual_segments.jsonl"
)


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def _utc_now() -> str:

    return (
        datetime.now(
            timezone.utc
        )
        .replace(
            microsecond=0
        )
        .isoformat()
    )


def _safe_int(
    value: Any,
    default: int = 0
) -> int:

    try:

        return int(
            round(
                float(value)
            )
        )

    except (
        TypeError,
        ValueError
    ):

        return default


def _safe_float(
    value: Any,
    default: float = 0.0
) -> float:

    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return default


def _json_safe(
    value: Any
) -> Any:
    """
    Wandelt Werte in JSON-kompatible Typen um.
    """

    if value is None:

        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool
        )
    ):

        return value

    if isinstance(
        value,
        datetime
    ):

        return (
            value
            .replace(
                microsecond=0
            )
            .isoformat()
        )

    if isinstance(
        value,
        dict
    ):

        return {
            str(k):
                _json_safe(v)
            for k, v
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple
        )
    ):

        return [
            _json_safe(v)
            for v
            in value
        ]

    return str(
        value
    )


# ============================================================
# LOKALE FALLBACK-SPEICHERUNG
# ============================================================

def _append_local_jsonl(
    relative_path: str,
    record: Dict[str, Any]
) -> str:

    target = (
        LOCAL_DATA_DIR
        / relative_path
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with target.open(
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                _json_safe(
                    record
                ),
                ensure_ascii=False
            )
        )

        file.write(
            "\n"
        )

    return str(
        target
    )


# ============================================================
# HF DATASET HILFSFUNKTION
# ============================================================

def _download_existing_jsonl(
    path_in_repo: str
) -> List[str]:
    """
    Lädt eine bereits vorhandene JSONL-Datei
    aus dem Dataset.

    Existiert sie noch nicht, wird [] zurückgegeben.
    """

    if not HF_DATASET_REPO:

        return []

    try:

        downloaded_path = (
            hf_hub_download(
                repo_id=
                    HF_DATASET_REPO,

                filename=
                    path_in_repo,

                repo_type=
                    "dataset",

                token=
                    HF_TOKEN
                    or None,
            )
        )

        with open(
            downloaded_path,
            "r",
            encoding="utf-8"
        ) as file:

            return [
                line.rstrip(
                    "\n"
                )
                for line
                in file
                if line.strip()
            ]

    except Exception:

        return []


def _append_hf_jsonl(
    path_in_repo: str,
    record: Dict[str, Any],
    commit_message: str
) -> Dict[str, Any]:
    """
    Bestehende JSONL laden,
    neuen Datensatz anhängen,
    komplette Datei wieder hochladen.

    Für die aktuelle Projektgröße ausreichend.

    Bei sehr großen Datenmengen wechseln wir später
    auf Parquet-Shards / Batch-Schreiben.
    """

    if not HF_DATASET_REPO:

        raise RuntimeError(
            "HF_DATASET_REPO fehlt."
        )

    if not HF_TOKEN:

        raise RuntimeError(
            "HF_TOKEN fehlt."
        )

    lines = (
        _download_existing_jsonl(
            path_in_repo
        )
    )

    new_line = json.dumps(
        _json_safe(
            record
        ),
        ensure_ascii=False
    )

    lines.append(
        new_line
    )

    api = HfApi(
        token=HF_TOKEN
    )

    with tempfile.TemporaryDirectory() as tmp:

        local_file = (
            Path(tmp)
            / Path(
                path_in_repo
            ).name
        )

        with local_file.open(
            "w",
            encoding="utf-8"
        ) as file:

            for line in lines:

                file.write(
                    line
                )

                file.write(
                    "\n"
                )

        api.upload_file(
            path_or_fileobj=
                str(
                    local_file
                ),

            path_in_repo=
                path_in_repo,

            repo_id=
                HF_DATASET_REPO,

            repo_type=
                "dataset",

            commit_message=
                commit_message,
        )

    return {
        "success":
            True,

        "storage":
            "huggingface",

        "repo":
            HF_DATASET_REPO,

        "path":
            path_in_repo,

        "records":
            len(
                lines
            ),
    }


# ============================================================
# ROBUSTER APPEND
# ============================================================

def append_record(
    path_in_repo: str,
    record: Dict[str, Any],
    commit_message: str
) -> Dict[str, Any]:
    """
    Versucht zuerst HF Dataset.

    Falls HF nicht funktioniert:
    lokale JSONL als Fallback.
    """

    prepared = dict(
        record
    )

    prepared.setdefault(
        "logged_at",
        _utc_now()
    )

    hf_error = None

    if (
        HF_DATASET_REPO
        and HF_TOKEN
    ):

        try:

            return _append_hf_jsonl(
                path_in_repo=
                    path_in_repo,

                record=
                    prepared,

                commit_message=
                    commit_message
            )

        except Exception as exc:

            hf_error = str(
                exc
            )

    local_path = (
        _append_local_jsonl(
            relative_path=
                path_in_repo,

            record=
                prepared
        )
    )

    return {
        "success":
            True,

        "storage":
            "local_fallback",

        "path":
            local_path,

        "hf_error":
            hf_error,
    }


# ============================================================
# TOUR-FORECAST AUFBEREITEN
# ============================================================

def _forecast_run_record(
    forecast: Dict[str, Any]
) -> Dict[str, Any]:

    segments = (
        forecast.get(
            "segments"
        )
        or []
    )

    traffic_delay_s = sum(
        _safe_int(
            segment.get(
                "tomtom_traffic_delay_s"
            )
        )
        for segment
        in segments
    )

    successful_tomtom_segments = sum(
        1
        for segment
        in segments
        if bool(
            segment.get(
                "tomtom_success"
            )
        )
    )

    failed_tomtom_segments = (
        len(
            segments
        )
        - successful_tomtom_segments
    )

    return {
        "tour_id":
            forecast.get(
                "tour_id"
            ),

        "forecast_version":
            _safe_int(
                forecast.get(
                    "forecast_version"
                ),
                1
            ),

        "vehicle_id":
            forecast.get(
                "vehicle_id"
            ),

        "started_at":
            forecast.get(
                "started_at"
            ),

        "finished_at_forecast":
            forecast.get(
                "finished_at_forecast"
            ),

        "stop_count":
            _safe_int(
                forecast.get(
                    "stop_count"
                )
            ),

        "segment_count":
            _safe_int(
                forecast.get(
                    "segment_count"
                )
            ),

        "distance_m":
            _safe_float(
                forecast.get(
                    "total_distance_m"
                )
            ),

        # ====================================================
        # DIE 5 ZEITKATEGORIEN
        # ====================================================

        "osrm_baseline_s":
            _safe_int(
                forecast.get(
                    "osrm_baseline_s"
                )
            ),

        "tomtom_no_traffic_s":
            _safe_int(
                forecast.get(
                    "tomtom_no_traffic_s"
                )
            ),

        "tomtom_historic_s":
            _safe_int(
                forecast.get(
                    "tomtom_historic_s"
                )
            ),

        "tomtom_live_s":
            _safe_int(
                forecast.get(
                    "tomtom_live_s"
                )
            ),

        "model_forecast_s":
            _safe_int(
                forecast.get(
                    "model_forecast_s"
                )
            ),

        "actual_travel_s":
            (
                _safe_int(
                    forecast.get(
                        "actual_travel_s"
                    )
                )
                if forecast.get(
                    "actual_travel_s"
                )
                is not None
                else None
            ),

        # ====================================================
        # SERVICE
        # ====================================================

        "planned_service_s":
            _safe_int(
                forecast.get(
                    "planned_service_s"
                )
            ),

        "actual_service_s":
            (
                _safe_int(
                    forecast.get(
                        "actual_service_s"
                    )
                )
                if forecast.get(
                    "actual_service_s"
                )
                is not None
                else None
            ),

        # ====================================================
        # VERKEHR
        # ====================================================

        "tomtom_traffic_delay_s":
            traffic_delay_s,

        "tomtom_segments_success":
            successful_tomtom_segments,

        "tomtom_segments_failed":
            failed_tomtom_segments,

        # ====================================================
        # REFORECAST
        # ====================================================

        "recalculated":
            bool(
                forecast.get(
                    "recalculated"
                )
            ),

        "recalculation_from_stop":
            forecast.get(
                "recalculation_from_stop"
            ),

        "logged_at":
            _utc_now(),
    }


# ============================================================
# SEGMENT AUFBEREITEN
# ============================================================

def _segment_record(
    forecast: Dict[str, Any],
    segment: Dict[str, Any]
) -> Dict[str, Any]:

    return {
        "tour_id":
            forecast.get(
                "tour_id"
            ),

        "forecast_version":
            _safe_int(
                forecast.get(
                    "forecast_version"
                ),
                1
            ),

        "vehicle_id":
            forecast.get(
                "vehicle_id"
            ),

        "segment_id":
            _safe_int(
                segment.get(
                    "segment_id"
                )
            ),

        # ====================================================
        # VON / NACH
        # ====================================================

        "from_stop_id":
            segment.get(
                "from_stop_id"
            ),

        "from_stop_name":
            segment.get(
                "from_stop_name"
            ),

        "to_stop_id":
            segment.get(
                "to_stop_id"
            ),

        "to_stop_name":
            segment.get(
                "to_stop_name"
            ),

        "from_lat":
            _safe_float(
                segment.get(
                    "from_lat"
                )
            ),

        "from_lon":
            _safe_float(
                segment.get(
                    "from_lon"
                )
            ),

        "to_lat":
            _safe_float(
                segment.get(
                    "to_lat"
                )
            ),

        "to_lon":
            _safe_float(
                segment.get(
                    "to_lon"
                )
            ),

        # ====================================================
        # ZEITPUNKTE
        # ====================================================

        "departure_time":
            segment.get(
                "departure_time"
            ),

        "arrival_time_forecast":
            segment.get(
                "arrival_time_forecast"
            ),

        # ====================================================
        # 5 ZEITKATEGORIEN
        # ====================================================

        "osrm_baseline_s":
            _safe_int(
                segment.get(
                    "osrm_baseline_s"
                )
            ),

        "tomtom_no_traffic_s":
            _safe_int(
                segment.get(
                    "tomtom_no_traffic_s"
                )
            ),

        "tomtom_historic_s":
            _safe_int(
                segment.get(
                    "tomtom_historic_s"
                )
            ),

        "tomtom_live_s":
            _safe_int(
                segment.get(
                    "tomtom_live_s"
                )
            ),

        "model_forecast_s":
            _safe_int(
                segment.get(
                    "model_forecast_s"
                )
            ),

        "actual_travel_s":
            (
                _safe_int(
                    segment.get(
                        "actual_travel_s"
                    )
                )
                if segment.get(
                    "actual_travel_s"
                )
                is not None
                else None
            ),

        # ====================================================
        # TOMTOM
        # ====================================================

        "tomtom_distance_m":
            _safe_float(
                segment.get(
                    "tomtom_distance_m"
                )
            ),

        "tomtom_traffic_delay_s":
            _safe_int(
                segment.get(
                    "tomtom_traffic_delay_s"
                )
            ),

        "data_confidence":
            _safe_float(
                segment.get(
                    "data_confidence"
                )
            ),

        "tomtom_success":
            bool(
                segment.get(
                    "tomtom_success"
                )
            ),

        "tomtom_error":
            segment.get(
                "tomtom_error"
            ),

        # ====================================================
        # SERVICE
        # ====================================================

        "planned_service_s":
            _safe_int(
                segment.get(
                    "planned_service_s"
                )
            ),

        "actual_service_s":
            (
                _safe_int(
                    segment.get(
                        "actual_service_s"
                    )
                )
                if segment.get(
                    "actual_service_s"
                )
                is not None
                else None
            ),

        # ====================================================
        # INCIDENTS
        # ====================================================

        "incident_count":
            _safe_int(
                segment.get(
                    "incident_count"
                )
            ),

        # ====================================================
        # REFORECAST
        # ====================================================

        "recalculated":
            bool(
                segment.get(
                    "recalculated"
                )
            ),

        "recalculation_from_stop":
            segment.get(
                "recalculation_from_stop"
            ),

        "logged_at":
            _utc_now(),
    }


# ============================================================
# FORECAST KOMPLETT SPEICHERN
# ============================================================

def log_forecast(
    forecast: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Speichert:

    1. eine Zeile pro Tour-Forecast
    2. eine Zeile pro Segment

    Jede Neuberechnung erhält dieselbe tour_id,
    aber eine neue forecast_version.
    """

    if not forecast:

        raise ValueError(
            "Forecast ist leer."
        )

    tour_id = str(
        forecast.get(
            "tour_id"
        )
        or ""
    )

    if not tour_id:

        raise ValueError(
            "Forecast enthält keine tour_id."
        )

    version = _safe_int(
        forecast.get(
            "forecast_version"
        ),
        1
    )

    # ========================================================
    # TOUR SPEICHERN
    # ========================================================

    tour_record = (
        _forecast_run_record(
            forecast
        )
    )

    tour_result = (
        append_record(
            path_in_repo=
                FORECAST_RUNS_FILE,

            record=
                tour_record,

            commit_message=(
                f"Forecast {tour_id} "
                f"Version {version}"
            ),
        )
    )

    # ========================================================
    # SEGMENTE SPEICHERN
    # ========================================================

    segment_results = []

    for segment in (
        forecast.get(
            "segments"
        )
        or []
    ):

        segment_record = (
            _segment_record(
                forecast=
                    forecast,

                segment=
                    segment
            )
        )

        result = (
            append_record(
                path_in_repo=
                    SEGMENTS_FILE,

          