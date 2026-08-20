# src/training_logger.py
# Logistik Forecast 4.9.5
# Persistiert Forecast- und IST-Daten in einem Hugging-Face-Dataset.
# Bei Fehlern wird lokal in training_data/ weitergeschrieben.

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from huggingface_hub import HfApi, hf_hub_download


HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "").strip()
LOCAL_DATA_DIR = Path(
    os.getenv(
        "LOCAL_TRAINING_DATA_DIR",
        "training_data"
    )
)

FORECAST_RUNS_FILE = "forecasts/forecast_runs.jsonl"
SEGMENTS_FILE = "forecasts/forecast_segments.jsonl"
ACTUAL_TOURS_FILE = "actual/actual_tours.jsonl"
ACTUAL_SEGMENTS_FILE = "actual/actual_segments.jsonl"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, datetime):
        return value.replace(microsecond=0).isoformat()

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return str(value)


def _append_local_jsonl(
    relative_path: str,
    record: Dict[str, Any]
) -> str:
    target = LOCAL_DATA_DIR / relative_path
    target.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with target.open(
        "a",
        encoding="utf-8"
    ) as handle:
        handle.write(
            json.dumps(
                _json_safe(record),
                ensure_ascii=False
            )
        )
        handle.write("\n")

    return str(target)


def _download_existing_jsonl(
    path_in_repo: str
) -> List[str]:
    if not HF_DATASET_REPO:
        return []

    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=path_in_repo,
            repo_type="dataset",
            token=HF_TOKEN or None,
        )

        with open(
            downloaded_path,
            "r",
            encoding="utf-8"
        ) as handle:
            return [
                line.rstrip("\n")
                for line in handle
                if line.strip()
            ]

    except Exception:
        return []


def _append_hf_jsonl(
    path_in_repo: str,
    record: Dict[str, Any],
    commit_message: str
) -> Dict[str, Any]:
    if not HF_DATASET_REPO:
        raise RuntimeError("HF_DATASET_REPO fehlt.")

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN fehlt.")

    lines = _download_existing_jsonl(
        path_in_repo
    )

    lines.append(
        json.dumps(
            _json_safe(record),
            ensure_ascii=False
        )
    )

    api = HfApi(
        token=HF_TOKEN
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        local_file = (
            Path(temp_dir)
            / Path(path_in_repo).name
        )

        with local_file.open(
            "w",
            encoding="utf-8"
        ) as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")

        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=path_in_repo,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=commit_message,
        )

    return {
        "success": True,
        "storage": "huggingface",
        "repo": HF_DATASET_REPO,
        "path": path_in_repo,
        "records": len(lines),
    }


def append_record(
    path_in_repo: str,
    record: Dict[str, Any],
    commit_message: str
) -> Dict[str, Any]:
    prepared = dict(record)
    prepared.setdefault(
        "logged_at",
        _utc_now()
    )

    hf_error = None

    if HF_DATASET_REPO and HF_TOKEN:
        try:
            return _append_hf_jsonl(
                path_in_repo=path_in_repo,
                record=prepared,
                commit_message=commit_message,
            )
        except Exception as exc:
            hf_error = str(exc)

    local_path = _append_local_jsonl(
        relative_path=path_in_repo,
        record=prepared,
    )

    return {
        "success": True,
        "storage": "local_fallback",
        "path": local_path,
        "hf_error": hf_error,
    }


def _forecast_run_record(
    forecast: Dict[str, Any]
) -> Dict[str, Any]:
    segments = (
        forecast.get("segments")
        or []
    )

    traffic_delay_s = sum(
        _safe_int(
            segment.get(
                "tomtom_traffic_delay_s"
            )
        )
        for segment in segments
    )

    successful_tomtom_segments = sum(
        1
        for segment in segments
        if bool(
            segment.get(
                "tomtom_success"
            )
        )
    )

    failed_tomtom_segments = (
        len(segments)
        - successful_tomtom_segments
    )

    return {
        "tour_id":
            forecast.get("tour_id"),

        "forecast_version":
            _safe_int(
                forecast.get(
                    "forecast_version"
                ),
                1
            ),

        "vehicle_id":
            forecast.get("vehicle_id"),

        "started_at":
            forecast.get("started_at"),

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

        "actual_travel_s": (
            _safe_int(
                forecast.get(
                    "actual_travel_s"
                )
            )
            if forecast.get(
                "actual_travel_s"
            ) is not None
            else None
        ),

        "planned_service_s":
            _safe_int(
                forecast.get(
                    "planned_service_s"
                )
            ),

        "actual_service_s": (
            _safe_int(
                forecast.get(
                    "actual_service_s"
                )
            )
            if forecast.get(
                "actual_service_s"
            ) is not None
            else None
        ),

        "tomtom_traffic_delay_s":
            traffic_delay_s,

        "tomtom_segments_success":
            successful_tomtom_segments,

        "tomtom_segments_failed":
            failed_tomtom_segments,

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


def _segment_record(
    forecast: Dict[str, Any],
    segment: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "tour_id":
            forecast.get("tour_id"),

        "forecast_version":
            _safe_int(
                forecast.get(
                    "forecast_version"
                ),
                1
            ),

        "vehicle_id":
            forecast.get("vehicle_id"),

        "segment_id":
            _safe_int(
                segment.get(
                    "segment_id"
                )
            ),

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

        "departure_time":
            segment.get(
                "departure_time"
            ),

        "arrival_time_forecast":
            segment.get(
                "arrival_time_forecast"
            ),

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

        "actual_travel_s": (
            _safe_int(
                segment.get(
                    "actual_travel_s"
                )
            )
            if segment.get(
                "actual_travel_s"
            ) is not None
            else None
        ),

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

        "planned_service_s":
            _safe_int(
                segment.get(
                    "planned_service_s"
                )
            ),

        "actual_service_s": (
            _safe_int(
                segment.get(
                    "actual_service_s"
                )
            )
            if segment.get(
                "actual_service_s"
            ) is not None
            else None
        ),

        "incident_count":
            _safe_int(
                segment.get(
                    "incident_count"
                )
            ),

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


def log_forecast(
    forecast: Dict[str, Any]
) -> Dict[str, Any]:
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

    tour_record = _forecast_run_record(
        forecast
    )

    tour_result = append_record(
        path_in_repo=FORECAST_RUNS_FILE,
        record=tour_record,
        commit_message=(
            f"Forecast {tour_id} "
            f"Version {version}"
        ),
    )

    segment_results = []

    for segment in (
        forecast.get(
            "segments"
        )
        or []
    ):
        segment_record = _segment_record(
            forecast=forecast,
            segment=segment,
        )

        result = append_record(
            path_in_repo=SEGMENTS_FILE,
            record=segment_record,
            commit_message=(
                f"Segment Forecast "
                f"{tour_id} "
                f"V{version} "
                f"S{segment_record['segment_id']}"
            ),
        )

        segment_results.append(
            result
        )

    return {
        "success": True,
        "tour_id": tour_id,
        "forecast_version": version,
        "tour_storage": tour_result,
        "segments_written": len(
            segment_results
        ),
        "segment_storage": segment_results,
    }


def log_actual_tour(
    actual: Dict[str, Any]
) -> Dict[str, Any]:
    if not actual:
        raise ValueError(
            "IST-Daten sind leer."
        )

    tour_id = str(
        actual.get(
            "tour_id"
        )
        or ""
    )

    if not tour_id:
        raise ValueError(
            "IST-Daten enthalten keine tour_id."
        )

    record = {
        "tour_id":
            tour_id,

        "vehicle_id":
            actual.get(
                "vehicle_id"
            ),

        "actual_departure":
            actual.get(
                "actual_departure"
            ),

        "actual_arrival":
            actual.get(
                "actual_arrival"
            ),

        "actual_travel_s": (
            _safe_int(
                actual.get(
                    "actual_travel_s"
                )
            )
            if actual.get(
                "actual_travel_s"
            ) is not None
            else None
        ),

        "actual_service_s": (
            _safe_int(
                actual.get(
                    "actual_service_s"
                )
            )
            if actual.get(
                "actual_service_s"
            ) is not None
            else None
        ),

        "actual_distance_m": (
            _safe_float(
                actual.get(
                    "actual_distance_m"
                )
            )
            if actual.get(
                "actual_distance_m"
            ) is not None
            else None
        ),

        "logged_at":
            _utc_now(),
    }

    return append_record(
        path_in_repo=ACTUAL_TOURS_FILE,
        record=record,
        commit_message=(
            f"IST Tour {tour_id}"
        ),
    )


def log_actual_segment(
    actual: Dict[str, Any]
) -> Dict[str, Any]:
    if not actual:
        raise ValueError(
            "IST-Segmentdaten sind leer."
        )

    tour_id = str(
        actual.get(
            "tour_id"
        )
        or ""
    )

    if not tour_id:
        raise ValueError(
            "IST-Segment enthält keine tour_id."
        )

    segment_id = _safe_int(
        actual.get(
            "segment_id"
        )
    )

    record = {
        "tour_id":
            tour_id,

        "vehicle_id":
            actual.get(
                "vehicle_id"
            ),

        "segment_id":
            segment_id,

        "from_stop_id":
            actual.get(
                "from_stop_id"
            ),

        "to_stop_id":
            actual.get(
                "to_stop_id"
            ),

        "actual_departure":
            actual.get(
                "actual_departure"
            ),

        "actual_arrival":
            actual.get(
                "actual_arrival"
            ),

        "actual_travel_s": (
            _safe_int(
                actual.get(
                    "actual_travel_s"
                )
            )
            if actual.get(
                "actual_travel_s"
            ) is not None
            else None
        ),

        "actual_service_s": (
            _safe_int(
                actual.get(
                    "actual_service_s"
                )
            )
            if actual.get(
                "actual_service_s"
            ) is not None
            else None
        ),

        "actual_distance_m": (
            _safe_float(
                actual.get(
                    "actual_distance_m"
                )
            )
            if actual.get(
                "actual_distance_m"
            ) is not None
            else None
        ),

        "logged_at":
            _utc_now(),
    }

    return append_record(
        path_in_repo=ACTUAL_SEGMENTS_FILE,
        record=record,
        commit_message=(
            f"IST Segment "
            f"{tour_id} "
            f"S{segment_id}"
        ),
    )


def training_logger_status() -> Dict[str, Any]:
    return {
        "hf_dataset_repo":
            HF_DATASET_REPO
            or None,

        "hf_token_configured":
            bool(HF_TOKEN),

        "dataset_configured":
            bool(HF_DATASET_REPO),

        "remote_logging_ready":
            bool(
                HF_TOKEN
                and HF_DATASET_REPO
            ),

        "local_fallback_directory":
            str(LOCAL_DATA_DIR),

        "files": {
            "forecast_runs":
                FORECAST_RUNS_FILE,

            "forecast_segments":
                SEGMENTS_FILE,

            "actual_tours":
                ACTUAL_TOURS_FILE,

            "actual_segments":
                ACTUAL_SEGMENTS_FILE,
        },
    }


if __name__ == "__main__":
    print(
        json.dumps(
            training_logger_status(),
            indent=2,
            ensure_ascii=False
        )
    )
