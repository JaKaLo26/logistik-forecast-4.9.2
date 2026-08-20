# src/forecast.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.tomtom_routing import TomTomRoutingProvider


# ============================================================
# DATENMODELLE
# ============================================================

@dataclass
class SegmentForecast:

    tour_id: str
    forecast_version: int
    vehicle_id: str

    segment_id: int

    from_stop_id: str
    from_stop_name: str

    to_stop_id: str
    to_stop_name: str

    from_lat: float
    from_lon: float

    to_lat: float
    to_lon: float

    departure_time: str
    arrival_time_forecast: str

    osrm_baseline_s: int

    tomtom_no_traffic_s: int
    tomtom_historic_s: int
    tomtom_live_s: int

    model_forecast_s: int

    actual_travel_s: Optional[int]

    tomtom_traffic_delay_s: int
    tomtom_distance_m: float

    planned_service_s: int
    actual_service_s: Optional[int]

    incident_count: int

    data_confidence: float

    recalculated: bool
    recalculation_from_stop: Optional[str]

    tomtom_success: bool
    tomtom_error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TourForecast:

    tour_id: str
    forecast_version: int
    vehicle_id: str

    started_at: str
    finished_at_forecast: str

    osrm_baseline_s: int
    tomtom_no_traffic_s: int
    tomtom_historic_s: int
    tomtom_live_s: int
    model_forecast_s: int

    actual_travel_s: Optional[int]

    planned_service_s: int
    actual_service_s: Optional[int]

    total_distance_m: float

    segment_count: int
    stop_count: int

    recalculated: bool
    recalculation_from_stop: Optional[str]

    segments: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

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
        return float(value)

    except (
        TypeError,
        ValueError
    ):
        return default


def _clamp(
    value: float,
    minimum: float,
    maximum: float
) -> float:

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


def _parse_datetime(
    value: Any
) -> datetime:

    if isinstance(
        value,
        datetime
    ):

        dt = value

    else:

        text = str(
            value or ""
        ).strip()

        if not text:

            dt = datetime.now(
                timezone.utc
            )

        else:

            text = text.replace(
                "Z",
                "+00:00"
            )

            dt = datetime.fromisoformat(
                text
            )

    if dt.tzinfo is None:

        dt = dt.replace(
            tzinfo=timezone.utc
        )

    return dt


def _iso(
    value: datetime
) -> str:

    return (
        value
        .replace(
            microsecond=0
        )
        .isoformat()
    )


def _stop_id(
    stop: Dict[str, Any],
    fallback: str
) -> str:

    for key in (
        "auftrag",
        "stop_id",
        "id",
        "kunde"
    ):

        value = stop.get(
            key
        )

        if value not in (
            None,
            ""
        ):

            return str(
                value
            )

    return fallback


def _stop_name(
    stop: Dict[str, Any],
    fallback: str
) -> str:

    for key in (
        "kunde",
        "name",
        "auftrag"
    ):

        value = stop.get(
            key
        )

        if value not in (
            None,
            ""
        ):

            return str(
                value
            )

    return fallback


def _stop_coordinate(
    stop: Dict[str, Any]
) -> Tuple[float, float]:

    return (
        float(
            stop["lat"]
        ),
        float(
            stop["lon"]
        )
    )


def _planned_service_s(
    stop: Dict[str, Any]
) -> int:

    if (
        stop.get(
            "planned_service_s"
        )
        is not None
    ):

        return max(
            0,
            _safe_int(
                stop.get(
                    "planned_service_s"
                )
            )
        )

    if (
        stop.get(
            "service_s"
        )
        is not None
    ):

        return max(
            0,
            _safe_int(
                stop.get(
                    "service_s"
                )
            )
        )

    return max(
        0,
        _safe_int(
            stop.get(
                "service_min",
                0
            )
            * 60
        )
    )


def _actual_service_s(
    stop: Dict[str, Any]
) -> Optional[int]:

    if (
        stop.get(
            "actual_service_s"
        )
        is not None
    ):

        return max(
            0,
            _safe_int(
                stop.get(
                    "actual_service_s"
                )
            )
        )

    if (
        stop.get(
            "actual_service_min"
        )
        is not None
    ):

        return max(
            0,
            _safe_int(
                stop.get(
                    "actual_service_min"
                )
                * 60
            )
        )

    return None


# ============================================================
# EIGENER FORECAST
# ============================================================

def calculate_model_forecast_s(
    osrm_baseline_s: int,
    tomtom_historic_s: int,
    tomtom_live_s: int,
    traffic_delay_s: int = 0,
    confidence: float = 1.0,
) -> int:
    """
    Vorläufige Forecast-Logik vor dem eigentlichen ML-Modell.

    Ziel:
    - Live-Verkehr stark gewichten
    - Historische Verkehrslage stabilisierend verwenden
    - OSRM als Baseline/Fallback behalten

    Später wird diese Funktion durch das trainierte Modell ersetzt.
    """

    osrm_baseline_s = max(
        0,
        _safe_int(
            osrm_baseline_s
        )
    )

    historic_s = max(
        0,
        _safe_int(
            tomtom_historic_s
        )
    )

    live_s = max(
        0,
        _safe_int(
            tomtom_live_s
        )
    )

    traffic_delay_s = max(
        0,
        _safe_int(
            traffic_delay_s
        )
    )

    confidence = _clamp(
        _safe_float(
            confidence,
            1.0
        ),
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # FALLBACKS
    # --------------------------------------------------------

    if live_s <= 0:

        if historic_s > 0:
            live_s = historic_s

        else:
            live_s = osrm_baseline_s

    if historic_s <= 0:
        historic_s = (
            osrm_baseline_s
            or live_s
        )

    if osrm_baseline_s <= 0:
        osrm_baseline_s = (
            historic_s
            or live_s
        )

    # --------------------------------------------------------
    # HEURISTISCHER FORECAST
    #
    # Live = wichtigster Faktor
    # Historie = Stabilisierung
    # OSRM = Basisreferenz
    # --------------------------------------------------------

    live_weight = (
        0.50
        + (
            0.20
            * confidence
        )
    )

    historic_weight = 0.20

    osrm_weight = (
        1.0
        - live_weight
        - historic_weight
    )

    forecast_s = (
        live_s
        * live_weight
        +
        historic_s
        * historic_weight
        +
        osrm_baseline_s
        * osrm_weight
    )

    # TomTom Live sollte grundsätzlich bereits Delay enthalten.
    # Deshalb wird traffic_delay_s NICHT zusätzlich vollständig addiert.
    #
    # Nur eine kleine Sicherheitskorrektur bei deutlichem Delay.
    if traffic_delay_s > 0:

        forecast_s += (
            traffic_delay_s
            * 0.10
        )

    return max(
        0,
        int(
            round(
                forecast_s
            )
        )
    )


# ============================================================
# OSRM LEG-DATEN
# ============================================================

def extract_osrm_leg_baselines(
    route: Dict[str, Any],
    expected_segments: int
) -> List[int]:
    """
    Extrahiert OSRM-Leg-Dauern.

    Erwartete Segmente bei Rundtour:

        Depot -> Stopp 1
        Stopp 1 -> Stopp 2
        ...
        letzter Stopp -> Depot
    """

    legs = (
        route.get(
            "legs"
        )
        or []
    )

    durations = []

    for leg in legs:

        durations.append(
            max(
                0,
                _safe_int(
                    leg.get(
                        "duration"
                    )
                )
            )
        )

    if (
        len(durations)
        == expected_segments
    ):

        return durations

    # --------------------------------------------------------
    # FALLBACK:
    # Gesamtzeit gleichmäßig verteilen
    # --------------------------------------------------------

    total_duration = max(
        0,
        _safe_int(
            route.get(
                "duration_s"
            )
        )
    )

    if expected_segments <= 0:

        return []

    if total_duration <= 0:

        return [
            0
            for _
            in range(
                expected_segments
            )
        ]

    average = (
        total_duration
        / expected_segments
    )

    result = [
        int(
            round(
                average
            )
        )
        for _
        in range(
            expected_segments
        )
    ]

    # Rundungsdifferenz korrigieren.
    difference = (
        total_duration
        - sum(
            result
        )
    )

    if result:

        result[-1] += difference

    return result


# ============================================================
# SEGMENT-FORECAST
# ============================================================

def calculate_segment_forecast(
    *,
    tour_id: str,
    forecast_version: int,
    vehicle_id: str,
    segment_id: int,

    origin_stop: Dict[str, Any],
    destination_stop: Dict[str, Any],

    departure_time: datetime,

    osrm_baseline_s: int,

    provider: TomTomRoutingProvider,

    vehicle_parameters: Optional[
        Dict[str, Any]
    ] = None,

    recalculated: bool = False,
    recalculation_from_stop: Optional[str] = None,

    actual_travel_s: Optional[int] = None,

) -> SegmentForecast:

    vehicle_parameters = (
        vehicle_parameters
        or {}
    )

    origin_coord = (
        _stop_coordinate(
            origin_stop
        )
    )

    destination_coord = (
        _stop_coordinate(
            destination_stop
        )
    )

    tomtom = (
        provider.calculate_segment(
            origin=
                origin_coord,

            destination=
                destination_coord,

            depart_at=
                departure_time,

            **vehicle_parameters
        )
    )

    # --------------------------------------------------------
    # TOMTOM FEHLERFALL
    # --------------------------------------------------------

    if tomtom.success:

        no_traffic_s = max(
            0,
            tomtom.tomtom_no_traffic_s
        )

        historic_s = max(
            0,
            tomtom.tomtom_historic_s
        )

        live_s = max(
            0,
            tomtom.tomtom_live_s
        )

        traffic_delay_s = max(
            0,
            tomtom.traffic_delay_s
        )

        distance_m = max(
            0.0,
            tomtom.distance_m
        )

        # Routing API liefert aktuell kein explizites
        # Confidence-Feld.
        #
        # Erfolgreiche vollständige Routing-Antwort:
        confidence = 1.0

    else:

        no_traffic_s = max(
            0,
            _safe_int(
                osrm_baseline_s
            )
        )

        historic_s = (
            no_traffic_s
        )

        live_s = (
            no_traffic_s
        )

        traffic_delay_s = 0
        distance_m = 0.0
        confidence = 0.0

    # --------------------------------------------------------
    # EIGENER FORECAST
    # --------------------------------------------------------

    model_forecast_s = (
        calculate_model_forecast_s(
            osrm_baseline_s=
                osrm_baseline_s,

            tomtom_historic_s=
                historic_s,

            tomtom_live_s=
                live_s,

            traffic_delay_s=
                traffic_delay_s,

            confidence=
                confidence
        )
    )

    arrival = (
        departure_time
        + timedelta(
            seconds=
                model_forecast_s
        )
    )

    # Servicezeit gehört immer zum Zielstopp.
    planned_service_s = (
        _planned_service_s(
            destination_stop
        )
    )

    actual_service_s = (
        _actual_service_s(
            destination_stop
        )
    )

    return SegmentForecast(
        tour_id=
            str(
                tour_id
            ),

        forecast_version=
            int(
                forecast_version
            ),

        vehicle_id=
            str(
                vehicle_id
            ),

        segment_id=
            int(
                segment_id
            ),

        from_stop_id=
            _stop_id(
                origin_stop,
                f"FROM-{segment_id}"
            ),

        from_stop_name=
            _stop_name(
                origin_stop,
                f"FROM-{segment_id}"
            ),

        to_stop_id=
            _stop_id(
                destination_stop,
                f"TO-{segment_id}"
            ),

        to_stop_name=
            _stop_name(
                destination_stop,
                f"TO-{segment_id}"
            ),

        from_lat=
            float(
                origin_coord[0]
            ),

        from_lon=
            float(
                origin_coord[1]
            ),

        to_lat=
            float(
                destination_coord[0]
            ),

        to_lon=
            float(
                destination_coord[1]
            ),

        departure_time=
            _iso(
                departure_time
            ),

        arrival_time_forecast=
            _iso(
                arrival
            ),

        osrm_baseline_s=
            max(
                0,
                _safe_int(
                    osrm_baseline_s
                )
            ),

        tomtom_no_traffic_s=
            no_traffic_s,

        tomtom_historic_s=
            historic_s,

        tomtom_live_s=
            live_s,

        model_forecast_s=
            model_forecast_s,

        actual_travel_s=(
            max(
                0,
                _safe_int(
                    actual_travel_s
                )
            )
            if actual_travel_s
            is not None
            else None
        ),

        tomtom_traffic_delay_s=
            traffic_delay_s,

        tomtom_distance_m=
            distance_m,

        planned_service_s=
            planned_service_s,

        actual_service_s=
            actual_service_s,

        incident_count=
            0,

        data_confidence=
            confidence,

        recalculated=
            bool(
                recalculated
            ),

        recalculation_from_stop=
            recalculation_from_stop,

        tomtom_success=
            bool(
                tomtom.success
            ),

        tomtom_error=
            tomtom.error,
    )


# ============================================================
# KOMPLETTE TOUR
# ============================================================

def calculate_tour_forecast(
    route: Dict[str, Any],
    depot: Dict[str, Any],
    start_time: Any,

    api_key: Optional[str] = None,
    timeout: int = 20,

    tour_id: Optional[str] = None,
    forecast_version: int = 1,

    vehicle_parameters: Optional[
        Dict[str, Any]
    ] = None,

    recalculation_from_stop: Optional[
        str
    ] = None,

    previous_segments: Optional[
        Sequence[
            Dict[str, Any]
        ]
    ] = None,
) -> Dict[str, Any]:
    """
    Berechnet eine komplette Rundtour segmentweise.

    Ablauf:

        Depot -> Stopp 1
        Service 1

        Stopp 1 -> Stopp 2
        Service 2

        ...

        letzter Stopp -> Depot

    Bei recalculation_from_stop werden bereits gefahrene
    Segmente aus previous_segments übernommen und nur die
    zukünftigen Segmente neu berechnet.
    """

    vehicle_id = str(
        route.get(
            "vehicle_id",
            "UNKNOWN"
        )
    )

    if not tour_id:

        tour_id = (
            f"{vehicle_id}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        )

    start_dt = (
        _parse_datetime(
            start_time
        )
    )

    provider = (
        TomTomRoutingProvider(
            api_key=api_key,
            timeout=timeout
        )
    )

    stops = [
        dict(
            stop
        )
        for stop
        in (
            route.get(
                "stops"
            )
            or []
        )
    ]

    if not stops:

        raise ValueError(
            "Route enthält keine Stopps."
        )

    depot_stop = {
        "auftrag":
            "DEPOT",

        "kunde":
            "Depot",

        "lat":
            float(
                depot[
                    "lat"
                ]
            ),

        "lon":
            float(
                depot[
                    "lon"
                ]
            ),

        "service_min":
            0,
    }

    # --------------------------------------------------------
    # SEGMENTKETTE
    # --------------------------------------------------------

    points = (
        [depot_stop]
        + stops
        + [depot_stop]
    )

    segment_count = (
        len(points)
        - 1
    )

    osrm_leg_times = (
        extract_osrm_leg_baselines(
            route=
                route,

            expected_segments=
                segment_count
        )
    )

    previous_map: Dict[
        int,
        Dict[str, Any]
    ] = {}

    for item in (
        previous_segments
        or []
    ):

        segment_id = (
            _safe_int(
                item.get(
                    "segment_id"
                ),
                -1
            )
        )

        if segment_id >= 0:

            previous_map[
                segment_id
            ] = dict(
                item
            )

    # --------------------------------------------------------
    # REFORECAST-GRENZE SUCHEN
    # --------------------------------------------------------

    recalc_stop_index: Optional[
        int
    ] = None

    if recalculation_from_stop:

        target = str(
            recalculation_from_stop
        )

        for idx, stop in enumerate(
            stops
        ):

            if (
                _stop_id(
                    stop,
                    str(
                        idx + 1
                    )
                )
                == target
            ):

                recalc_stop_index = idx
                break

    segments: List[
        Dict[str, Any]
    ] = []

    current_time = (
        start_dt
    )

    # --------------------------------------------------------
    # ALLE SEGMENTE
