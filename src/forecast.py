# src/forecast.py
# Logistik Forecast 4.9.5
# Reforecast-Fix:
# - IST-Servicezeit am gewählten Stopp wird korrekt übernommen
# - neue Abfahrtszeit kann den Reforecast direkt setzen
# - nur zukünftige Segmente werden neu berechnet
# - Tour-Gesamtzeit nutzt bekannte IST-Servicezeiten + geplante Rest-Servicezeiten

from __future__ import annotations

from dataclasses import asdict, dataclass
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
            value
            or ""
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
            stop[
                "lat"
            ]
        ),

        float(
            stop[
                "lon"
            ]
        ),
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
            _safe_float(
                stop.get(
                    "service_min",
                    0
                )
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
                _safe_float(
                    stop.get(
                        "actual_service_min"
                    )
                )
                * 60
            )
        )

    return None


def _effective_service_s(
    segment: Dict[str, Any]
) -> int:
    """
    Für die aktuelle Zeitachse gilt:

    bekannte IST-Servicezeit
    vor
    geplanter Servicezeit.
    """

    if (
        segment.get(
            "actual_service_s"
        )
        is not None
    ):

        return max(
            0,
            _safe_int(
                segment.get(
                    "actual_service_s"
                )
            )
        )

    return max(
        0,
        _safe_int(
            segment.get(
                "planned_service_s"
            )
        )
    )


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
    Übergangslogik bis zum trainierten ML-Modell.

    Gewichtung:

    TomTom Live
        stärkster Faktor

    TomTom Historisch
        stabilisierender Faktor

    OSRM
        Baseline/Fallback
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
        1.0,
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
    # GEWICHTUNG
    # --------------------------------------------------------

    live_weight = (
        0.50
        +
        (
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

    # TomTom Live enthält Verkehrsverzögerungen bereits.
    # Deshalb KEIN vollständiges zusätzliches Addieren.
    #
    # Nur kleine Sicherheitskorrektur.
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
    Erwartete Segmente:

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

    durations = [

        max(
            0,
            _safe_int(
                leg.get(
                    "duration"
                )
            )
        )

        for leg
        in legs
    ]

    if (
        len(
            durations
        )
        == expected_segments
    ):

        return durations

    # --------------------------------------------------------
    # FALLBACK
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

    difference = (
        total_duration
        - sum(
            result
        )
    )

    if result:

        result[
            -1
        ] += difference

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

    recalculation_from_stop: Optional[
        str
    ] = None,

    actual_travel_s: Optional[
        int
    ] = None,

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

    # --------------------------------------------------------
    # TOMTOM SEGMENT
    # --------------------------------------------------------

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
    # TOMTOM OK
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

        confidence = 1.0

    # --------------------------------------------------------
    # TOMTOM FALLBACK
    # --------------------------------------------------------

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
    # MODELL-FORECAST
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
                confidence,
        )
    )

    arrival = (
        departure_time
        +
        timedelta(
            seconds=
                model_forecast_s
        )
    )

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
                origin_coord[
                    0
                ]
            ),

        from_lon=
            float(
                origin_coord[
                    1
                ]
            ),

        to_lat=
            float(
                destination_coord[
                    0
                ]
            ),

        to_lon=
            float(
                destination_coord[
                    1
                ]
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

    recalculation_departure_time: Optional[
        Any
    ] = None,

) -> Dict[str, Any]:
    """
    Kompletten Tour-Forecast segmentweise berechnen.

    Normal:

    Depot -> Stopp 1
    Service Stopp 1

    Stopp 1 -> Stopp 2
    Service Stopp 2

    ...

    letzter Stopp -> Depot


    Reforecast ab Stopp 3:

    Depot -> 1
        bleibt

    1 -> 2
        bleibt

    2 -> 3
        bleibt

    Service an Stopp 3
        wird mit IST-Service überschrieben,
        falls vorhanden

    3 -> 4
        NEU

    4 -> 5
        NEU

    ...

    letzter Stopp -> Depot
        NEU
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

            api_key=
                api_key,

            timeout=
                timeout,
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

    # ========================================================
    # SEGMENTKETTE
    # ========================================================

    points = (

        [
            depot_stop
        ]

        +
        stops

        +
        [
            depot_stop
        ]
    )

    segment_count = (
        len(
            points
        )
        - 1
    )

    osrm_leg_times = (
        extract_osrm_leg_baselines(

            route=
                route,

            expected_segments=
                segment_count,
        )
    )

    # ========================================================
    # VORHERIGE SEGMENTE
    # ========================================================

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

    # ========================================================
    # REFORECAST STOPP SUCHEN
    # ========================================================

    recalc_stop_index: Optional[
        int
    ] = None

    recalc_stop: Optional[
        Dict[str, Any]
    ] = None

    if (
        recalculation_from_stop
        is not None
    ):

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

                recalc_stop = stop

                break

        if (
            recalc_stop_index
            is None
        ):

            raise ValueError(
                f"Reforecast-Stopp "
                f"{recalculation_from_stop} "
                f"wurde nicht gefunden."
            )

    # ========================================================
    # NEUE ABFAHRT
    # ========================================================

    forced_departure = None

    if (
        recalculation_departure_time
        is not None
    ):

        forced_departure = (
            _parse_datetime(
                recalculation_departure_time
            )
        )

    # ========================================================
    # BERECHNUNG
    # ========================================================

    segments: List[
        Dict[str, Any]
    ] = []

    current_time = (
        start_dt
    )

    for segment_index in range(
        segment_count
    ):

        segment_id = (
            segment_index
            + 1
        )

        origin = (
            points[
                segment_index
            ]
        )

        destination = (
            points[
                segment_index
                + 1
            ]
        )

        # ====================================================
        # ENTSCHEIDUNG:
        # ALTES SEGMENT ODER NEU BERECHNEN
        # ====================================================

        reuse_previous = False

        if (
            recalc_stop_index
            is not None
        ):

            # Beispiel:
            #
            # Reforecast ab Stopp 3
            #
            # Segmentindex:
            #
            # 0 Depot -> 1
            # 1 1 -> 2
            # 2 2 -> 3
            # 3 3 -> 4
            #
            # Neu ab Index 3.

            first_new_segment_index = (
                recalc_stop_index
                + 1
            )

            if (
                segment_index
                < first_new_segment_index

                and

                segment_id
                in previous_map
            ):

                reuse_previous = True

        # ====================================================
        # ALTES SEGMENT ÜBERNEHMEN
        # ====================================================

        if reuse_previous:

            old = dict(
                previous_map[
                    segment_id
                ]
            )

            departure = (
                _parse_datetime(
                    old.get(
                        "departure_time",
                        current_time
                    )
                )
            )

            actual_travel = (
                old.get(
                    "actual_travel_s"
                )
            )

            if (
                actual_travel
                is not None
            ):

                travel_s = max(
                    0,
                    _safe_int(
                        actual_travel
                    )
                )

            else:

                travel_s = max(
                    0,
                    _safe_int(
                        old.get(
                            "model_forecast_s"
                        )
                    )
                )

            arrival_time = (
                departure
                +
                timedelta(
                    seconds=
                        travel_s
                )
            )

            # =================================================
            # REFORECAST-GRENZE
            # =================================================

            is_boundary_segment = (

                recalc_stop_index
                is not None

                and

                segment_index
                == recalc_stop_index
            )

            if (
                is_boundary_segment
                and
                recalc_stop
                is not None
            ):

                # ---------------------------------------------
                # IST SERVICEZEIT DES GEWÄHLTEN STOPPS
                # ---------------------------------------------

                updated_actual_service = (
                    _actual_service_s(
                        recalc_stop
                    )
                )

                if (
                    updated_actual_service
                    is not None
                ):

                    old[
                        "actual_service_s"
                    ] = (
                        updated_actual_service
                    )

                # ---------------------------------------------
                # ALTES SEGMENT ERST JETZT SPEICHERN
                # ---------------------------------------------

                segments.append(
                    old
                )

                # ---------------------------------------------
                # NEUE ABFAHRT EXPLIZIT GESETZT
                # ---------------------------------------------

                if (
                    forced_departure
                    is not None
                ):

                    if (
                        forced_departure
                        < arrival_time
                    ):

                        raise ValueError(
                            "Die neue Abfahrtszeit liegt "
                            "vor der Ankunftszeit am "
                            "Reforecast-Stopp."
                        )

                    current_time = (
                        forced_departure
                    )

                    continue

                # ---------------------------------------------
                # SONST:
                # ANKUNFT + IST/PLAN-SERVICE
                # ---------------------------------------------

                current_time = (

                    arrival_time

                    +

                    timedelta(
                        seconds=
                            _effective_service_s(
                                old
                            )
                    )
                )

                continue

            # =================================================
            # NORMALES ALTES SEGMENT
            # =================================================

            segments.append(
                old
            )

            current_time = (

                arrival_time

                +

                timedelta(
                    seconds=
                        _effective_service_s(
                            old
                        )
                )
            )

            continue

        # ====================================================
        # ZUKÜNFTIGES SEGMENT NEU BERECHNEN
        # ====================================================

        segment = (
            calculate_segment_forecast(

                tour_id=
                    tour_id,

                forecast_version=
                    forecast_version,

                vehicle_id=
                    vehicle_id,

                segment_id=
                    segment_id,

                origin_stop=
                    origin,

                destination_stop=
                    destination,

                departure_time=
                    current_time,

                osrm_baseline_s=
                    osrm_leg_times[
                        segment_index
                    ],

                provider=
                    provider,

                vehicle_parameters=
                    vehicle_parameters,

                recalculated=(
                    recalc_stop_index
                    is not None
                ),

                recalculation_from_stop=
                    recalculation_from_stop,
            )
        )

        segment_dict = (
            segment.to_dict()
        )

        segments.append(
            segment_dict
        )

        # ----------------------------------------------------
        # ANKUNFT
        # ----------------------------------------------------

        current_time += timedelta(

            seconds=
                segment.model_forecast_s
        )

        # ----------------------------------------------------
        # SERVICE AM ZIELSTOPP
        # ----------------------------------------------------

        current_time += timedelta(

            seconds=
                _effective_service_s(
                    segment_dict
                )
        )

    # ========================================================
    # TOURSUMMEN
    # ========================================================

    osrm_baseline_s = sum(

        _safe_int(
            segment.get(
                "osrm_baseline_s"
            )
        )

        for segment
        in segments
    )

    no_traffic_s = sum(

        _safe_int(
            segment.get(
                "tomtom_no_traffic_s"
            )
        )

        for segment
        in segments
    )

    historic_s = sum(

        _safe_int(
            segment.get(
                "tomtom_historic_s"
            )
        )

        for segment
        in segments
    )

    live_s = sum(

        _safe_int(
            segment.get(
                "tomtom_live_s"
            )
        )

        for segment
        in segments
    )

    model_forecast_s = sum(

        _safe_int(
            segment.get(
                "model_forecast_s"
            )
        )

        for segment
        in segments
    )

    planned_service_s = sum(

        _safe_int(
            segment.get(
                "planned_service_s"
            )
        )

        for segment
        in segments
    )

    # ========================================================
    # BEKANNTE IST SERVICEZEITEN
    # ========================================================

    actual_service_values = [

        segment.get(
            "actual_service_s"
        )

        for segment
        in segments

        if (
            segment.get(
                "actual_service_s"
            )
            is not None
        )
    ]

    actual_service_s = (

        sum(

            _safe_int(
                value
            )

            for value
            in actual_service_values
        )

        if actual_service_values

        else None
    )

    # ========================================================
    # IST FAHRZEIT
    # ========================================================

    actual_travel_values = [

        segment.get(
            "actual_travel_s"
        )

        for segment
        in segments

        if (
            segment.get(
                "actual_travel_s"
            )
            is not None
        )
    ]

    actual_travel_s = (

        sum(

            _safe_int(
                value
            )

            for value
            in actual_travel_values
        )

        if (
            len(
                actual_travel_values
            )
            ==
            len(
                segments
            )
        )

        else None
    )

    # ========================================================
    # DISTANZ
    # ========================================================

    total_distance_m = sum(

        _safe_float(
            segment.get(
                "tomtom_distance_m"
            )
        )

        for segment
        in segments
    )

    if (
        total_distance_m
        <= 0
    ):

        total_distance_m = (
            _safe_float(
                route.get(
                    "distance_m"
                )
            )
        )

    # ========================================================
    # TOUR
    # ========================================================

    tour = TourForecast(

        tour_id=
            str(
                tour_id
            ),

        forecast_version=
            int(
                forecast_version
            ),

        vehicle_id=
            vehicle_id,

        started_at=
            _iso(
                start_dt
            ),

        finished_at_forecast=
            _iso(
                current_time
            ),

        osrm_baseline_s=
            osrm_baseline_s,

        tomtom_no_traffic_s=
            no_traffic_s,

        tomtom_historic_s=
            historic_s,

        tomtom_live_s=
            live_s,

        model_forecast_s=
            model_forecast_s,

        actual_travel_s=
            actual_travel_s,

        planned_service_s=
            planned_service_s,

        actual_service_s=
            actual_service_s,

        total_distance_m=
            total_distance_m,

        segment_count=
            len(
                segments
            ),

        stop_count=
            len(
                stops
            ),

        recalculated=(
            recalc_stop_index
            is not None
        ),

        recalculation_from_stop=
            recalculation_from_stop,

        segments=
            segments,
    )

    return (
        tour.to_dict()
    )


# ============================================================
# REFORECAST AB STOPP
# ============================================================

def recalculate_tour_from_stop(
    previous_forecast: Dict[str, Any],
    route: Dict[str, Any],
    depot: Dict[str, Any],

    from_stop_id: str,

    new_departure_time: Optional[
        Any
    ] = None,

    actual_service_s: Optional[
        int
    ] = None,

    api_key: Optional[str] = None,
    timeout: int = 20,

    vehicle_parameters: Optional[
        Dict[str, Any]
    ] = None,

) -> Dict[str, Any]:
    """
    Neuberechnung ab einem gewählten Stopp.

    Beispiel:

    Forecast V1

    Depot -> 1
    1 -> 2
    2 -> 3
    3 -> 4
    4 -> Depot


    Reforecast ab Stopp 3

    Depot -> 1
        bleibt

    1 -> 2
        bleibt

    2 -> 3
        bleibt

    Service Stopp 3
        IST-Wert wird übernommen

    3 -> 4
        neu

    4 -> Depot
        neu


    Optional:

    new_departure_time

    setzt die tatsächliche Abfahrt am gewählten Stopp
    direkt.
    """

    if not previous_forecast:

        raise ValueError(
            "Vorheriger Forecast fehlt."
        )

    old_segments = [

        dict(
            segment
        )

        for segment
        in (
            previous_forecast.get(
                "segments"
            )
            or []
        )
    ]

    previous_version = (
        _safe_int(
            previous_forecast.get(
                "forecast_version"
            ),
            1
        )
    )

    new_version = (
        previous_version
        + 1
    )

    tour_id = str(
        previous_forecast.get(
            "tour_id"
        )
        or ""
    )

    if not tour_id:

        raise ValueError(
            "Vorheriger Forecast enthält "
            "keine tour_id."
        )

    # ========================================================
    # ROUTE KOPIEREN
    # ========================================================

    route_copy = dict(
        route
    )

    route_copy[
        "stops"
    ] = [

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

    # ========================================================
    # GEWÄHLTEN STOPP SUCHEN
    # ========================================================

    target_stop: Optional[
        Dict[str, Any]
    ] = None

    for stop in (
        route_copy[
            "stops"
        ]
    ):

        if (
            _stop_id(
                stop,
                ""
            )
            ==
            str(
                from_stop_id
            )
        ):

            target_stop = stop

            break

    if (
        target_stop
        is None
    ):

        raise ValueError(
            f"Stopp {from_stop_id} "
            f"wurde in der Route "
            f"nicht gefunden."
        )

    # ========================================================
    # IST SERVICEZEIT
    # ========================================================

    if (
        actual_service_s
        is not None
    ):

        target_stop[
            "actual_service_s"
        ] = max(
            0,
            _safe_int(
                actual_service_s
            )
        )

    # ========================================================
    # NEUE ABFAHRT
    # ========================================================

    normalized_departure = None

    if (
        new_departure_time
        is not None
    ):

        text = str(
            new_departure_time
        ).strip()

        if text:

            normalized_departure = (
                _parse_datetime(
                    new_departure_time
                )
            )

    # ========================================================
    # URSPRÜNGLICHER TOURSTART
    # ========================================================

    original_start = (
        previous_forecast.get(
            "started_at"
        )
    )

    if not original_start:

        raise ValueError(
            "Vorheriger Forecast enthält "
            "keine started_at-Zeit."
        )

    # ========================================================
    # NEUE VERSION BERECHNEN
    # ========================================================

    return (
        calculate_tour_forecast(

            route=
                route_copy,

            depot=
                depot,

            start_time=
                original_start,

            api_key=
                api_key,

            timeout=
                timeout,

            tour_id=
                tour_id,

            forecast_version=
                new_version,

            vehicle_parameters=
                vehicle_parameters,

            recalculation_from_stop=
                str(
                    from_stop_id
                ),

            previous_segments=
                old_segments,

            recalculation_departure_time=
                normalized_departure,
        )
    )


# ============================================================
# SEGMENT-TABELLE
# ============================================================

def segment_rows(
    forecast: Dict[str, Any]
) -> List[Dict[str, Any]]:

    rows = []

    for segment in (
        forecast.get(
            "segments"
        )
        or []
    ):

        rows.append({

            "segment":
                segment.get(
                    "segment_id"
                ),

            "von":
                segment.get(
                    "from_stop_name"
                ),

            "nach":
                segment.get(
                    "to_stop_name"
                ),

            "abfahrt":
                segment.get(
                    "departure_time"
                ),

            "ankunft_forecast":
                segment.get(
                    "arrival_time_forecast"
                ),

            # ------------------------------------------------
            # 1 OSRM
            # ------------------------------------------------

            "osrm_basis_min":
                round(
                    _safe_int(
                        segment.get(
                            "osrm_baseline_s"
                        )
                    )
                    / 60
                ),

            # ------------------------------------------------
            # TOMTOM REFERENZ
            # ------------------------------------------------

            "tomtom_ohne_verkehr_min":
                round(
                    _safe_int(
                        segment.get(
                            "tomtom_no_traffic_s"
                        )
                    )
                    / 60
                ),

            # ------------------------------------------------
            # 3 HISTORISCH
            # ------------------------------------------------

            "tomtom_historisch_min":
                round(
                    _safe_int(
                        segment.get(
                            "tomtom_historic_s"
                        )
                    )
                    / 60
                ),

            # ------------------------------------------------
            # 2 LIVE
            # ------------------------------------------------

            "tomtom_live_min":
                round(
                    _safe_int(
                        segment.get(
                            "tomtom_live_s"
                        )
                    )
                    / 60
                ),

            # ------------------------------------------------
            # 4 EIGENER FORECAST
            # ------------------------------------------------

            "forecast_min":
                round(
                    _safe_int(
                        segment.get(
                            "model_forecast_s"
                        )
                    )
                    / 60
                ),

            # ------------------------------------------------
            # 5 IST
            # ------------------------------------------------

            "ist_min": (

                round(
                    _safe_int(
                        segment.get(
                            "actual_travel_s"
                        )
                    )
                    / 60
                )

                if (
                    segment.get(
                        "actual_travel_s"
                    )
                    is not None
                )

                else None
            ),

            # ------------------------------------------------
            # TRAFFIC
            # ------------------------------------------------

            "traffic_delay_min":
                round(
                    _safe_int(
                        segment.get(
                            "tomtom_traffic_delay_s"
                        )
                    )
                    / 60
                ),

            # ------------------------------------------------
            # SERVICE
            # ------------------------------------------------

            "service_geplant_min":
                round(
                    _safe_int(
                        segment.get(
                            "planned_service_s"
                        )
                    )
                    / 60
                ),

            "service_ist_min": (

                round(
                    _safe_int(
                        segment.get(
                            "actual_service_s"
                        )
                    )
                    / 60
                )

                if (
                    segment.get(
                        "actual_service_s"
                    )
                    is not None
                )

                else None
            ),

            "service_verwendet_min":
                round(
                    _effective_service_s(
                        segment
                    )
                    / 60
                ),

            # ------------------------------------------------
            # REFORECAST
            # ------------------------------------------------

            "neu_berechnet":
                bool(
                    segment.get(
                        "recalculated"
                    )
                ),

            "tomtom_ok":
                bool(
                    segment.get(
                        "tomtom_success"
                    )
                ),
        })

    return rows


# ============================================================
# TOUR SUMMARY
# ============================================================

def tour_summary(
    forecast: Dict[str, Any]
) -> Dict[str, Any]:

    segments = (
        forecast.get(
            "segments"
        )
        or []
    )

    # ========================================================
    # AKTUELL VERWENDETE SERVICEZEIT
    #
    # IST sofern bekannt,
    # sonst geplant.
    # ========================================================

    effective_service_s = sum(

        _effective_service_s(
            segment
        )

        for segment
        in segments
    )

    # ========================================================
    # NUR BEKANNTE IST SERVICEZEITEN
    # ========================================================

    known_actual_service_s = sum(

        _safe_int(
            segment.get(
                "actual_service_s"
            )
        )

        for segment
        in segments

        if (
            segment.get(
                "actual_service_s"
            )
            is not None
        )
    )

    return {

        "tour_id":
            forecast.get(
                "tour_id"
            ),

        "forecast_version":
            forecast.get(
                "forecast_version"
            ),

        "vehicle_id":
            forecast.get(
                "vehicle_id"
            ),

        "stopps":
            forecast.get(
                "stop_count"
            ),

        "segmente":
            forecast.get(
                "segment_count"
            ),

        "distanz_km":
            round(
                _safe_float(
                    forecast.get(
                        "total_distance_m"
                    )
                )
                / 1000,
                1
            ),

        # ====================================================
        # 1 OSRM BASIS
        # ====================================================

        "osrm_basis_min":
            round(
                _safe_int(
                    forecast.get(
                        "osrm_baseline_s"
                    )
                )
                / 60
            ),

        # ====================================================
        # TOMTOM REFERENZ
        # ====================================================

        "tomtom_ohne_verkehr_min":
            round(
                _safe_int(
                    forecast.get(
                        "tomtom_no_traffic_s"
                    )
                )
                / 60
            ),

        # ====================================================
        # 3 TOMTOM HISTORISCH
        # ====================================================

        "tomtom_historisch_min":
            round(
                _safe_int(
                    forecast.get(
                        "tomtom_historic_s"
                    )
                )
                / 60
            ),

        # ====================================================
        # 2 TOMTOM LIVE
        # ====================================================

        "tomtom_live_min":
            round(
                _safe_int(
                    forecast.get(
                        "tomtom_live_s"
                    )
                )
                / 60
            ),

        # ====================================================
        # 4 EIGENER FORECAST
        # ====================================================

        "forecast_fahrzeit_min":
            round(
                _safe_int(
                    forecast.get(
                        "model_forecast_s"
                    )
                )
                / 60
            ),

        # ====================================================
        # SERVICE
        # ====================================================

        "service_geplant_min":
            round(
                _safe_int(
                    forecast.get(
                        "planned_service_s"
                    )
                )
                / 60
            ),

        "service_ist_bekannt_min":
            round(
                known_actual_service_s
                / 60
            ),

        "service_verwendet_min":
            round(
                effective_service_s
                / 60
            ),

        # ====================================================
        # GESAMT FORECAST
        #
        # Forecast-Fahrzeit
        # +
        # bekannte IST-Servicezeiten
        # +
        # geplante Servicezeiten der restlichen Stopps
        # ====================================================

        "forecast_gesamt_min":
            round(

                (
                    _safe_int(
                        forecast.get(
                            "model_forecast_s"
                        )
                    )

                    +

                    effective_service_s
                )

                / 60
            ),

        # ====================================================
        # 5 IST
        # ====================================================

        "ist_fahrzeit_min": (

            round(
                _safe_int(
                    forecast.get(
                        "actual_travel_s"
                    )
                )
                / 60
            )

            if (
                forecast.get(
                    "actual_travel_s"
                )
                is not None
            )

            else None
        ),

        # ====================================================
        # REFORECAST
        # ====================================================

        "neu_berechnet":
            bool(
                forecast.get(
                    "recalculated"
                )
            ),

        "ab_stopp":
            forecast.get(
                "recalculation_from_stop"
            ),
    }


# ============================================================
# ABWÄRTSKOMPATIBILITÄT
# ============================================================

def forecast_summary(
    distance_m,
    baseline_s,
    traffic_s,
    service_min
):
    """
    Alte Funktion bleibt bestehen,
    falls ältere Komponenten sie noch importieren.
    """

    distance_m = float(
        distance_m
        or 0
    )

    baseline_s = float(
        baseline_s
        or 0
    )

    traffic_s = float(
        traffic_s
        or 0
    )

    service_min = float(
        service_min
        or 0
    )

    total_s = (

        baseline_s

        +

        traffic_s

        +

        service_min
        * 60
    )

    return {

        "distanz_km":
            round(
                distance_m
                / 1000,
                1
            ),

        "basis_fahrzeit_min":
            round(
                baseline_s
                / 60
            ),

        "live_zuschlag_min":
            round(
                traffic_s
                / 60
            ),

        "servicezeit_min":
            round(
                service_min
            ),

        "gesamtzeit_min":
            round(
                total_s
                / 60
            ),
    }