# src/forecast.py
# Logistik Forecast 4.9.5
# Reforecast-Fix:
# - IST-Servicezeit am gewählten Stopp wird korrekt übernommen
# - neue Abfahrtszeit kann den Reforecast direkt setzen
# - nur zukünftige Segmente werden neu berechnet
# - Tour-Gesamtzeit nutzt bekannte IST-Servicezeiten + geplante Rest-Servicezeiten

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.clustering import plan_day
from src.tomtom_routing import TomTomRoutingProvider
from src.ml_predict import predict_if_configured
from src.ml_training import predict_seconds, predict_details
from uuid import uuid4


def _forecast_break_plan(segments, driver_profile):
    return plan_day(
        [float(s.get("actual_travel_s") if s.get("actual_travel_s") is not None
               else s["model_forecast_s"]) / 60 for s in segments],
        [_effective_service_s(s) / 60 for s in segments],
        driver_profile or {},
    )


def _apply_segment_breaks(segment, previous, departure, driver_profile):
    """Keep driving targets separate from breaks; shift arrival and service end."""
    before = _forecast_break_plan(previous, driver_profile)
    report = _forecast_break_plan(previous + [segment], driver_profile)
    events = [e for e in report["break_events"] if e["leg_index"] == len(previous)]
    drive_break_s = sum(round(e["duration_min"] * 60) for e in events if e["during"] == "drive")
    service_break_s = sum(round(e["duration_min"] * 60) for e in events if e["during"] == "service")
    segment["planned_break_s"] = drive_break_s + service_break_s
    segment["break_during_drive_s"] = drive_break_s
    segment["break_during_service_s"] = service_break_s
    segment["break_events"] = [dict(e, starts_at=_iso(departure + timedelta(
        seconds=round((e["after_elapsed_min"] - before["tourdauer_min"]) * 60)))) for e in events]
    arrival = departure + timedelta(seconds=segment["model_forecast_s"] + drive_break_s)
    segment["arrival_time_forecast"] = _iso(arrival)
    return arrival + timedelta(seconds=_effective_service_s(segment) + service_break_s)


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
    forecast_source: str = "Referenzberechnung"
    forecast_warning: Optional[str] = None
    raw_model_s: Optional[int] = None
    prediction_created_at: str = ""
    distance_m: float = 0.0
    model_id: str = "heuristic-v1"

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

def calculate_model_forecast_details(osrm_baseline_s, tomtom_historic_s,
                                     tomtom_live_s, traffic_delay_s=0, confidence=1.0):
    """Canonical seconds-only inputs and a traceable prediction decision."""
    import math

    def seconds(value):
        try:
            number = float(value)
            return max(0, int(round(number))) if math.isfinite(number) else 0
        except (TypeError, ValueError):
            return 0

    osrm, historic, live, delay = map(seconds, [osrm_baseline_s, tomtom_historic_s,
                                               tomtom_live_s, traffic_delay_s])
    confidence = _clamp(_safe_float(confidence, 0), 0, 1)
    references = [v for v in (osrm, historic, live) if v > 0]
    if not references:
        raise ValueError("Keine gültige Fahrzeit für dieses Segment vorhanden.")
    prediction = predict_if_configured(osrm, historic, live, delay, confidence)
    warning = None
    if prediction is not None:
        # Broad engineering guard, not a confidence interval or a calibrated accuracy claim.
        if 0.5 * min(references) <= prediction <= 2 * max(references):
            return {"seconds": prediction, "source": "configured_model", "warning": None}
        warning = "Modellwert außerhalb der Plausibilitätsgrenzen; Referenzberechnung verwendet."

    # Fallbacks are calculation-local. Never write them into provider observation fields.
    effective_live = live or historic or osrm
    effective_historic = historic or osrm or live
    effective_osrm = osrm or historic or live
    weight = .5 + .2 * confidence
    value = round(weight * effective_live + .2 * effective_historic + (0.8-weight) * effective_osrm)
    # Live already contains traffic delay: do not add it a second time.
    return {"seconds": max(0, value),
            "source": "weighted_reference" if historic or live else "osrm_fallback",
            "warning": warning}


def calculate_model_forecast_s(osrm_baseline_s, tomtom_historic_s, tomtom_live_s,
                               traffic_delay_s=0, confidence=1.0):
    return calculate_model_forecast_details(osrm_baseline_s, tomtom_historic_s,
                                            tomtom_live_s, traffic_delay_s, confidence)["seconds"]


def extract_osrm_leg_baselines(route, expected_segments):
    """Use ordered leg seconds only; total/stop-count is not a segment duration."""
    import math
    if expected_segments <= 0:
        return []
    legs = route.get("legs") or []
    if len(legs) != expected_segments:
        raise ValueError("OSRM-Teilstrecken passen nicht zur Stoppreihenfolge. Route neu berechnen.")
    result = []
    for leg in legs:
        try:
            value = float(leg["duration"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("OSRM-Teilstrecke ohne gültige Dauer in Sekunden.") from None
        if not math.isfinite(value) or value < 0:
            raise ValueError("Ungültige OSRM-Teilstreckendauer.")
        result.append(round(value))
    return result


def _provider_minutes(record, field):
    """A total is comparable only if every segment has an observed provider value."""
    segments = record.get("segments")
    if segments is not None:
        if not segments or any(_provider_minutes(part, field) is None for part in segments):
            return None
        return round(sum(float(part[field]) for part in segments) / 60)
    if not record.get("tomtom_success") or _safe_float(record.get(field), 0) <= 0:
        return None
    return round(float(record[field]) / 60)


# ============================================================
# SEGMENT-FORECAST
# ============================================================

def remaining_delivery_load(stops):
    """Validate each delivery before summing remaining goods and pallet tare."""
    import math
    total = 0.0
    for stop in stops:
        goods, pallets = float(stop['warengewicht_kg']), float(stop['paletten'])
        if not all(math.isfinite(v) and v >= 0 for v in (goods, pallets)) or not pallets.is_integer():
            raise ValueError("Ungültige Restladung oder Palettenanzahl.")
        total += goods + 25 * pallets
    return total


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
    ml_bundle=None,
    osrm_distance_m: float = 0.0,
    remaining_load_kg: Optional[float] = None,

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

        # Missing provider observations remain missing (zero sentinel), never copied OSRM.
        no_traffic_s = 0
        historic_s = 0
        live_s = 0

        traffic_delay_s = 0

        distance_m = 0.0

        confidence = 0.0

    # --------------------------------------------------------
    # MODELL-FORECAST
    # --------------------------------------------------------

    model_id = "heuristic-v1"
    raw_model_s = None
    if origin_coord == destination_coord and osrm_baseline_s == 0:
        model_forecast_s, source, warning = 0, "same_location", None
    elif ml_bundle is None:
        decision = calculate_model_forecast_details(
            osrm_baseline_s, historic_s, live_s, traffic_delay_s, confidence
        )
        model_forecast_s = decision["seconds"]
        source, warning = decision["source"], decision["warning"]
    else:
        measured = None
        if os.environ.get("FORECAST_MODEL_PATH"):
            measured = calculate_model_forecast_details(
                osrm_baseline_s, historic_s, live_s, traffic_delay_s, confidence
            )
        if measured and measured["source"] == "configured_model":
            decision = dict(measured, raw_model_s=measured["seconds"])
            model_id = "measured-configured"
        else:
            decision = predict_details(ml_bundle, osrm_baseline_s, osrm_distance_m,
                                       _iso(departure_time), remaining_load_kg)
            model_id = ml_bundle["report"]["model_id"] if not decision["warning"] else "reference-fallback"
        model_forecast_s = decision['seconds']
        source, warning, raw_model_s = decision['source'], decision['warning'], decision['raw_model_s']
        if warning and tomtom.success:
            if live_s > 0:
                model_forecast_s, source = live_s, "TomTom-Live-Ersatzwert"
            elif historic_s > 0:
                model_forecast_s, source = historic_s, "TomTom-Historie-Ersatzwert"

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

    # Future destination service is unknown at prediction time.
    actual_service_s = None

    return SegmentForecast(
        forecast_source=source, forecast_warning=warning, raw_model_s=raw_model_s,
        prediction_created_at=datetime.now(timezone.utc).isoformat(),
        distance_m=osrm_distance_m,
        model_id=model_id,

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
    ml_bundle=None,
    driver_profile: Optional[Dict[str, Any]] = None,

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
                        travel_s + _safe_int(old.get("break_during_drive_s"))
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
                            (_effective_service_s(old) + _safe_int(old.get("break_during_service_s")))
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
                        (_effective_service_s(old) + _safe_int(old.get("break_during_service_s")))
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
                ml_bundle=ml_bundle,
                remaining_load_kg=(remaining_delivery_load(stops[segment_index:]) if ml_bundle is not None and ml_bundle.get('schema_version') else None),
                osrm_distance_m=float((route.get("legs") or [{}] * segment_count)[segment_index].get("distance", 0)) if len(route.get("legs") or []) == segment_count else 0.0,

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

        segment_dict = segment.to_dict()
        current_time = _apply_segment_breaks(segment_dict, segments, current_time, driver_profile)
        segments.append(segment_dict)

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

    result = tour.to_dict()
    result["driver_profile"] = dict(driver_profile or {})
    result["driver_time_plan"] = _forecast_break_plan(segments, driver_profile)
    result["planned_break_s"] = sum(s.get("planned_break_s", 0) for s in segments)
    result["break_policy"] = "45 Minuten nach spätestens 270 Minuten Lenkzeit; Service ist Arbeit. Pausenorte ungeprüft."
    result["schema_version"] = 2
    result["run_id"] = str(uuid4())
    result["vehicle_parameters"] = dict(vehicle_parameters or {})
    return result


# ============================================================
# REFORECAST AB STOPP
# ============================================================

def recalculate_tour_from_stop(
    previous_forecast: Dict[str, Any],
    route: Dict[str, Any],
    depot: Dict[str, Any],

    from_stop_id: str,
    ml_bundle=None,

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
            ml_bundle=ml_bundle,

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

            driver_profile=previous_forecast.get("driver_profile", {}),

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
    baseline_clock = _parse_datetime(forecast["started_at"])

    for segment in (
        forecast.get(
            "segments"
        )
        or []
    ):

        baseline_clock += timedelta(seconds=_safe_int(segment.get("osrm_baseline_s")))
        baseline_arrival = baseline_clock
        predicted_arrival = _parse_datetime(segment["arrival_time_forecast"])
        baseline_clock += timedelta(seconds=_safe_int(segment.get("planned_service_s")))
        rows.append({
            "forecast_quelle": segment.get("forecast_source", "legacy"),
            "forecast_hinweis": segment.get("forecast_warning"),
            "ankunft_plan_osrm": _iso(baseline_arrival),
            "abweichung_plan_min": round((predicted_arrival-baseline_arrival).total_seconds()/60, 1),
            "modell": segment.get("model_id", "heuristic-v1"),

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
                _provider_minutes(segment, "tomtom_no_traffic_s"),

            # ------------------------------------------------
            # 3 HISTORISCH
            # ------------------------------------------------

            "tomtom_historisch_min":
                _provider_minutes(segment, "tomtom_historic_s"),

            # ------------------------------------------------
            # 2 LIVE
            # ------------------------------------------------

            "tomtom_live_min":
                _provider_minutes(segment, "tomtom_live_s"),

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
        "forecast_warnungen": " | ".join(dict.fromkeys(s["forecast_warning"] for s in forecast.get("segments", []) if s.get("forecast_warning"))),

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
            _provider_minutes(forecast, "tomtom_no_traffic_s"),

        # ====================================================
        # 3 TOMTOM HISTORISCH
        # ====================================================

        "tomtom_historisch_min":
            _provider_minutes(forecast, "tomtom_historic_s"),

        # ====================================================
        # 2 TOMTOM LIVE
        # ====================================================

        "tomtom_live_min":
            _provider_minutes(forecast, "tomtom_live_s"),

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
