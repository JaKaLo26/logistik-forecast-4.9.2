# src/traffic.py

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


# ============================================================
# KONFIGURATION
# ============================================================

TOMTOM_API_KEY = os.getenv(
    "TOMTOM_API_KEY",
    ""
).strip()

REQUEST_TIMEOUT = int(
    os.getenv(
        "REQUEST_TIMEOUT_SECONDS",
        "20"
    )
)

DEFAULT_SAMPLE_POINTS = int(
    os.getenv(
        "TOMTOM_SAMPLE_POINTS",
        "12"
    )
)

TOMTOM_FLOW_URL = (
    "https://api.tomtom.com/traffic/services/4/"
    "flowSegmentData/absolute/10/json"
)

TOMTOM_INCIDENT_URL = (
    "https://api.tomtom.com/traffic/services/5/"
    "incidentDetails"
)


# ============================================================
# DATENMODELL
# ============================================================

@dataclass
class TrafficResult:
    provider: str
    delay_s: int = 0
    score: float = 0.0
    confidence: float = 0.0
    incidents: Optional[List[Dict[str, Any]]] = None
    debug: Optional[Dict[str, Any]] = None

    def __post_init__(self):

        if self.incidents is None:
            self.incidents = []

        if self.debug is None:
            self.debug = {}

    def to_dict(
        self
    ) -> Dict[str, Any]:

        return asdict(self)

    # Damit ältere Stellen optional auch
    # result["delay_s"] verwenden könnten.
    def __getitem__(
        self,
        key
    ):

        return getattr(
            self,
            key
        )


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

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


def _looks_like_lon_lat(
    first: float,
    second: float
) -> bool:
    """
    Erkennt typische OSRM-GeoJSON-Koordinaten.

    OSRM GeoJSON:
        [longitude, latitude]

    Deutschland:
        lon ungefähr 5 bis 16
        lat ungefähr 47 bis 55
    """

    if not (
        -180 <= first <= 180
        and -90 <= second <= 90
    ):
        return False

    # Für Deutschland / Europa ist dies zuverlässig genug.
    if (
        abs(first) <= 35
        and abs(second) >= 35
    ):
        return True

    return False


def normalize_coordinate(
    point: Any
) -> Optional[Tuple[float, float]]:
    """
    Wandelt eine Koordinate immer in:

        (latitude, longitude)

    um.

    Unterstützt:

        [lon, lat]        OSRM GeoJSON
        (lon, lat)
        {"lat": ..., "lon": ...}
        {"lat": ..., "lng": ...}
        {"latitude": ..., "longitude": ...}
    """

    # --------------------------------------------------------
    # DICT
    # --------------------------------------------------------

    if isinstance(
        point,
        dict
    ):

        lat = point.get(
            "lat"
        )

        if lat is None:
            lat = point.get(
                "latitude"
            )

        lon = point.get(
            "lon"
        )

        if lon is None:
            lon = point.get(
                "lng"
            )

        if lon is None:
            lon = point.get(
                "longitude"
            )

        if (
            lat is None
            or lon is None
        ):
            return None

        try:

            lat = float(lat)
            lon = float(lon)

        except (
            TypeError,
            ValueError
        ):

            return None

        if not (
            -90 <= lat <= 90
        ):
            return None

        if not (
            -180 <= lon <= 180
        ):
            return None

        return (
            lat,
            lon
        )

    # --------------------------------------------------------
    # LIST / TUPLE
    # --------------------------------------------------------

    if (
        isinstance(
            point,
            (
                list,
                tuple
            )
        )
        and len(point) >= 2
    ):

        try:

            first = float(
                point[0]
            )

            second = float(
                point[1]
            )

        except (
            TypeError,
            ValueError
        ):

            return None

        if _looks_like_lon_lat(
            first,
            second
        ):

            lon = first
            lat = second

        else:

            lat = first
            lon = second

        if not (
            -90 <= lat <= 90
        ):
            return None

        if not (
            -180 <= lon <= 180
        ):
            return None

        return (
            lat,
            lon
        )

    return None


def normalize_route(
    route: Any
) -> List[Tuple[float, float]]:
    """
    Extrahiert Koordinaten aus verschiedenen möglichen
    OSRM-/GeoJSON-Strukturen.

    Rückgabe:

        [
            (lat, lon),
            (lat, lon),
            ...
        ]
    """

    if route is None:
        return []

    # --------------------------------------------------------
    # GEOJSON / ROUTE DICT
    # --------------------------------------------------------

    if isinstance(
        route,
        dict
    ):

        # GeoJSON LineString
        if (
            "coordinates"
            in route
        ):

            route = route[
                "coordinates"
            ]

        # {
        #   "geometry": {
        #       "coordinates": [...]
        #   }
        # }
        elif (
            "geometry"
            in route
        ):

            geometry = route[
                "geometry"
            ]

            if isinstance(
                geometry,
                dict
            ):

                route = geometry.get(
                    "coordinates",
                    []
                )

            elif isinstance(
                geometry,
                (
                    list,
                    tuple
                )
            ):

                route = geometry

            else:

                return []

        elif (
            "route"
            in route
        ):

            route = route[
                "route"
            ]

        elif (
            "points"
            in route
        ):

            route = route[
                "points"
            ]

    if not isinstance(
        route,
        (
            list,
            tuple
        )
    ):

        return []

    result: List[
        Tuple[
            float,
            float
        ]
    ] = []

    for point in route:

        normalized = (
            normalize_coordinate(
                point
            )
        )

        if (
            normalized
            is not None
        ):

            result.append(
                normalized
            )

    return result


# ============================================================
# ROUTEN-SAMPLING
# ============================================================

def sample_route(
    route: Sequence[
        Tuple[
            float,
            float
        ]
    ],
    max_points: int = DEFAULT_SAMPLE_POINTS
) -> List[Tuple[float, float]]:
    """
    TomTom muss nicht für jeden einzelnen OSRM-Punkt
    abgefragt werden.

    Es werden gleichmäßig verteilte Punkte verwendet.
    """

    points = list(
        route
    )

    if not points:

        return []

    max_points = max(
        1,
        int(max_points)
    )

    if (
        len(points)
        <= max_points
    ):

        return points

    if (
        max_points
        == 1
    ):

        return [
            points[
                len(points)
                // 2
            ]
        ]

    result = []

    for i in range(
        max_points
    ):

        index = round(
            i
            * (
                len(points)
                - 1
            )
            / (
                max_points
                - 1
            )
        )

        point = points[
            index
        ]

        if (
            point
            not in result
        ):

            result.append(
                point
            )

    return result


# ============================================================
# TOMTOM FLOW
# ============================================================

def tomtom_flow(
    lat: float,
    lon: float,
    api_key: Optional[str] = None,
    timeout: int = REQUEST_TIMEOUT
) -> Dict[str, Any]:

    key = (
        api_key
        or TOMTOM_API_KEY
    ).strip()

    if not key:

        raise RuntimeError(
            "TOMTOM_API_KEY fehlt."
        )

    params = {
        "key":
            key,

        "point":
            f"{lat},{lon}",

        "unit":
            "KMPH",

        # Straße möglichst passend zur Fahrtrichtung finden.
        "openLr":
            "false",
    }

    response = requests.get(
        TOMTOM_FLOW_URL,
        params=params,
        timeout=timeout
    )

    # --------------------------------------------------------
    # KLARE FEHLERMELDUNGEN
    # --------------------------------------------------------

    if (
        response.status_code
        == 401
    ):

        raise RuntimeError(
            "TomTom HTTP 401: "
            "API-Key wurde nicht akzeptiert."
        )

    if (
        response.status_code
        == 403
    ):

        raise RuntimeError(
            "TomTom HTTP 403: "
            "API-Key oder Traffic-API-Berechtigung prüfen."
        )

    if (
        response.status_code
        == 429
    ):

        raise RuntimeError(
            "TomTom HTTP 429: "
            "API-Anfragelimit erreicht."
        )

    response.raise_for_status()

    data = response.json()

    flow = (
        data.get(
            "flowSegmentData"
        )
        or {}
    )

    current_speed = _safe_float(
        flow.get(
            "currentSpeed"
        )
    )

    free_flow_speed = _safe_float(
        flow.get(
            "freeFlowSpeed"
        )
    )

    current_time = _safe_int(
        flow.get(
            "currentTravelTime"
        )
    )

    free_flow_time = _safe_int(
        flow.get(
            "freeFlowTravelTime"
        )
    )

    confidence = _safe_float(
        flow.get(
            "confidence"
        )
    )

    road_closure = bool(
        flow.get(
            "roadClosure",
            False
        )
    )

    # --------------------------------------------------------
    # DELAY
    # --------------------------------------------------------

    delay_s = max(
        0,
        current_time
        - free_flow_time
    )

    # --------------------------------------------------------
    # SCORE
    #
    # 0 = frei
    # 1 = Stillstand / Sperrung
    # --------------------------------------------------------

    if road_closure:

        score = 1.0

    elif (
        free_flow_speed
        > 0
    ):

        speed_ratio = (
            current_speed
            / free_flow_speed
        )

        score = (
            1.0
            - speed_ratio
        )

        score = _clamp(
            score,
            0.0,
            1.0
        )

    else:

        score = 0.0

    return {
        "lat":
            lat,

        "lon":
            lon,

        "current_speed_kmh":
            current_speed,

        "free_flow_speed_kmh":
            free_flow_speed,

        "current_travel_time_s":
            current_time,

        "free_flow_travel_time_s":
            free_flow_time,

        "delay_s":
            delay_s,

        "score":
            round(
                score,
                3
            ),

        "confidence":
            round(
                _clamp(
                    confidence,
                    0.0,
                    1.0
                ),
                3
            ),

        "road_closure":
            road_closure,

        "frc":
            flow.get(
                "frc"
            ),
    }


# ============================================================
# TOMTOM INCIDENTS
# ============================================================

def _route_bbox(
    coordinates: Sequence[
        Tuple[
            float,
            float
        ]
    ],
    padding: float = 0.01
) -> Optional[str]:
    """
    TomTom Incident API erwartet:

        minLon,minLat,maxLon,maxLat
    """

    if not coordinates:

        return None

    lats = [
        point[0]
        for point
        in coordinates
    ]

    lons = [
        point[1]
        for point
        in coordinates
    ]

    min_lat = (
        min(lats)
        - padding
    )

    max_lat = (
        max(lats)
        + padding
    )

    min_lon = (
        min(lons)
        - padding
    )

    max_lon = (
        max(lons)
        + padding
    )

    return (
        f"{min_lon},"
        f"{min_lat},"
        f"{max_lon},"
        f"{max_lat}"
    )


def tomtom_incidents(
    route_coordinates: Sequence[
        Tuple[
            float,
            float
        ]
    ],
    api_key: Optional[str] = None,
    timeout: int = REQUEST_TIMEOUT
) -> List[Dict[str, Any]]:

    key = (
        api_key
        or TOMTOM_API_KEY
    ).strip()

    if not key:

        return []

    bbox = _route_bbox(
        route_coordinates
    )

    if not bbox:

        return []

    fields = (
        "{incidents{"
        "type,"
        "geometry{type,coordinates},"
        "properties{"
        "id,"
        "iconCategory,"
        "magnitudeOfDelay,"
        "events{"
        "description,"
        "code,"
        "iconCategory"
        "},"
        "startTime,"
        "endTime,"
        "from,"
        "to,"
        "length,"
        "delay,"
        "roadNumbers"
        "}"
        "}}"
    )

    params = {
        "key":
            key,

        "bbox":
            bbox,

        "fields":
            fields,

        "language":
            "de-DE",

        "timeValidityFilter":
            "present",
    }

    try:

        response = requests.get(
            TOMTOM_INCIDENT_URL,
            params=params,
            timeout=timeout
        )

        if (
            response.status_code
            == 429
        ):

            return [{
                "type":
                    "provider_warning",

                "description":
                    (
                        "TomTom Incident API "
                        "Rate Limit erreicht."
                    )
            }]

        response.raise_for_status()

        data = (
            response.json()
        )

    except Exception as exc:

        return [{
            "type":
                "provider_error",

            "description":
                str(exc)
        }]

    incidents = []

    for item in data.get(
        "incidents",
        []
    ):

        properties = (
            item.get(
                "properties"
            )
            or {}
        )

        descriptions = []

        for event in (
            properties.get(
                "events"
            )
            or []
        ):

            description = (
                event.get(
                    "description"
                )
            )

            if description:

                descriptions.append(
                    str(
                        description
                    )
                )

        incidents.append({
            "id":
                properties.get(
                    "id"
                ),

            "category":
                properties.get(
                    "iconCategory"
                ),

            "magnitude":
                properties.get(
                    "magnitudeOfDelay"
                ),

            "delay_s":
                _safe_int(
                    properties.get(
                        "delay"
                    )
                ),

            "from":
                properties.get(
                    "from"
                ),

            "to":
                properties.get(
                    "to"
                ),

            "length_m":
                properties.get(
                    "length"
                ),

            "roads":
                properties.get(
                    "roadNumbers"
                )
                or [],

            "description":
                " | ".join(
                    descriptions
                ),

            "start_time":
                properties.get(
                    "startTime"
                ),

            "end_time":
                properties.get(
                    "endTime"
                ),

            "geometry":
                item.get(
                    "geometry"
                ),
        })

    return incidents


# ============================================================
# TOMTOM PROVIDER
# ============================================================

class TomTomProvider:

    name = (
        "TomTom Traffic"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_sample_points: int = DEFAULT_SAMPLE_POINTS,
        timeout: int = REQUEST_TIMEOUT,
        **kwargs
    ):
        """
        Hauptprovider für Live-Verkehr.

        timeout und **kwargs sind absichtlich enthalten,
        damit ältere Aufrufe nicht sofort abstürzen.
        """

        self.api_key = (
            api_key
            or os.getenv(
                "TOMTOM_API_KEY",
                ""
            )
        ).strip()

        self.max_sample_points = max(
            1,
            int(
                max_sample_points
            )
        )

        self.timeout = max(
            1,
            int(
                timeout
            )
        )

        self.extra_config = (
            kwargs
        )

    # --------------------------------------------------------
    # HAUPTFUNKTION
    # --------------------------------------------------------

    def get_traffic(
        self,
        route: Any,
        *_,
        **__
    ) -> TrafficResult:

        coordinates = (
            normalize_route(
                route
            )
        )

        if not coordinates:

            return TrafficResult(
                provider=
                    self.name,

                delay_s=
                    0,

                score=
                    0.0,

                confidence=
                    0.0,

                incidents=
                    [],

                debug={
                    "status":
                        "no_route_coordinates",

                    "note":
                        (
                            "Keine verwertbare "
                            "OSRM-Routengeometrie erhalten."
                        ),

                    "received_type":
                        type(
                            route
                        ).__name__,
                }
            )

        if not self.api_key:

            return TrafficResult(
                provider=
                    self.name,

                delay_s=
                    0,

                score=
                    0.0,

                confidence=
                    0.0,

                incidents=
                    [],

                debug={
                    "status":
                        "missing_api_key",

                    "note":
                        (
                            "TOMTOM_API_KEY fehlt. "
                            "Bitte in Hugging Face "
                            "als Secret hinterlegen."
                        )
                }
            )

        sampled = sample_route(
            coordinates,
            self.max_sample_points
        )

        samples: List[
            Dict[
                str,
                Any
            ]
        ] = []

        errors: List[
            Dict[
                str,
                Any
            ]
        ] = []

        # ----------------------------------------------------
        # FLOW JE ROUTENPUNKT
        # ----------------------------------------------------

        for lat, lon in sampled:

            try:

                result = (
                    tomtom_flow(
                        lat=
                            lat,

                        lon=
                            lon,

                        api_key=
                            self.api_key,

                        timeout=
                            self.timeout
                    )
                )

                samples.append(
                    result
                )

            except Exception as exc:

                errors.append({
                    "lat":
                        lat,

                    "lon":
                        lon,

                    "error":
                        str(exc)
                })

        # ----------------------------------------------------
        # KEIN EINZIGER ERFOLGREICHER REQUEST
        # ----------------------------------------------------

        if not samples:

            return TrafficResult(
                provider=
                    self.name,

                delay_s=
                    0,

                score=
                    0.0,

                confidence=
                    0.0,

                incidents=
                    [],

                debug={
                    "status":
                        "tomtom_failed",

                    "route_points":
                        len(
                            coordinates
                        ),

                    "sample_points":
                        len(
                            sampled
                        ),

                    "errors":
                        errors,
                }
            )

        # ----------------------------------------------------
        # FLOW AUSWERTEN
        # ----------------------------------------------------

        scores = [
            _safe_float(
                sample.get(
                    "score"
                )
            )
            for sample
            in samples
        ]

        confidences = [
            _safe_float(
                sample.get(
                    "confidence"
                )
            )
            for sample
            in samples
        ]

        delays = [
            _safe_int(
                sample.get(
                    "delay_s"
                )
            )
            for sample
            in samples
        ]

        closures = [
            sample
            for sample
            in samples
            if sample.get(
                "road_closure"
            )
        ]

        average_score = (
            sum(scores)
            / len(scores)
            if scores
            else 0.0
        )

        average_confidence = (
            sum(confidences)
            / len(confidences)
            if confidences
            else 0.0
        )

        # ----------------------------------------------------
        # DELAY
        #
        # FlowSegmentData liefert lokale Segmentwerte.
        # Wir verwenden die Summe der Stichproben zunächst
        # als Live-Zuschlagsindikator.
        #
        # Später kann dies mit Route Monitoring / historischen
        # Daten noch genauer modelliert werden.
        # ----------------------------------------------------

        flow_delay_s = sum(
            delays
        )

        # ----------------------------------------------------
        # INCIDENTS
        # ----------------------------------------------------

        incidents = (
            tomtom_incidents(
                route_coordinates=
                    coordinates,

                api_key=
                    self.api_key,

                timeout=
                    self.timeout
            )
        )

        incident_delay_s = sum(
            _safe_int(
                incident.get(
                    "delay_s"
                )
            )
            for incident
            in incidents
            if isinstance(
                incident,
                dict
            )
            and incident.get(
                "type"
            )
            not in {
                "provider_error",
                "provider_warning"
            }
        )

        # Incident Delay wird NICHT zusätzlich blind
        # auf Flow Delay addiert, weil derselbe Stau sonst
        # doppelt gezählt werden könnte.

        return TrafficResult(
            provider=
                self.name,

            delay_s=
                int(
                    flow_delay_s
                ),

            score=
                round(
                    _clamp(
                        average_score,
                        0.0,
                        1.0
                    ),
                    3
                ),

            confidence=
                round(
                    _clamp(
                        average_confidence,
                        0.0,
                        1.0
                    ),
                    3
                ),

            incidents=
                incidents,

            debug={
                "status":
                    "ok",

                "provider":
                    "TomTom Traffic API",

                "route_points":
                    len(
                        coordinates
                    ),

                "sample_points_requested":
                    len(
                        sampled
                    ),

                "sample_points_successful":
                    len(
                        samples
                    ),

                "sample_points_failed":
                    len(
                        errors
                    ),

                "road_closures":
                    len(
                        closures
                    ),

                "flow_delay_s":
                    flow_delay_s,

                "incident_delay_s":
                    incident_delay_s,

                "average_score":
                    round(
                        average_score,
                        3
                    ),

                "average_confidence":
                    round(
                        average_confidence,
                        3
                    ),

                "flow_samples":
                    samples,

                "errors":
                    errors,
            }
        )

    # --------------------------------------------------------
    # ALTERNATIVE METHODENNAMEN
    # --------------------------------------------------------

    def analyze_route(
        self,
        route: Any,
        duration_s: Optional[float] = None,
        *args,
        **kwargs
    ) -> TrafficResult:
        """
        Abwärtskompatibilität für ältere app.py-Versionen.
        """

        return self.get_traffic(
            route
        )

    def fetch(
        self,
        route: Any,
        *args,
        **kwargs
    ) -> TrafficResult:

        return self.get_traffic(
            route
        )

    def calculate(
        self,
        route: Any,
        *args,
        **kwargs
    ) -> TrafficResult:

        return self.get_traffic(
            route
        )

    def __call__(
        self,
        route: Any,
        *args,
        **kwargs
    ) -> TrafficResult:

        return self.get_traffic(
            route
        )


# ============================================================
# ABWÄRTSKOMPATIBILITÄT
# ============================================================

class AutobahnProvider(
    TomTomProvider
):
    """
    Nur noch für alte Imports vorhanden.

    Neue app.py sollte verwenden:

        from src.traffic import TomTomProvider

    Intern wird ausschließlich TomTom verwendet.
    """

    name = (
        "TomTom Traffic"
    )


# ============================================================
# COMBINE
# ============================================================

def combine(
    results,
    weights=None
) -> Dict[str, Any]:
    """
    Kompatible Aggregation für ältere Projektversionen.

    Akzeptiert beispielsweise:

        combine(
            [traffic_result],
            {"TomTom Traffic": 1.0}
        )

    Rückgabe ist bewusst ein Dictionary.
    """

    if results is None:

        results = []

    if not isinstance(
        results,
        (
            list,
            tuple
        )
    ):

        results = [
            results
        ]

    normalized: List[
        Dict[
            str,
            Any
        ]
    ] = []

    for result in results:

        if result is None:
            continue

        if isinstance(
            result,
            TrafficResult
        ):

            normalized.append(
                result.to_dict()
            )

        elif isinstance(
            result,
            dict
        ):

            normalized.append(
                result
            )

    if not normalized:

        return {
            "delay_s":
                0,

            "score":
                0.0,

            "confidence":
                0.0,

            "incidents":
                [],

            "provider_results":
                [],
        }

    # --------------------------------------------------------
    # DELAY
    # --------------------------------------------------------

    delay_s = max(
        _safe_int(
            result.get(
                "delay_s"
            )
        )
        for result
        in normalized
    )

    # --------------------------------------------------------
    # SCORE GEWICHTET MIT CONFIDENCE
    # --------------------------------------------------------

    confidence_weights = [
        max(
            0.01,
            _safe_float(
                result.get(
                    "confidence"
                )
            )
        )
        for result
        in normalized
    ]

    weight_sum = sum(
        confidence_weights
    )

    score = (
        sum(
            _safe_float(
                result.get(
                    "score"
                )
            )
            * weight
            for result, weight
            in zip(
                normalized,
                confidence_weights
            )
        )
        / weight_sum
        if weight_sum
        else 0.0
    )

    confidence = (
        sum(
            _safe_float(
                result.get(
                    "confidence"
                )
            )
            for result
            in normalized
        )
        / len(
            normalized
        )
    )

    incidents = []

    for result in normalized:

        incidents.extend(
            result.get(
                "incidents"
            )
            or []
        )

    return {
        "delay_s":
            int(
                delay_s
            ),

        "score":
            round(
                _clamp(
                    score,
                    0.0,
                    1.0
                ),
                3
            ),

        "confidence":
            round(
                _clamp(
                    confidence,
                    0.0,
                    1.0
                ),
                3
            ),

        "incidents":
            incidents,

        "provider_results":
            normalized,
    }


# ============================================================
# EINFACHER WRAPPER
# ============================================================

def calculate_live_traffic(
    route: Any,
    api_key: Optional[str] = None
) -> Dict[str, Any]:

    provider = TomTomProvider(
        api_key=api_key
    )

    result = (
        provider.get_traffic(
            route
        )
    )

    return {
        "delay_s":
            result.delay_s,

        "score":
            result.score,

        "confidence":
            result.confidence,

        "incidents":
            result.incidents,

        "provider_results":
            [
                result.to_dict()
            ],
    }


# ============================================================
# STATUS / DEBUG
# ============================================================

def tomtom_status() -> Dict[str, Any]:
    """
    Gibt niemals den tatsächlichen API-Key aus.
    """

    key = os.getenv(
        "TOMTOM_API_KEY",
        ""
    ).strip()

    return {
        "provider":
            "TomTom Traffic",

        "configured":
            bool(key),

        "key_length":
            len(key)
            if key
            else 0,

        "sample_points":
            DEFAULT_SAMPLE_POINTS,

        "timeout_seconds":
            REQUEST_TIMEOUT,
    }


# ============================================================
# LOKALER TEST
# ============================================================

if __name__ == "__main__":

    import json

    print(
        json.dumps(
            tomtom_status(),
            ensure_ascii=False,
            indent=2
        )
    )

    if TOMTOM_API_KEY:

        # Stuttgart Testpunkte im OSRM-Format:
        # [longitude, latitude]

        test_route = {
            "type":
                "LineString",

            "coordinates": [
                [
                    9.1829,
                    48.7758
                ],
                [
                    9.1900,
                    48.7800
                ],
                [
                    9.2050,
                    48.7900
                ],
                [
                    9.2200,
                    48.8000
                ],
            ]
        }

        provider = (
            TomTomProvider()
        )

        result = (
            provider.get_traffic(
                test_route
            )
        )

        print(
            json.dumps(
                result.to_dict(),
                ensure_ascii=False,
                indent=2,
                default=str
            )
        )