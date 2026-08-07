# src/traffic.py

from __future__ import annotations

import os
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


# ============================================================
# KONFIGURATION
# ============================================================

TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "").strip()

REQUEST_TIMEOUT = 10

TOMTOM_FLOW_URL = (
    "https://api.tomtom.com/traffic/services/4/"
    "flowSegmentData/absolute/10/json"
)

TOMTOM_INCIDENT_URL = (
    "https://api.tomtom.com/traffic/services/5/"
    "incidentDetails"
)

# Nicht jeden einzelnen OSRM-Punkt abfragen.
# 12 Punkte reichen zunächst für eine Route.
DEFAULT_SAMPLE_POINTS = 12


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _looks_like_lon_lat(a: float, b: float) -> bool:
    """
    Hilfsheuristik.

    Deutschland:
    Longitude ungefähr 5-16
    Latitude ungefähr 47-55

    OSRM GeoJSON:
        [lon, lat]
    """

    if -180 <= a <= 180 and -90 <= b <= 90:
        if abs(a) <= 30 and abs(b) >= 35:
            return True

    return False


def normalize_coordinate(point: Any) -> Optional[Tuple[float, float]]:
    """
    Gibt immer:
        (lat, lon)

    Unterstützt:
        [lon, lat]      <- OSRM
        (lon, lat)
        {"lat": ..., "lon": ...}
        {"latitude": ..., "longitude": ...}
    """

    if isinstance(point, dict):

        lat = point.get("lat")

        if lat is None:
            lat = point.get("latitude")

        lon = point.get("lon")

        if lon is None:
            lon = point.get("lng")

        if lon is None:
            lon = point.get("longitude")

        if lat is None or lon is None:
            return None

        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            return None

        if not (-90 <= lat <= 90):
            return None

        if not (-180 <= lon <= 180):
            return None

        return lat, lon

    if isinstance(point, (list, tuple)) and len(point) >= 2:

        try:
            a = float(point[0])
            b = float(point[1])
        except (TypeError, ValueError):
            return None

        # OSRM GeoJSON -> [longitude, latitude]
        if _looks_like_lon_lat(a, b):
            lon = a
            lat = b

        else:
            lat = a
            lon = b

        if not (-90 <= lat <= 90):
            return None

        if not (-180 <= lon <= 180):
            return None

        return lat, lon

    return None


def normalize_route(
    route: Any,
) -> List[Tuple[float, float]]:

    if route is None:
        return []

    # --------------------------------------------------------
    # OSRM GeoJSON Geometry
    # --------------------------------------------------------

    if isinstance(route, dict):

        if "coordinates" in route:
            route = route["coordinates"]

        elif "geometry" in route:

            geometry = route["geometry"]

            if isinstance(geometry, dict):
                route = geometry.get("coordinates", [])

        elif "route" in route:
            route = route["route"]

        elif "points" in route:
            route = route["points"]

    if not isinstance(route, (list, tuple)):
        return []

    result: List[Tuple[float, float]] = []

    for point in route:

        normalized = normalize_coordinate(point)

        if normalized is not None:
            result.append(normalized)

    return result


def sample_route(
    route: Sequence[Tuple[float, float]],
    max_points: int = DEFAULT_SAMPLE_POINTS,
) -> List[Tuple[float, float]]:

    points = list(route)

    if not points:
        return []

    if len(points) <= max_points:
        return points

    if max_points <= 1:
        return [points[len(points) // 2]]

    result = []

    for i in range(max_points):

        index = round(
            i * (len(points) - 1)
            / (max_points - 1)
        )

        point = points[index]

        if point not in result:
            result.append(point)

    return result


# ============================================================
# TOMTOM FLOW
# ============================================================

def tomtom_flow(
    lat: float,
    lon: float,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:

    key = (api_key or TOMTOM_API_KEY).strip()

    if not key:
        raise RuntimeError(
            "TOMTOM_API_KEY fehlt. "
            "Bitte im Hugging-Face-Space als Secret "
            "TOMTOM_API_KEY hinterlegen."
        )

    params = {
        "key": key,
        "point": f"{lat},{lon}",
        "unit": "KMPH",
    }

    response = requests.get(
        TOMTOM_FLOW_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code == 401:
        raise RuntimeError(
            "TomTom: HTTP 401 - API-Key nicht akzeptiert."
        )

    if response.status_code == 403:
        raise RuntimeError(
            "TomTom: HTTP 403 - API-Key oder Traffic-API-Berechtigung prüfen."
        )

    if response.status_code == 429:
        raise RuntimeError(
            "TomTom: HTTP 429 - Anfrage-Limit erreicht."
        )

    response.raise_for_status()

    data = response.json()

    flow = data.get("flowSegmentData") or {}

    current_speed = _safe_float(
        flow.get("currentSpeed")
    )

    free_flow_speed = _safe_float(
        flow.get("freeFlowSpeed")
    )

    current_time = _safe_int(
        flow.get("currentTravelTime")
    )

    free_flow_time = _safe_int(
        flow.get("freeFlowTravelTime")
    )

    confidence = _safe_float(
        flow.get("confidence")
    )

    road_closure = bool(
        flow.get("roadClosure", False)
    )

    delay_s = max(
        0,
        current_time - free_flow_time
    )

    if road_closure:
        score = 1.0

    elif free_flow_speed > 0:

        score = 1.0 - (
            current_speed / free_flow_speed
        )

        score = _clamp(
            score,
            0.0,
            1.0
        )

    else:
        score = 0.0

    return {
        "lat": lat,
        "lon": lon,

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
            round(score, 3),

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
            flow.get("frc"),
    }


# ============================================================
# TOMTOM INCIDENTS
# ============================================================

def _route_bbox(
    coordinates: Sequence[Tuple[float, float]],
    padding: float = 0.005,
) -> Optional[str]:

    if not coordinates:
        return None

    lats = [
        point[0]
        for point in coordinates
    ]

    lons = [
        point[1]
        for point in coordinates
    ]

    return (
        f"{min(lons) - padding},"
        f"{min(lats) - padding},"
        f"{max(lons) + padding},"
        f"{max(lats) + padding}"
    )


def tomtom_incidents(
    route_coordinates: Sequence[Tuple[float, float]],
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:

    key = (api_key or TOMTOM_API_KEY).strip()

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
        "events{description,code,iconCategory},"
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
        "key": key,
        "bbox": bbox,
        "fields": fields,
        "language": "de-DE",
        "timeValidityFilter": "present",
    }

    try:

        response = requests.get(
            TOMTOM_INCIDENT_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

    except Exception:
        return []

    incidents = []

    for item in data.get("incidents", []):

        properties = (
            item.get("properties")
            or {}
        )

        descriptions = []

        for event in (
            properties.get("events")
            or []
        ):

            description = event.get(
                "description"
            )

            if description:
                descriptions.append(
                    description
                )

        incidents.append({
            "id":
                properties.get("id"),

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
                    properties.get("delay")
                ),

            "from":
                properties.get("from"),

            "to":
                properties.get("to"),

            "length_m":
                properties.get("length"),

            "roads":
                properties.get(
                    "roadNumbers"
                ) or [],

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
                item.get("geometry"),
        })

    return incidents


# ============================================================
# TOMTOM PROVIDER
# ============================================================

class TomTomProvider:

    name = "TomTom Traffic"

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_sample_points: int = DEFAULT_SAMPLE_POINTS,
    ):

        self.api_key = (
            api_key
            or TOMTOM_API_KEY
        ).strip()

        self.max_sample_points = (
            max_sample_points
        )

    def get_traffic(
        self,
        route: Any,
        *args,
        **kwargs,
    ) -> TrafficResult:

        coordinates = normalize_route(
            route
        )

        if not coordinates:

            return TrafficResult(
                provider=self.name,
                delay_s=0,
                score=0.0,
                confidence=0.0,
                incidents=[],
                debug={
                    "status":
                        "no_route_coordinates",

                    "note":
                        "Keine verwertbare "
                        "OSRM-Routengeometrie erhalten."
                }
            )

        if not self.api_key:

            return TrafficResult(
                provider=self.name,
                delay_s=0,
                score=0.0,
                confidence=0.0,
                incidents=[],
                debug={
                    "status":
                        "missing_api_key",

                    "note":
                        "TOMTOM_API_KEY fehlt."
                }
            )

        sampled = sample_route(
            coordinates,
            self.max_sample_points,
        )

        samples = []
        errors = []

        for lat, lon in sampled:

            try:

                result = tomtom_flow(
                    lat,
                    lon,
                    api_key=self.api_key,
                )

                samples.append(result)

            except Exception as exc:

                errors.append(
                    str(exc)
                )

        if not samples:

            return TrafficResult(
                provider=self.name,
                delay_s=0,
                score=0.0,
                confidence=0.0,
                incidents=[],
                debug={
                    "status":
                        "tomtom_failed",

                    "errors":
                        errors,

                    "sample_points":
                        len(sampled),
                }
            )

        # ----------------------------------------------------
        # Durchschnittlicher Traffic Score
        # ----------------------------------------------------

        scores = [
            _safe_float(
                sample.get("score")
            )
            for sample in samples
        ]

        confidences = [
            _safe_float(
                sample.get("confidence")
            )
            for sample in samples
        ]

        delay_values = [
            _safe_int(
                sample.get("delay_s")
            )
            for sample in samples
        ]

        score = (
            sum(scores)
            / len(scores)
            if scores
            else 0.0
        )

        confidence = (
            sum(confidences)
            / len(confidences)
            if confidences
            else 0.0
        )

        # Die Segmentzeiten werden nicht als komplette
        # Routendauer interpretiert.
        # Sie dienen als Live-Verkehrsindikator.
        delay_s = sum(
            delay_values
        )

        closures = sum(
            1
            for sample in samples
            if sample.get(
                "road_closure"
            )
        )

        incidents = tomtom_incidents(
            coordinates,
            api_key=self.api_key,
        )

        return TrafficResult(
            provider=self.name,

            delay_s=int(delay_s),

            score=round(
                _clamp(
                    score,
                    0.0,
                    1.0
                ),
                3
            ),

            confidence=round(
                _clamp(
                    confidence,
                    0.0,
                    1.0
                ),
                3
            ),

            incidents=incidents,

            debug={
                "status":
                    "ok",

                "route_points":
                    len(coordinates),

                "sample_points":
                    len(sampled),

                "successful_samples":
                    len(samples),

                "failed_samples":
                    len(errors),

                "road_closures":
                    closures,

                "flow_samples":
                    samples,

                "errors":
                    errors,
            }
        )

    # --------------------------------------------------------
    # Mehrere Methodennamen zur Kompatibilität
    # --------------------------------------------------------

    def fetch(
        self,
        route: Any,
        *args,
        **kwargs,
    ) -> TrafficResult:

        return self.get_traffic(
            route,
            *args,
            **kwargs,
        )

    def calculate(
        self,
        route: Any,
        *args,
        **kwargs,
    ) -> TrafficResult:

        return self.get_traffic(
            route,
            *args,
            **kwargs,
        )

    def __call__(
        self,
        route: Any,
        *args,
        **kwargs,
    ) -> TrafficResult:

        return self.get_traffic(
            route,
            *args,
            **kwargs,
        )


# ============================================================
# ABWÄRTSKOMPATIBILITÄT
# ============================================================

class AutobahnProvider(TomTomProvider):
    """
    WICHTIG:

    Der Name bleibt erhalten, weil app.py aktuell noch:

        from src.traffic import AutobahnProvider

    importiert.

    Intern läuft die Klasse aber jetzt über TomTom.

    Dadurch muss app.py zunächst nicht umgeschrieben werden.
    """

    name = "TomTom Traffic"


# ============================================================
# PROVIDER KOMBINIEREN
# ============================================================

def combine(
    *results,
    **kwargs,
) -> TrafficResult:
    """
    Kompatible Aggregationsfunktion.

    Akzeptiert:
        TrafficResult
        dict
        Listen davon

    Dadurch soll bestehender Code, der combine(...)
    verwendet, weiter funktionieren.
    """

    flattened = []

    for result in results:

        if result is None:
            continue

        if isinstance(
            result,
            (list, tuple)
        ):
            flattened.extend(result)

        else:
            flattened.append(result)

    normalized = []

    for result in flattened:

        if isinstance(
            result,
            TrafficResult
        ):
            normalized.append(result)

        elif isinstance(
            result,
            dict
        ):

            normalized.append(
                TrafficResult(
                    provider=str(
                        result.get(
                            "provider",
                            "Traffic Provider"
                        )
                    ),

                    delay_s=_safe_int(
                        result.get(
                            "delay_s"
                        )
                    ),

                    score=_safe_float(
                        result.get(
                            "score"
                        )
                    ),

                    confidence=_safe_float(
                        result.get(
                            "confidence"
                        )
                    ),

                    incidents=result.get(
                        "incidents"
                    ) or [],

                    debug=result.get(
                        "debug"
                    ) or {},
                )
            )

    if not normalized:

        return TrafficResult(
            provider="Traffic",
            delay_s=0,
            score=0.0,
            confidence=0.0,
            incidents=[],
            debug={
                "status":
                    "no_provider_results"
            }
        )

    # --------------------------------------------------------
    # Confidence-gewichteter Score
    # --------------------------------------------------------

    confidence_sum = sum(
        max(
            result.confidence,
            0.01
        )
        for result in normalized
    )

    weighted_score = sum(
        result.score
        * max(
            result.confidence,
            0.01
        )
        for result in normalized
    )

    score = (
        weighted_score
        / confidence_sum
    )

    confidence = sum(
        result.confidence
        for result in normalized
    ) / len(normalized)

    delay_s = max(
        result.delay_s
        for result in normalized
    )

    incidents = []

    for result in normalized:
        incidents.extend(
            result.incidents
            or []
        )

    return TrafficResult(
        provider="Combined Traffic",

        delay_s=int(
            delay_s
        ),

        score=round(
            _clamp(
                score,
                0.0,
                1.0
            ),
            3
        ),

        confidence=round(
            _clamp(
                confidence,
                0.0,
                1.0
            ),
            3
        ),

        incidents=incidents,

        debug={
            "status":
                "ok",

            "provider_results": [
                result.to_dict()
                for result
                in normalized
            ]
        }
    )


# ============================================================
# EINFACHER WRAPPER
# ============================================================

def calculate_live_traffic(
    route: Any,
) -> Dict[str, Any]:

    provider = TomTomProvider()

    result = provider.get_traffic(
        route
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

        "provider_results": [
            result.to_dict()
        ],
    }


# ============================================================
# API-KEY TEST
# ============================================================

def tomtom_status() -> Dict[str, Any]:

    return {
        "provider":
            "TomTom Traffic",

        "api_key_configured":
            bool(TOMTOM_API_KEY),

        "api_key_length":
            len(TOMTOM_API_KEY)
            if TOMTOM_API_KEY
            else 0,
    }