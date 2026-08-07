"""
traffic.py
Logistik Forecast - Live Traffic Provider

Primärer Provider:
    TomTom Traffic Flow

Optional ergänzend:
    TomTom Traffic Incidents

Benötigte Umgebungsvariable:
    TOMTOM_API_KEY

Installation:
    pip install requests
"""

from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


# ============================================================
# KONFIGURATION
# ============================================================

TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "").strip()

TOMTOM_FLOW_URL = (
    "https://api.tomtom.com/traffic/services/4/"
    "flowSegmentData/absolute/10/json"
)

TOMTOM_INCIDENT_URL = (
    "https://api.tomtom.com/traffic/services/5/"
    "incidentDetails"
)

REQUEST_TIMEOUT = 8

# Maximal so viele Flow-Punkte je Route abfragen.
# Bei langen Touren kann später erhöht werden.
DEFAULT_SAMPLE_POINTS = 12

# Kleine Pause zwischen Requests.
# Reduziert unnötige Request-Spitzen.
REQUEST_PAUSE_S = 0.05


# ============================================================
# DATENSTRUKTUR
# ============================================================

@dataclass
class TrafficResult:
    provider: str
    delay_s: int
    score: float
    confidence: float
    incidents: List[Dict[str, Any]]
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


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


def _normalize_coordinate(
    coordinate: Any,
) -> Optional[Tuple[float, float]]:
    """
    Gibt Koordinate als (lat, lon) zurück.

    Unterstützt:
        (lat, lon)
        [lat, lon]

        {"lat": ..., "lon": ...}
        {"latitude": ..., "longitude": ...}

    Falls OSRM-Koordinaten als [lon, lat] geliefert werden,
    sollte beim Aufruf osrm_lon_lat=True verwendet werden.
    """

    if isinstance(coordinate, dict):
        lat = (
            coordinate.get("lat")
            if coordinate.get("lat") is not None
            else coordinate.get("latitude")
        )

        lon = (
            coordinate.get("lon")
            if coordinate.get("lon") is not None
            else coordinate.get("lng")
        )

        if lon is None:
            lon = coordinate.get("longitude")

        if lat is not None and lon is not None:
            return float(lat), float(lon)

    if isinstance(coordinate, (list, tuple)) and len(coordinate) >= 2:
        return float(coordinate[0]), float(coordinate[1])

    return None


def normalize_route_coordinates(
    coordinates: Sequence[Any],
    osrm_lon_lat: bool = False,
) -> List[Tuple[float, float]]:
    """
    Wandelt Routenkoordinaten nach (lat, lon) um.

    OSRM GeoJSON liefert normalerweise:
        [longitude, latitude]

    Dann:
        osrm_lon_lat=True
    """

    result: List[Tuple[float, float]] = []

    for item in coordinates:
        coord = _normalize_coordinate(item)

        if coord is None:
            continue

        a, b = coord

        if osrm_lon_lat:
            lon = a
            lat = b
        else:
            lat = a
            lon = b

        if -90 <= lat <= 90 and -180 <= lon <= 180:
            result.append((lat, lon))

    return result


def sample_route(
    coordinates: Sequence[Tuple[float, float]],
    max_points: int = DEFAULT_SAMPLE_POINTS,
) -> List[Tuple[float, float]]:
    """
    Nimmt gleichmäßig verteilte Punkte der Route.

    Dadurch wird nicht jeder einzelne OSRM-Geometriepunkt
    an TomTom geschickt.
    """

    coords = list(coordinates)

    if not coords:
        return []

    if len(coords) <= max_points:
        return coords

    if max_points <= 1:
        return [coords[len(coords) // 2]]

    selected: List[Tuple[float, float]] = []

    for i in range(max_points):
        index = round(i * (len(coords) - 1) / (max_points - 1))
        selected.append(coords[index])

    # Duplikate entfernen
    deduplicated = []

    for point in selected:
        if point not in deduplicated:
            deduplicated.append(point)

    return deduplicated


# ============================================================
# TOMTOM FLOW
# ============================================================

def get_tomtom_flow(
    lat: float,
    lon: float,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:

    key = (api_key or TOMTOM_API_KEY).strip()

    if not key:
        raise RuntimeError(
            "TOMTOM_API_KEY fehlt. "
            "Bitte als Umgebungsvariable/Secret hinterlegen."
        )

    params = {
        "key": key,
        "point": f"{lat},{lon}",
        "unit": "kmph",
    }

    response = requests.get(
        TOMTOM_FLOW_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code == 403:
        raise RuntimeError(
            "TomTom antwortet mit 403. "
            "API-Key ist ungültig oder Traffic API ist nicht freigeschaltet."
        )

    if response.status_code == 429:
        raise RuntimeError(
            "TomTom Rate Limit erreicht (HTTP 429)."
        )

    response.raise_for_status()

    data = response.json()

    flow = data.get("flowSegmentData", {})

    return {
        "lat": lat,
        "lon": lon,

        "frc": flow.get("frc"),

        "current_speed_kmh":
            _safe_float(flow.get("currentSpeed")),

        "free_flow_speed_kmh":
            _safe_float(flow.get("freeFlowSpeed")),

        "current_travel_time_s":
            _safe_int(flow.get("currentTravelTime")),

        "free_flow_travel_time_s":
            _safe_int(flow.get("freeFlowTravelTime")),

        "confidence":
            _safe_float(flow.get("confidence")),

        "road_closure":
            bool(flow.get("roadClosure", False)),
    }


# ============================================================
# FLOW SCORE
# ============================================================

def calculate_flow_score(flow: Dict[str, Any]) -> float:
    """
    score:
        0.0 = frei
        1.0 = sehr dichter Verkehr / Stillstand

    Grundlage:
        currentSpeed / freeFlowSpeed
    """

    current_speed = _safe_float(
        flow.get("current_speed_kmh")
    )

    free_speed = _safe_float(
        flow.get("free_flow_speed_kmh")
    )

    if flow.get("road_closure"):
        return 1.0

    if free_speed <= 0:
        return 0.0

    speed_ratio = current_speed / free_speed

    score = 1.0 - speed_ratio

    return round(_clamp(score, 0.0, 1.0), 3)


def calculate_delay(flow: Dict[str, Any]) -> int:

    current = _safe_int(
        flow.get("current_travel_time_s")
    )

    free = _safe_int(
        flow.get("free_flow_travel_time_s")
    )

    return max(0, current - free)


# ============================================================
# TOMTOM INCIDENTS
# ============================================================

def _bbox_from_coordinates(
    coordinates: Sequence[Tuple[float, float]],
    padding_deg: float = 0.01,
) -> Optional[str]:

    if not coordinates:
        return None

    lats = [x[0] for x in coordinates]
    lons = [x[1] for x in coordinates]

    min_lat = min(lats) - padding_deg
    max_lat = max(lats) + padding_deg

    min_lon = min(lons) - padding_deg
    max_lon = max(lons) + padding_deg

    # TomTom:
    # minLon,minLat,maxLon,maxLat
    return (
        f"{min_lon},{min_lat},"
        f"{max_lon},{max_lat}"
    )


def get_tomtom_incidents(
    coordinates: Sequence[Tuple[float, float]],
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:

    key = (api_key or TOMTOM_API_KEY).strip()

    if not key or not coordinates:
        return []

    bbox = _bbox_from_coordinates(coordinates)

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

    except Exception as exc:

        return [{
            "type": "provider_error",
            "description": str(exc),
        }]

    result: List[Dict[str, Any]] = []

    for incident in data.get("incidents", []):

        properties = incident.get("properties", {})

        events = properties.get("events") or []

        descriptions = []

        for event in events:
            description = event.get("description")

            if description:
                descriptions.append(description)

        result.append({
            "id":
                properties.get("id"),

            "category":
                properties.get("iconCategory"),

            "delay_s":
                _safe_int(properties.get("delay")),

            "magnitude":
                properties.get("magnitudeOfDelay"),

            "from":
                properties.get("from"),

            "to":
                properties.get("to"),

            "length_m":
                properties.get("length"),

            "roads":
                properties.get("roadNumbers", []),

            "description":
                " | ".join(descriptions),

            "geometry":
                incident.get("geometry"),

            "start_time":
                properties.get("startTime"),

            "end_time":
                properties.get("endTime"),
        })

    return result


# ============================================================
# ROUTEN-TRAFFIC
# ============================================================

def get_tomtom_route_traffic(
    route_coordinates: Sequence[Any],
    api_key: Optional[str] = None,
    max_sample_points: int = DEFAULT_SAMPLE_POINTS,
    osrm_lon_lat: bool = True,
    include_incidents: bool = True,
) -> TrafficResult:
    """
    Hauptfunktion.

    Erwartet normalerweise OSRM GeoJSON Koordinaten:

        [
            [longitude, latitude],
            [longitude, latitude],
            ...
        ]

    Deshalb ist:
        osrm_lon_lat=True

    Standard.
    """

    coords = normalize_route_coordinates(
        route_coordinates,
        osrm_lon_lat=osrm_lon_lat,
    )

    if not coords:
        return TrafficResult(
            provider="TomTom Traffic",
            delay_s=0,
            score=0.0,
            confidence=0.0,
            incidents=[],
            debug={
                "status": "no_route_coordinates",
                "note":
                    "Keine gültigen Routenkoordinaten erhalten.",
            },
        )

    sampled = sample_route(
        coords,
        max_points=max_sample_points,
    )

    flows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for lat, lon in sampled:

        try:

            flow = get_tomtom_flow(
                lat=lat,
                lon=lon,
                api_key=api_key,
            )

            flow["score"] = calculate_flow_score(flow)
            flow["delay_s"] = calculate_delay(flow)

            flows.append(flow)

        except Exception as exc:

            errors.append(
                f"{lat:.5f},{lon:.5f}: {exc}"
            )

        time.sleep(REQUEST_PAUSE_S)

    if not flows:

        return TrafficResult(
            provider="TomTom Traffic",
            delay_s=0,
            score=0.0,
            confidence=0.0,
            incidents=[],
            debug={
                "status": "provider_failed",
                "sample_points":
                    len(sampled),

                "errors":
                    errors,
            },
        )

    # --------------------------------------------------------
    # Aggregation
    # --------------------------------------------------------

    scores = [
        _safe_float(x.get("score"))
        for x in flows
    ]

    confidences = [
        _safe_float(x.get("confidence"))
        for x in flows
    ]

    delays = [
        _safe_int(x.get("delay_s"))
        for x in flows
    ]

    closures = [
        x
        for x in flows
        if x.get("road_closure")
    ]

    average_score = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0.0
    )

    # Wichtig:
    # Die abgefragten Flow-Segmente können unterschiedlich lang sein.
    # Deshalb addieren wir nicht einfach alle TravelTimes.
    #
    # Für den Forecast verwenden wir hier den mittleren relativen
    # Verkehrsverlust und liefern zusätzlich die Einzelwerte.
    #
    # delay_s dient hier als indikative Summe der beobachteten
    # Segmentverzögerungen.

    total_delay_s = sum(delays)

    incidents: List[Dict[str, Any]] = []

    if include_incidents:

        try:

            incidents = get_tomtom_incidents(
                coords,
                api_key=api_key,
            )

        except Exception as exc:

            errors.append(
                f"Incident API: {exc}"
            )

    # Incident Delay zusätzlich erfassen,
    # aber nicht blind doppelt in Flow-Delay addieren.

    incident_delay_s = sum(
        _safe_int(x.get("delay_s"))
        for x in incidents
        if isinstance(x, dict)
    )

    return TrafficResult(
        provider="TomTom Traffic",

        delay_s=int(total_delay_s),

        score=round(
            _clamp(
                average_score,
                0.0,
                1.0,
            ),
            3,
        ),

        confidence=round(
            _clamp(
                average_confidence,
                0.0,
                1.0,
            ),
            3,
        ),

        incidents=incidents,

        debug={
            "status": "ok",

            "provider":
                "TomTom Traffic API",

            "route_points":
                len(coords),

            "sample_points_requested":
                len(sampled),

            "sample_points_successful":
                len(flows),

            "sample_points_failed":
                len(errors),

            "road_closures":
                len(closures),

            "flow_delay_s":
                total_delay_s,

            "incident_delay_s":
                incident_delay_s,

            "flow_samples":
                flows,

            "errors":
                errors,
        },
    )


# ============================================================
# KOMPATIBILITÄTSFUNKTION
# ============================================================

def calculate_live_traffic(
    route_coordinates: Sequence[Any],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Einfache Wrapper-Funktion für bestehende Forecast-Logik.

    Rückgabe ähnlich deiner bisherigen Struktur.
    """

    result = get_tomtom_route_traffic(
        route_coordinates=route_coordinates,
        api_key=api_key,
        osrm_lon_lat=True,
        include_incidents=True,
    )

    return {
        "delay_s":
            result.delay_s,

        "score":
            result.score,

        "confidence":
            result.confidence,

        "provider_results": [
            result.to_dict()
        ],
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # Stuttgart Beispielroute
    # OSRM-Reihenfolge:
    # [longitude, latitude]

    test_route = [
        [9.1829, 48.7758],
        [9.1900, 48.7800],
        [9.2050, 48.7900],
        [9.2200, 48.8000],
    ]

    traffic = calculate_live_traffic(
        test_route
    )

    import json

    print(
        json.dumps(
            traffic,
            indent=2,
            ensure_ascii=False,
        )
    )