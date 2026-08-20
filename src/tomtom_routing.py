# src/tomtom_routing.py

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

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

TOMTOM_ROUTING_BASE_URL = (
    "https://api.tomtom.com/routing/1/calculateRoute"
)


# ============================================================
# DATENMODELL
# ============================================================

@dataclass
class TomTomSegmentResult:

    from_lat: float
    from_lon: float

    to_lat: float
    to_lon: float

    departure_time: str
    arrival_time: Optional[str]

    distance_m: float

    tomtom_best_s: int
    tomtom_no_traffic_s: int
    tomtom_historic_s: int
    tomtom_live_s: int

    traffic_delay_s: int
    traffic_length_m: int

    success: bool

    error: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(
        self
    ) -> Dict[str, Any]:

        return asdict(
            self
        )


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

        return float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return default


def _normalize_coordinate(
    coordinate: Any
) -> Tuple[float, float]:
    """
    Erwartet bevorzugt:

        (lat, lon)

    Unterstützt zusätzlich:

        {
            "lat": ...,
            "lon": ...
        }
    """

    if isinstance(
        coordinate,
        dict
    ):

        lat = coordinate.get(
            "lat"
        )

        lon = coordinate.get(
            "lon"
        )

        if lon is None:

            lon = coordinate.get(
                "lng"
            )

        if (
            lat is None
            or lon is None
        ):

            raise ValueError(
                "Koordinate enthält kein lat/lon."
            )

        lat = float(
            lat
        )

        lon = float(
            lon
        )

    elif (
        isinstance(
            coordinate,
            (
                list,
                tuple
            )
        )
        and len(
            coordinate
        ) >= 2
    ):

        lat = float(
            coordinate[0]
        )

        lon = float(
            coordinate[1]
        )

    else:

        raise ValueError(
            "Ungültiges Koordinatenformat."
        )

    if not (
        -90
        <= lat
        <= 90
    ):

        raise ValueError(
            f"Ungültige Latitude: {lat}"
        )

    if not (
        -180
        <= lon
        <= 180
    ):

        raise ValueError(
            f"Ungültige Longitude: {lon}"
        )

    return (
        lat,
        lon
    )


def _format_departure_time(
    departure: Optional[Any]
) -> str:
    """
    TomTom akzeptiert ISO-8601-Zeitstempel.

    Unterstützt:
    - datetime
    - ISO-String
    - None -> aktuelle UTC-Zeit
    """

    if departure is None:

        return (
            datetime.now(
                timezone.utc
            )
            .replace(
                microsecond=0
            )
            .isoformat()
        )

    if isinstance(
        departure,
        datetime
    ):

        if departure.tzinfo is None:

            departure = departure.replace(
                tzinfo=timezone.utc
            )

        return (
            departure
            .replace(
                microsecond=0
            )
            .isoformat()
        )

    value = str(
        departure
    ).strip()

    if not value:

        return (
            datetime.now(
                timezone.utc
            )
            .replace(
                microsecond=0
            )
            .isoformat()
        )

    return value


# ============================================================
# TOMTOM SEGMENT ROUTING
# ============================================================

class TomTomRoutingProvider:

    name = (
        "TomTom Routing"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = REQUEST_TIMEOUT,
    ):

        self.api_key = (
            api_key
            or os.getenv(
                "TOMTOM_API_KEY",
                ""
            )
        ).strip()

        self.timeout = max(
            1,
            int(
                timeout
            )
        )

        self.session = (
            requests.Session()
        )

    # --------------------------------------------------------
    # EIN SEGMENT
    # --------------------------------------------------------

    def calculate_segment(
        self,
        origin: Any,
        destination: Any,
        depart_at: Optional[Any] = None,
        vehicle_weight_kg: Optional[int] = None,
        vehicle_height_m: Optional[float] = None,
        vehicle_width_m: Optional[float] = None,
        vehicle_length_m: Optional[float] = None,
        vehicle_max_speed_kmh: Optional[int] = None,
        vehicle_commercial: bool = True,
    ) -> TomTomSegmentResult:

        origin_lat, origin_lon = (
            _normalize_coordinate(
                origin
            )
        )

        destination_lat, destination_lon = (
            _normalize_coordinate(
                destination
            )
        )

        departure_time = (
            _format_departure_time(
                depart_at
            )
        )

        if not self.api_key:

            return TomTomSegmentResult(
                from_lat=
                    origin_lat,

                from_lon=
                    origin_lon,

                to_lat=
                    destination_lat,

                to_lon=
                    destination_lon,

                departure_time=
                    departure_time,

                arrival_time=
                    None,

                distance_m=
                    0.0,

                tomtom_best_s=
                    0,

                tomtom_no_traffic_s=
                    0,

                tomtom_historic_s=
                    0,

                tomtom_live_s=
                    0,

                traffic_delay_s=
                    0,

                traffic_length_m=
                    0,

                success=
                    False,

                error=
                    "TOMTOM_API_KEY fehlt.",

                raw=
                    None
            )

        route_points = (
            f"{origin_lat},{origin_lon}:"
            f"{destination_lat},{destination_lon}"
        )

        url = (
            f"{TOMTOM_ROUTING_BASE_URL}/"
            f"{route_points}/json"
        )

        params: Dict[
            str,
            Any
        ] = {
            "key":
                self.api_key,

            "traffic":
                "true",

            "departAt":
                departure_time,

            "computeTravelTimeFor":
                "all",

            "routeType":
                "fastest",

            "travelMode":
                "truck",

            "vehicleCommercial":
                "true"
                if vehicle_commercial
                else "false",

            "routeRepresentation":
                "summaryOnly",
        }

        # ----------------------------------------------------
        # OPTIONAL: LKW-PARAMETER
        # ----------------------------------------------------

        if (
            vehicle_weight_kg
            is not None
        ):

            params[
                "vehicleWeight"
            ] = int(
                vehicle_weight_kg
            )

        if (
            vehicle_height_m
            is not None
        ):

            params[
                "vehicleHeight"
            ] = float(
                vehicle_height_m
            )

        if (
            vehicle_width_m
            is not None
        ):

            params[
                "vehicleWidth"
            ] = float(
                vehicle_width_m
            )

        if (
            vehicle_length_m
            is not None
        ):

            params[
                "vehicleLength"
            ] = float(
                vehicle_length_m
            )

        if (
            vehicle_max_speed_kmh
            is not None
        ):

            params[
                "vehicleMaxSpeed"
            ] = int(
                vehicle_max_speed_kmh
            )

        try:

            response = (
                self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )
            )

            # ------------------------------------------------
            # SPEZIFISCHE API-FEHLER
            # ------------------------------------------------

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
                    "Routing-API-Berechtigung prüfen."
                )

            if (
                response.status_code
                == 429
            ):

                raise RuntimeError(
                    "TomTom HTTP 429: "
                    "API-Limit erreicht."
                )

            response.raise_for_status()

            payload = (
                response.json()
            )

            routes = (
                payload.get(
                    "routes"
                )
                or []
            )

            if not routes:

                raise RuntimeError(
                    "TomTom hat keine Route geliefert."
                )

            route = (
                routes[0]
            )

            summary = (
                route.get(
                    "summary"
                )
                or {}
            )

            best_s = (
                _safe_int(
                    summary.get(
                        "travelTimeInSeconds"
                    )
                )
            )

            no_traffic_s = (
                _safe_int(
                    summary.get(
                        "noTrafficTravelTimeInSeconds"
                    ),
                    best_s
                )
            )

            historic_s = (
                _safe_int(
                    summary.get(
                        "historicTrafficTravelTimeInSeconds"
                    ),
                    best_s
                )
            )

            live_s = (
                _safe_int(
                    summary.get(
                        "liveTrafficIncidentsTravelTimeInSeconds"
                    ),
                    best_s
                )
            )

            traffic_delay_s = (
                _safe_int(
                    summary.get(
                        "trafficDelayInSeconds"
                    )
                )
            )

            traffic_length_m = (
                _safe_int(
                    summary.get(
                        "trafficLengthInMeters"
                    )
                )
            )

            distance_m = (
                _safe_float(
                    summary.get(
                        "lengthInMeters"
                    )
                )
            )

            arrival_time = (
                summary.get(
                    "arrivalTime"
                )
            )

            return TomTomSegmentResult(
                from_lat=
                    origin_lat,

                from_lon=
                    origin_lon,

                to_lat=
                    destination_lat,

                to_lon=
                    destination_lon,

                departure_time=
                    departure_time,

                arrival_time=
                    arrival_time,

                distance_m=
                    distance_m,

                tomtom_best_s=
                    best_s,

                tomtom_no_traffic_s=
                    no_traffic_s,

                tomtom_historic_s=
                    historic_s,

                tomtom_live_s=
                    live_s,

                traffic_delay_s=
                    traffic_delay_s,

                traffic_length_m=
                    traffic_length_m,

                success=
                    True,

                error=
                    None,

                raw=
                    payload
            )

        except Exception as exc:

            return TomTomSegmentResult(
                from_lat=
                    origin_lat,

                from_lon=
                    origin_lon,

                to_lat=
                    destination_lat,

                to_lon=
                    destination_lon,

                departure_time=
                    departure_time,

                arrival_time=
                    None,

                distance_m=
                    0.0,

                tomtom_best_s=
                    0,

                tomtom_no_traffic_s=
                    0,

                tomtom_historic_s=
                    0,

                tomtom_live_s=
                    0,

                traffic_delay_s=
                    0,

                traffic_length_m=
                    0,

                success=
                    False,

                error=
                    str(
                        exc
                    ),

                raw=
                    None
            )


# ============================================================
# KOMFORTFUNKTION
# ============================================================

def calculate_tomtom_segment(
    origin: Any,
    destination: Any,
    depart_at: Optional[Any] = None,
    api_key: Optional[str] = None,
    timeout: int = REQUEST_TIMEOUT,
    **vehicle_kwargs
) -> Dict[str, Any]:

    provider = (
        TomTomRoutingProvider(
            api_key=api_key,
            timeout=timeout
        )
    )

    result = (
        provider.calculate_segment(
            origin=
                origin,

            destination=
                destination,

            depart_at=
                depart_at,

            **vehicle_kwargs
        )
    )

    return (
        result.to_dict()
    )


# ============================================================
# STATUS
# ============================================================

def tomtom_routing_status() -> Dict[str, Any]:

    key = os.getenv(
        "TOMTOM_API_KEY",
        ""
    ).strip()

    return {
        "provider":
            "TomTom Routing API v1",

        "configured":
            bool(
                key
            ),

        "timeout_seconds":
            REQUEST_TIMEOUT,

        "returns":
            [
                "noTrafficTravelTimeInSeconds",
                "historicTrafficTravelTimeInSeconds",
                "liveTrafficIncidentsTravelTimeInSeconds",
                "trafficDelayInSeconds",
            ],
    }


# ============================================================
# LOKALER TEST
# ============================================================

if __name__ == "__main__":

    import json

    print(
        json.dumps(
            tomtom_routing_status(),
            indent=2,
            ensure_ascii=False
        )
    )

    if TOMTOM_API_KEY:

        # Stuttgart -> Ludwigsburg

        result = (
            calculate_tomtom_segment(
                origin=(
                    48.7758,
                    9.1829
                ),

                destination=(
                    48.8973,
                    9.1916
                ),

                vehicle_weight_kg=
                    40000,

                vehicle_height_m=
                    4.0,

                vehicle_width_m=
                    2.55,

                vehicle_length_m=
                    16.5,

                vehicle_max_speed_kmh=
                    80,
            )
        )

        # raw bewusst beim Terminal-Test entfernen,
        # damit die Ausgabe übersichtlich bleibt.

        result.pop(
            "raw",
            None
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str
            )
        )