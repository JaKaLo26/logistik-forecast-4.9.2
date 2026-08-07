from __future__ import annotations
import requests


class OSRMRouter:
    def __init__(self, base_url: str, timeout: int = 20):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    @staticmethod
    def _coord_string(coordinates):
        return ";".join(f"{lon},{lat}" for lat, lon in coordinates)

    def route(self, coordinates: list[tuple[float, float]]) -> dict:
        if len(coordinates) < 2:
            return {"distance_m": 0, "duration_s": 0, "geometry": [], "legs": []}

        url = f"{self.base_url}/route/v1/driving/{self._coord_string(coordinates)}"
        r = self.session.get(
            url,
            params={
                "overview": "full",
                "geometries": "geojson",
                "steps": "true",
                "annotations": "true",
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        payload = r.json()
        if payload.get("code") != "Ok" or not payload.get("routes"):
            raise RuntimeError(f"OSRM: {payload.get('message', 'keine Route')}")

        route = payload["routes"][0]
        return {
            "distance_m": route["distance"],
            "duration_s": route["duration"],
            "geometry": [
                (lat, lon) for lon, lat in route["geometry"]["coordinates"]
            ],
            "legs": route.get("legs", []),
            "raw": payload,
        }

    def optimize_roundtrip(
        self,
        depot: tuple[float, float],
        stops: list[dict],
    ) -> dict:
        """
        Optimiert die Reihenfolge der Stopps über OSRM Trip.
        Depot ist erster Punkt, roundtrip=true und source=first.
        """
        if not stops:
            return {
                "distance_m": 0,
                "duration_s": 0,
                "geometry": [],
                "ordered_stops": [],
                "legs": [],
            }

        coordinates = [depot] + [
            (float(s["lat"]), float(s["lon"])) for s in stops
        ]
        url = f"{self.base_url}/trip/v1/driving/{self._coord_string(coordinates)}"

        r = self.session.get(
            url,
            params={
                "roundtrip": "true",
                "source": "first",
                "overview": "full",
                "geometries": "geojson",
                "steps": "true",
                "annotations": "true",
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        payload = r.json()

        if payload.get("code") != "Ok" or not payload.get("trips"):
            # Fallback: Reihenfolge unverändert, aber Route berechnen.
            fallback = self.route(coordinates + [depot])
            fallback["ordered_stops"] = stops
            fallback["optimizer"] = "fallback-route"
            return fallback

        trip = payload["trips"][0]
        waypoints = payload.get("waypoints", [])

        # waypoint_index ist die Position in der optimierten Rundreise.
        indexed = []
        for original_idx, wp in enumerate(waypoints):
            if original_idx == 0:
                continue  # Depot
            indexed.append((int(wp.get("waypoint_index", original_idx)), original_idx - 1))
        indexed.sort()

        ordered_stops = [
            stops[stop_idx]
            for _, stop_idx in indexed
            if 0 <= stop_idx < len(stops)
        ]

        return {
            "distance_m": trip["distance"],
            "duration_s": trip["duration"],
            "geometry": [
                (lat, lon) for lon, lat in trip["geometry"]["coordinates"]
            ],
            "legs": trip.get("legs", []),
            "ordered_stops": ordered_stops,
            "optimizer": "osrm-trip",
            "raw": payload,
        }
