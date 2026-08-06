from __future__ import annotations
import requests

class OSRMRouter:
    def __init__(self, base_url: str, timeout: int = 20):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def route(self, coordinates: list[tuple[float,float]]) -> dict:
        if len(coordinates) < 2:
            return {'distance_m': 0, 'duration_s': 0, 'geometry': [], 'legs': []}
        coord_string = ';'.join(f'{lon},{lat}' for lat, lon in coordinates)
        url = f'{self.base_url}/route/v1/driving/{coord_string}'
        r = requests.get(url, params={'overview':'full','geometries':'geojson','steps':'true','annotations':'true'}, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        if payload.get('code') != 'Ok' or not payload.get('routes'):
            raise RuntimeError(f"OSRM: {payload.get('message','keine Route')}")
        route = payload['routes'][0]
        return {
            'distance_m': route['distance'], 'duration_s': route['duration'],
            'geometry': [(lat, lon) for lon, lat in route['geometry']['coordinates']],
            'legs': route.get('legs', []), 'raw': payload,
        }
