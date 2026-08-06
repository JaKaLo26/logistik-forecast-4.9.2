from __future__ import annotations
import requests

class Geocoder:
    def __init__(self, base_url: str, timeout: int = 20):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'LogistikForecast/4.9.2 contact@example.invalid'})

    def search(self, address: str, limit: int = 5) -> list[dict]:
        r = self.session.get(f'{self.base_url}/search', params={
            'q': address, 'format': 'jsonv2', 'addressdetails': 1, 'limit': limit, 'countrycodes': 'de'
        }, timeout=self.timeout)
        r.raise_for_status()
        out = []
        for item in r.json():
            importance = float(item.get('importance') or 0)
            out.append({
                'display_name': item.get('display_name', ''),
                'lat': float(item['lat']), 'lon': float(item['lon']),
                'confidence': round(min(1.0, 0.45 + importance), 3),
                'raw': item,
            })
        return out
