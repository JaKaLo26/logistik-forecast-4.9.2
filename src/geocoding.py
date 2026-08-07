from __future__ import annotations

import os
import time
import requests


class Geocoder:
    """Kostenloser Multi-Provider-Geocoder: Photon, danach Nominatim."""

    def __init__(self, timeout: int = 20):
        self.timeout = timeout
        self.photon_base_url = os.getenv(
            "PHOTON_BASE_URL", "https://photon.komoot.io"
        ).rstrip("/")
        self.nominatim_base_url = os.getenv(
            "NOMINATIM_BASE_URL", "https://nominatim.openstreetmap.org"
        ).rstrip("/")
        self.contact_email = os.getenv(
            "GEOCODER_CONTACT_EMAIL", "lagerforecast@example.com"
        )
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"LagerForecast/4.9.3 (contact: {self.contact_email})",
            "Accept": "application/json",
            "Accept-Language": "de,en;q=0.8",
        })
        self.last_nominatim_request = 0.0

    def search(self, address: str, limit: int = 5) -> list[dict]:
        address = str(address or "").strip()
        if not address:
            return []

        try:
            hits = self._photon(address, limit)
            if hits:
                return hits
        except Exception:
            pass

        try:
            return self._nominatim(address, limit)
        except Exception:
            return []

    def _photon(self, address: str, limit: int) -> list[dict]:
        r = self.session.get(
            f"{self.photon_base_url}/api/",
            params={"q": address, "limit": min(max(int(limit), 1), 10), "lang": "de"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        out = []
        for idx, feature in enumerate(r.json().get("features", [])):
            coords = (feature.get("geometry") or {}).get("coordinates") or []
            if len(coords) < 2:
                continue

            lon, lat = coords[:2]
            p = feature.get("properties") or {}
            countrycode = str(p.get("countrycode") or "").lower()
            if countrycode and countrycode not in {"de", "deu"}:
                continue

            street = p.get("street") or p.get("name")
            number = p.get("housenumber")
            postcode = p.get("postcode")
            city = (
                p.get("city")
                or p.get("town")
                or p.get("village")
                or p.get("municipality")
            )
            first = " ".join(str(x) for x in [street, number] if x)
            second = " ".join(str(x) for x in [postcode, city] if x)
            display = ", ".join(x for x in [first, second, "Deutschland"] if x)

            if number:
                confidence = 0.90
            elif p.get("street"):
                confidence = 0.80
            elif city:
                confidence = 0.67
            else:
                confidence = max(0.50, 0.75 - idx * 0.05)

            out.append({
                "display_name": display,
                "lat": float(lat),
                "lon": float(lon),
                "confidence": round(confidence, 3),
                "provider": "Photon",
                "raw": feature,
            })
        return out

    def _nominatim(self, address: str, limit: int) -> list[dict]:
        elapsed = time.monotonic() - self.last_nominatim_request
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)

        r = self.session.get(
            f"{self.nominatim_base_url}/search",
            params={
                "q": address,
                "format": "jsonv2",
                "addressdetails": 1,
                "limit": min(max(int(limit), 1), 10),
                "countrycodes": "de",
                "email": self.contact_email,
            },
            timeout=self.timeout,
        )
        self.last_nominatim_request = time.monotonic()
        r.raise_for_status()

        out = []
        for item in r.json():
            importance = float(item.get("importance") or 0)
            out.append({
                "display_name": item.get("display_name", ""),
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
                "confidence": round(min(0.92, 0.45 + importance), 3),
                "provider": "Nominatim",
                "raw": item,
            })
        return out
