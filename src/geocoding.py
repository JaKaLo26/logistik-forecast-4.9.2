from __future__ import annotations

import os
import time
from typing import Any

import requests


class Geocoder:
    """
    Multi-Provider-Geocoder für LagerForecast 4.9.2.

    Reihenfolge:
    1. HERE Geocoding & Search (wenn HERE_API_KEY vorhanden)
    2. Photon / Komoot
    3. Nominatim als letzter Fallback, gedrosselt

    search() liefert immer dieselbe interne Struktur zurück:
    {
        "display_name": str,
        "lat": float,
        "lon": float,
        "confidence": float,
        "provider": str,
        "raw": dict
    }
    """

    def __init__(self, base_url: str | None = None, timeout: int = 20):
        self.timeout = timeout

        # Bestehende Konfiguration weiter unterstützen
        self.nominatim_base_url = (
            base_url
            or os.getenv("NOMINATIM_BASE_URL")
            or "https://nominatim.openstreetmap.org"
        ).rstrip("/")

        self.here_api_key = os.getenv("HERE_API_KEY", "").strip()
        self.here_base_url = os.getenv(
            "HERE_GEOCODING_BASE_URL",
            "https://geocode.search.hereapi.com/v1",
        ).rstrip("/")

        self.photon_base_url = os.getenv(
            "PHOTON_BASE_URL",
            "https://photon.komoot.io",
        ).rstrip("/")

        self.contact_email = os.getenv(
            "GEOCODER_CONTACT_EMAIL",
            "lagerforecast@example.com",
        ).strip()

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    f"LagerForecast/4.9.2 "
                    f"(contact: {self.contact_email})"
                ),
                "Accept": "application/json",
                "Accept-Language": "de,en;q=0.8",
            }
        )

        self.last_nominatim_request = 0.0

    def search(self, address: str, limit: int = 5) -> list[dict]:
        address = str(address or "").strip()
        if not address:
            return []

        errors: list[str] = []

        # 1) HERE
        if self.here_api_key:
            try:
                hits = self._search_here(address, limit)
                if hits:
                    return hits
            except Exception as exc:
                errors.append(f"HERE: {type(exc).__name__}: {exc}")

        # 2) Photon
        try:
            hits = self._search_photon(address, limit)
            if hits:
                return hits
        except Exception as exc:
            errors.append(f"Photon: {type(exc).__name__}: {exc}")

        # 3) Nominatim
        try:
            hits = self._search_nominatim(address, limit)
            if hits:
                return hits
        except Exception as exc:
            errors.append(f"Nominatim: {type(exc).__name__}: {exc}")

        # Alle Provider ohne Treffer / nicht erreichbar.
        # Kein harter Fehler nötig: app.py markiert den Datensatz anschließend
        # als "MANUELL PRÜFEN".
        return []

    # ------------------------------------------------------------------
    # HERE
    # ------------------------------------------------------------------

    def _search_here(self, address: str, limit: int) -> list[dict]:
        response = self.session.get(
            f"{self.here_base_url}/geocode",
            params={
                "q": address,
                "in": "countryCode:DEU",
                "limit": max(1, min(int(limit), 20)),
                "lang": "de-DE",
                "apiKey": self.here_api_key,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        items = data.get("items", [])

        out: list[dict] = []

        for idx, item in enumerate(items):
            position = item.get("position") or {}
            lat = position.get("lat")
            lon = position.get("lng")

            if lat is None or lon is None:
                continue

            address_obj = item.get("address") or {}
            title = item.get("title") or self._format_here_address(address_obj)

            result_type = str(item.get("resultType") or "").lower()
            house_number_type = str(
                item.get("houseNumberType") or ""
            ).lower()

            # Konservative interne Vertrauensbewertung.
            if result_type in {"houseNumber".lower(), "pointaddress"}:
                confidence = 0.97
            elif result_type in {"street", "intersection"}:
                confidence = 0.84
            elif result_type in {"locality", "administrativearea"}:
                confidence = 0.65
            else:
                confidence = max(0.55, 0.90 - idx * 0.06)

            if house_number_type == "pa":
                confidence = min(0.99, confidence + 0.01)

            out.append(
                {
                    "display_name": title,
                    "lat": float(lat),
                    "lon": float(lon),
                    "confidence": round(float(confidence), 3),
                    "provider": "HERE",
                    "raw": item,
                }
            )

        return out

    @staticmethod
    def _format_here_address(address_obj: dict[str, Any]) -> str:
        parts = [
            address_obj.get("street"),
            address_obj.get("houseNumber"),
            address_obj.get("postalCode"),
            address_obj.get("city"),
            address_obj.get("countryName"),
        ]
        return ", ".join(str(x) for x in parts if x)

    # ------------------------------------------------------------------
    # Photon
    # ------------------------------------------------------------------

    def _search_photon(self, address: str, limit: int) -> list[dict]:
        response = self.session.get(
            f"{self.photon_base_url}/api/",
            params={
                "q": address,
                "limit": max(1, min(int(limit), 10)),
                "lang": "de",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        features = data.get("features", [])

        out: list[dict] = []

        for idx, feature in enumerate(features):
            geometry = feature.get("geometry") or {}
            coordinates = geometry.get("coordinates") or []

            if len(coordinates) < 2:
                continue

            lon, lat = coordinates[0], coordinates[1]
            props = feature.get("properties") or {}

            country = str(props.get("country") or "").lower()
            countrycode = str(props.get("countrycode") or "").lower()

            # Deutschland bevorzugen.
            if countrycode and countrycode not in {"de", "deu"}:
                continue
            if country and "deutsch" not in country and country != "germany":
                # Nur verwerfen, wenn tatsächlich ein Land angegeben wurde.
                continue

            display_name = self._format_photon_address(props)

            osm_type = str(props.get("osm_value") or "").lower()
            if props.get("housenumber"):
                confidence = 0.90
            elif props.get("street"):
                confidence = 0.80
            elif props.get("city") or props.get("town"):
                confidence = 0.67
            else:
                confidence = max(0.50, 0.75 - idx * 0.05)

            if osm_type in {"house", "building"}:
                confidence = max(confidence, 0.88)

            out.append(
                {
                    "display_name": display_name,
                    "lat": float(lat),
                    "lon": float(lon),
                    "confidence": round(float(confidence), 3),
                    "provider": "Photon",
                    "raw": feature,
                }
            )

        return out

    @staticmethod
    def _format_photon_address(props: dict[str, Any]) -> str:
        street = props.get("street") or props.get("name")
        number = props.get("housenumber")
        postcode = props.get("postcode")
        city = (
            props.get("city")
            or props.get("town")
            or props.get("village")
            or props.get("municipality")
        )
        state = props.get("state")

        first = " ".join(
            str(x) for x in [street, number] if x
        ).strip()

        second = " ".join(
            str(x) for x in [postcode, city] if x
        ).strip()

        parts = [x for x in [first, second, state, "Deutschland"] if x]
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Nominatim
    # ------------------------------------------------------------------

    def _search_nominatim(self, address: str, limit: int) -> list[dict]:
        # Öffentlicher Nominatim-Dienst: höchstens ca. 1 Anfrage/Sekunde.
        elapsed = time.monotonic() - self.last_nominatim_request
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)

        response = self.session.get(
            f"{self.nominatim_base_url}/search",
            params={
                "q": address,
                "format": "jsonv2",
                "addressdetails": 1,
                "limit": max(1, min(int(limit), 10)),
                "countrycodes": "de",
                "email": self.contact_email,
            },
            timeout=self.timeout,
        )
        self.last_nominatim_request = time.monotonic()
        response.raise_for_status()

        out: list[dict] = []

        for item in response.json():
            importance = float(item.get("importance") or 0)

            out.append(
                {
                    "display_name": item.get("display_name", ""),
                    "lat": float(item["lat"]),
                    "lon": float(item["lon"]),
                    "confidence": round(
                        min(0.92, 0.45 + importance),
                        3,
                    ),
                    "provider": "Nominatim",
                    "raw": item,
                }
            )

        return out
