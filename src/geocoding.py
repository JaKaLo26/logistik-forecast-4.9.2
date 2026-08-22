from __future__ import annotations

import logging
import os
import re
import time
import requests


LOGGER = logging.getLogger(__name__)


class Geocoder:
    """Kostenloser Multi-Provider-Geocoder mit Photon und Nominatim."""

    def __init__(self, timeout: int = 20):
        self.timeout = max(1, int(timeout))
        self.photon_base_url = os.getenv(
            "PHOTON_BASE_URL", "https://photon.komoot.io"
        ).rstrip("/")
        self.nominatim_base_url = os.getenv(
            "NOMINATIM_BASE_URL", "https://nominatim.openstreetmap.org"
        ).rstrip("/")
        self.contact_email = os.getenv("GEOCODER_CONTACT_EMAIL", "").strip()
        self.session = requests.Session()

        user_agent = (
            "LogistikForecast/4.9.5 "
            "(https://github.com/JaKaLo26/logistik-forecast-4.9.2)"
        )
        if self.contact_email:
            user_agent += f" contact={self.contact_email}"

        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/json",
            "Accept-Language": "de,en;q=0.8",
        })
        self.last_nominatim_request = 0.0
        self.last_errors: list[str] = []

    def search(self, address: str, limit: int = 5) -> list[dict]:
        """Sucht bei beiden Providern und liefert kompatible Treffer zurück.

        Bei null Treffern werden wenige, kontrollierte Schreibvarianten versucht.
        Providerfehler werden protokolliert und in ``last_errors`` bereitgestellt.
        Wenn kein Provider erreichbar ist, wird der Fehler nicht mehr verschluckt.
        """
        address = self._clean_address(address)
        if not address:
            return []

        requested_limit = min(max(int(limit), 1), 10)
        self.last_errors = []
        all_hits: list[dict] = []
        successful_requests = 0

        for variant_index, query in enumerate(self._query_variants(address)):
            for provider_name, provider in (
                ("Photon", self._photon),
                ("Nominatim", self._nominatim),
            ):
                try:
                    hits = provider(query, requested_limit)
                    successful_requests += 1
                    for hit in hits:
                        hit["query"] = query
                    all_hits.extend(hits)
                except (
                    requests.exceptions.RequestException,
                    ValueError,
                    KeyError,
                    TypeError,
                ) as exc:
                    message = f"{provider_name}: {type(exc).__name__}: {exc}"
                    self.last_errors.append(message)
                    LOGGER.warning("Geocoder-Anfrage fehlgeschlagen: %s", message)

            # Varianten sind nur ein Fallback. Sobald die Originalabfrage Treffer
            # liefert, werden keine zusätzlichen externen Anfragen erzeugt.
            if variant_index == 0 and all_hits:
                break
            if all_hits:
                break

        if successful_requests == 0:
            details = " | ".join(self.last_errors) or "unbekannter Providerfehler"
            raise RuntimeError(f"Kein Geocoding-Provider erreichbar. {details}")

        return self._deduplicate_and_rank(all_hits, address, requested_limit)

    @staticmethod
    def _clean_address(address: str) -> str:
        value = str(address or "").strip()
        value = re.sub(r"\s+", " ", value)
        value = re.sub(r"\s*,\s*", ", ", value)
        return value.strip(" ,")

    @classmethod
    def _query_variants(cls, address: str) -> list[str]:
        variants = [address]

        # Häufige Schreibabweichung bei deutschen Straßennamen, z. B.
        # "Leutzebad" -> "Leuzebad". Nur als Fallback nach null Treffern.
        spelling = re.sub(r"tz", "z", address, flags=re.IGNORECASE)
        if spelling != address:
            variants.append(spelling)

        # Manche Provider erkennen Hausnummern mit Buchstaben zuverlässiger,
        # wenn zwischen Zahl und Suffix ein Leerzeichen steht.
        spaced_suffix = re.sub(r"(?<=\d)([a-zA-Z])\b", r" \1", address)
        if spaced_suffix != address:
            variants.append(spaced_suffix)

        # Letzter Fallback: Hausnummer entfernen, damit zumindest die richtige
        # Straße als manueller Vorschlag erscheinen kann.
        if "," in address:
            street, locality = address.split(",", 1)
            street_without_number = re.sub(
                r"\s+\d+[a-zA-Z]?(?:\s*[-/]\s*\d+[a-zA-Z]?)?\s*$",
                "",
                street,
            ).strip()
            if street_without_number and street_without_number != street.strip():
                variants.append(f"{street_without_number}, {locality.strip()}")

        result = []
        seen = set()
        for value in variants:
            cleaned = cls._clean_address(value)
            key = cleaned.casefold()
            if cleaned and key not in seen:
                seen.add(key)
                result.append(cleaned)
        return result[:4]

    def _photon(self, address: str, limit: int) -> list[dict]:
        response = self.session.get(
            f"{self.photon_base_url}/api/",
            params={"q": address, "limit": limit, "lang": "de"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        features = payload.get("features", [])
        if not isinstance(features, list):
            raise ValueError("Photon-Antwort enthält keine gültige Trefferliste")

        out = []
        for idx, feature in enumerate(features):
            coords = (feature.get("geometry") or {}).get("coordinates") or []
            if len(coords) < 2:
                continue

            # Photon/GeoJSON liefert [lon, lat]. Intern verwendet die App
            # durchgehend (lat, lon).
            lat, lon = self._coordinates(coords[1], coords[0])
            properties = feature.get("properties") or {}
            countrycode = str(properties.get("countrycode") or "").lower()
            if countrycode and countrycode not in {"de", "deu"}:
                continue

            street = properties.get("street") or properties.get("name")
            number = properties.get("housenumber")
            postcode = properties.get("postcode")
            city = (
                properties.get("city")
                or properties.get("town")
                or properties.get("village")
                or properties.get("municipality")
            )
            first = " ".join(str(value) for value in (street, number) if value)
            second = " ".join(str(value) for value in (postcode, city) if value)
            display = ", ".join(
                value for value in (first, second, "Deutschland") if value
            )
            if not display:
                continue

            if number:
                confidence = 0.90
            elif properties.get("street"):
                confidence = 0.80
            elif city:
                confidence = 0.67
            else:
                confidence = max(0.50, 0.75 - idx * 0.05)

            out.append({
                "display_name": display,
                "lat": lat,
                "lon": lon,
                "confidence": round(confidence, 3),
                "provider": "Photon",
                "raw": feature,
            })
        return out

    def _nominatim(self, address: str, limit: int) -> list[dict]:
        elapsed = time.monotonic() - self.last_nominatim_request
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)

        params = {
            "q": address,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": limit,
            "countrycodes": "de",
        }
        if self.contact_email:
            params["email"] = self.contact_email

        try:
            response = self.session.get(
                f"{self.nominatim_base_url}/search",
                params=params,
                timeout=self.timeout,
            )
        finally:
            self.last_nominatim_request = time.monotonic()

        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("Nominatim-Antwort enthält keine gültige Trefferliste")

        out = []
        for item in payload:
            lat, lon = self._coordinates(item["lat"], item["lon"])
            display = str(item.get("display_name") or "").strip()
            if not display:
                continue
            importance = float(item.get("importance") or 0)
            out.append({
                "display_name": display,
                "lat": lat,
                "lon": lon,
                "confidence": round(min(0.92, 0.45 + importance), 3),
                "provider": "Nominatim",
                "raw": item,
            })
        return out

    @staticmethod
    def _coordinates(lat_value, lon_value) -> tuple[float, float]:
        lat = float(lat_value)
        lon = float(lon_value)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Provider lieferte ungültige Koordinaten")
        return lat, lon

    @staticmethod
    def _deduplicate_and_rank(
        hits: list[dict], address: str, limit: int
    ) -> list[dict]:
        input_tokens = set(re.findall(r"\w+", address.casefold()))
        unique: dict[tuple, dict] = {}

        for hit in hits:
            display = str(hit.get("display_name") or "").strip()
            if not display:
                continue
            key = (round(float(hit["lat"]), 5), round(float(hit["lon"]), 5))
            display_tokens = set(re.findall(r"\w+", display.casefold()))
            overlap = len(input_tokens & display_tokens) / max(1, len(input_tokens))
            rank = float(hit.get("confidence") or 0) + (0.25 * overlap)
            candidate = dict(hit)
            candidate["_rank"] = rank

            previous = unique.get(key)
            if previous is None or rank > previous["_rank"]:
                unique[key] = candidate

        ranked = sorted(
            unique.values(),
            key=lambda item: (-item["_rank"], item["display_name"].casefold()),
        )
        for hit in ranked:
            hit.pop("_rank", None)
        return ranked[:limit]
