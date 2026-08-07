from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TrafficResult:
    provider: str
    delay_s: float
    score: float
    confidence: float
    incidents: list[dict]
    debug: dict


class AutobahnProvider:
    """
    Adapter für die offizielle Autobahn-API.
    In 4.9.3 bleibt die räumliche A-Nummern-Zuordnung transparent als
    Entwicklungsstufe markiert; der Forecast erfindet keinen Zuschlag.
    """

    def __init__(self, base_url="https://verkehr.autobahn.de/o/autobahn", timeout=20):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def analyze_route(self, geometry, baseline_duration_s):
        return TrafficResult(
            "Autobahn API",
            0,
            0,
            0,
            [],
            {
                "status": "prepared",
                "note": (
                    "Nächster Ausbau: Autobahnnummern aus OSRM-Schritten erkennen "
                    "und Ereignisse räumlich auf die bereits optimierte Route matchen."
                ),
            },
        )


def combine(results: list[TrafficResult], weights: dict[str, float]) -> dict:
    active = [r for r in results if r.confidence > 0]
    if not active:
        return {
            "delay_s": 0,
            "score": 0,
            "confidence": 0,
            "provider_results": [r.__dict__ for r in results],
        }

    denom = sum(weights.get(r.provider, 1) for r in active)
    score = sum(r.score * weights.get(r.provider, 1) for r in active) / denom
    delay = sum(r.delay_s * weights.get(r.provider, 1) for r in active) / denom
    confidence = sum(r.confidence * weights.get(r.provider, 1) for r in active) / denom
    return {
        "delay_s": delay,
        "score": score,
        "confidence": confidence,
        "provider_results": [r.__dict__ for r in results],
    }
