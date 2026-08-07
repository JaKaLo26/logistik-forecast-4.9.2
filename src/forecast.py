from __future__ import annotations


def forecast_summary(distance_m, baseline_s, traffic_s, service_min):
    distance_m = float(distance_m or 0)
    baseline_s = float(baseline_s or 0)
    traffic_s = float(traffic_s or 0)
    service_min = float(service_min or 0)

    total_s = baseline_s + traffic_s + service_min * 60

    return {
        "distanz_km": round(distance_m / 1000, 1),
        "basis_fahrzeit_min": round(baseline_s / 60),
        "live_zuschlag_min": round(traffic_s / 60),
        "servicezeit_min": round(service_min),
        "gesamtzeit_min": round(total_s / 60),
    }