from __future__ import annotations

def forecast_summary(distance_m, baseline_s, traffic_s, service_min):
    total_s=baseline_s+traffic_s+service_min*60
    return {
        'distanz_km':round(distance_m/1000,1),
        'basis_fahrzeit_min':round(baseline_s/60),
        'live_zuschlag_min':round(traffic_s/60),
        'servicezeit_min':round(service_min),
        'gesamtzeit_min':round(total_s/60),
    }
