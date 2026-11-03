"""Deterministic offline fixtures from existing tour order CSVs."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

VERSION = '1.0'


def generate_training_data(orders, *, seed=42, tour_count=30):
    required = {'auftrag', 'kunde', 'strasse', 'plz', 'ort', 'paletten',
                'warengewicht_kg', 'service_min'}
    missing = required - set(orders.columns)
    if missing:
        raise ValueError('Fehlende Auftragsspalten: ' + ', '.join(sorted(missing)))
    if orders.empty or orders[list(required)].isna().any().any():
        raise ValueError('Die Auftrags-CSV enthält leere Pflichtfelder.')
    service_minutes = pd.to_numeric(orders.service_min, errors='raise')
    if not (np.isfinite(service_minutes) & (service_minutes >= 0)).all():
        raise ValueError('Servicezeiten müssen endlich und nicht negativ sein.')
    if not 5 <= tour_count <= 500:
        raise ValueError('Bitte zwischen 5 und 500 Touren wählen.')

    rng = np.random.default_rng(seed)
    rows = []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for tour in range(tour_count):
        planned = start + timedelta(days=tour, hours=int(rng.integers(5, 15)))
        simulated = planned
        snapshot = planned - timedelta(hours=1)
        stops = rng.permutation(len(orders))[:10]
        for segment, stop in enumerate(stops, start=1):
            distance = int(rng.integers(2000, 25001))
            # Offline reference assumes 45 km/h. It is not an OSRM response.
            baseline = distance / 12.5
            peak = planned.hour in (6, 7, 8, 14, 15, 16)
            duration = max(1, round(baseline * (1.25 if peak else 1.05)
                                    + rng.normal(0, 45)))
            arrival = simulated + timedelta(seconds=duration)
            rows.append(dict(
                tour_id=f'SYN-{tour:04}', segment_id=segment,
                prediction_created_at=snapshot.isoformat(),
                planned_departure_at=planned.isoformat(),
                actual_departure_at=simulated.isoformat(),
                actual_arrival_at=arrival.isoformat(),
                actual_recorded_at=arrival.isoformat(),
                osrm_baseline_s=baseline, distance_m=distance, data_origin='synthetic',
                source_order=str(orders.iloc[stop].auftrag),
            ))
            service = timedelta(minutes=float(service_minutes.iloc[stop]))
            planned += timedelta(seconds=baseline) + service
            simulated = arrival + service
    return pd.DataFrame(rows), {
        'version': VERSION, 'seed': seed, 'tour_count': tour_count,
        'max_stops_per_tour': 10, 'distance_m_range': [2000, 25000],
        'baseline_speed_kmh': 45, 'peak_utc_hours': [6, 7, 8, 14, 15, 16],
        'peak_multiplier': 1.25, 'off_peak_multiplier': 1.05, 'noise_std_s': 45,
    }
