"""Deterministic offline fixtures from existing tour order CSVs."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

VERSION = '2.0'


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
    if not 5 <= tour_count <= 5000:
        raise ValueError('Bitte zwischen 5 und 5000 Touren wählen.')

    balanced_states = 'bundesland' in orders.columns
    if balanced_states:
        states = sorted(orders.bundesland.dropna().unique())
        if len(states) != 16 or orders.bundesland.isna().any():
            raise ValueError('Für Deutschlandtraining sind genau 16 Bundesländer erforderlich.')
        by_state = {state: orders[orders.bundesland == state] for state in states}
        if any(len(items) < 10 for items in by_state.values()):
            raise ValueError('Mindestens zehn verschiedene Teststopps je Bundesland erforderlich.')
        # Round-robin spreads each state over the entire timeline. Distribute
        # 8 * tour_count stops as evenly as possible across all 16 states.
        state_plan = [states[i % 16] for i in range(tour_count)]
        counts = {state: state_plan.count(state) for state in states}
        target = {state: 8 * tour_count // 16 + int(i < (8 * tour_count % 16))
                  for i, state in enumerate(states)}
        stop_plan = []
        assigned = {state: 0 for state in states}
        for state in state_plan:
            n = counts[state]
            index = assigned[state]
            remaining = target[state] - 8 * n
            extra = (1 if remaining > 0 and index < remaining else
                     -1 if remaining < 0 and index < -remaining else 0)
            stop_plan.append(8 + extra)
            assigned[state] += 1
        if any(n < 1 or n > 10 for n in stop_plan):
            raise ValueError('Tourenzahl erlaubt keine ausgeglichene Stoppverteilung.')

    rng = np.random.default_rng(seed)
    rows = []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for tour in range(tour_count):
        planned = start + timedelta(days=tour // 4 if balanced_states else tour,
                                    hours=int(rng.integers(5, 15)))
        simulated = planned
        snapshot = planned - timedelta(hours=1)
        selected = by_state[state_plan[tour]] if balanced_states else orders
        stops = rng.permutation(len(selected))[:stop_plan[tour] if balanced_states else 10]
        for segment, stop in enumerate(stops, start=1):
            order = selected.iloc[stop]
            distance = int(rng.integers(2000, 25001))
            # Offline reference assumes 45 km/h. It is not an OSRM response.
            baseline = distance / 12.5
            local_hour = pd.Timestamp(planned).tz_convert('Europe/Berlin').hour
            peak = local_hour in (6, 7, 8, 14, 15, 16)
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
                source_order=str(order.auftrag),
                bundesland=str(order.bundesland) if balanced_states else '',
                address_status='ungeprüfte Straßenangabe ohne Hausnummer' if balanced_states else '',
            ))
            service = timedelta(minutes=float(order.service_min))
            planned += timedelta(seconds=baseline) + service
            simulated = arrival + service
    return pd.DataFrame(rows), {
        'version': VERSION, 'seed': seed, 'tour_count': tour_count,
        'max_stops_per_tour': 10, 'distance_m_range': [2000, 25000],
        'baseline_speed_kmh': 45, 'peak_local_hours': [6, 7, 8, 14, 15, 16],
        'peak_multiplier': 1.25, 'off_peak_multiplier': 1.05, 'noise_std_s': 45,
        'regional_distribution': '16 states, 500 stops each' if balanced_states and tour_count == 1000 else None,
        'routing_source': 'none; independent random distances, no geocoding or OSRM',
    }
