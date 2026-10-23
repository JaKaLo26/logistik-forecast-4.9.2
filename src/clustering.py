"""Street-time clustering and conservative single-day planning, not legal certification.
Normal 9h driving / 11h daily rest policy; no reduced rest or 10h extensions.
Driver budgets must be supplied from actual records. Breaks are time allowances,
not verified parking locations. Source: EC 561/2006 arts. 6–8, Directive 2002/15.
"""
from __future__ import annotations
import math
import os
import pandas as pd

DRIVER_DEFAULTS = {
    'remaining_drive_min': 540, 'remaining_work_min': 480,
    'remaining_duty_min': 780, 'drive_since_break_min': 0,
    'work_since_break_min': 0, 'remaining_week_drive_min': 3360,
    'remaining_fortnight_drive_min': 5400,
    'minutes_until_weekly_rest': 780,
}


def number(value, label):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{label}: endliche, nicht negative Zahl erforderlich.')
    return value


def plan_day(legs_min, service_min, vehicle):
    """Account for driving and other work separately; use full 45m breaks.
    Conservative: a break at 4.5h driving precedes any further work.
    Daily rest is protected by a caller-supplied deadline, never reset here.
    """
    budget = {k: number(vehicle.get(k, v), k) for k, v in DRIVER_DEFAULTS.items()}
    if budget['drive_since_break_min'] > 270 or budget['work_since_break_min'] > 360:
        raise ValueError('Fahrer-Vorzeiten überschreiten das Planungsprofil; erst klären.')
    drive = work = elapsed = pauses = 0.0
    ds = budget['drive_since_break_min']
    ws = budget['work_since_break_min']
    events = []
    for index, leg in enumerate(legs_min):
        for kind, amount in [('drive', leg), ('service', service_min[index] if index < len(service_min) else 0)]:
            remaining = number(amount, kind)
            while remaining > 1e-8:
                if ds >= 270 - 1e-8 or ws >= 360 - 1e-8:
                    events.append({'after_elapsed_min': round(elapsed, 2), 'duration_min': 45,
                                   'leg_index': index, 'during': kind, 'parking_verified': False})
                    pauses += 45
                    elapsed += 45
                    ds = ws = 0.0
                chunk = min(remaining, 360-ws, 270-ds if kind == 'drive' else remaining)
                elapsed += chunk
                work += chunk
                ws += chunk
                if kind == 'drive':
                    drive += chunk
                    ds += chunk
                remaining -= chunk
    limits = [min(540, budget['remaining_drive_min'], budget['remaining_week_drive_min'],
                  budget['remaining_fortnight_drive_min']) - drive,
              budget['remaining_work_min'] - work,
              min(780, budget['remaining_duty_min'], budget['minutes_until_weekly_rest']) - elapsed]
    return {'feasible': min(limits) >= -1e-8, 'fahrzeit_min': round(drive, 2),
            'service_min': round(work-drive, 2), 'pausen_min': pauses,
            'tourdauer_min': round(elapsed, 2), 'break_events': events}


def road_matrix(coordinates):
    """Directed OSRM durations in batches, without a straight-line fallback."""
    from src.routing import OSRMRouter
    router = OSRMRouter(os.getenv('OSRM_BASE_URL', 'https://router.project-osrm.org'),
                        int(os.getenv('REQUEST_TIMEOUT_SECONDS', '20')))
    count = len(coordinates)
    matrix = [[math.inf] * count for _ in coordinates]
    for a in range(0, count, 40):
        sources = list(range(a, min(a+40, count)))
        for b in range(0, count, 40):
            destinations = list(range(b, min(b+40, count)))
            indices = list(dict.fromkeys(sources + destinations))
            local = {original: i for i, original in enumerate(indices)}
            url = router.base_url + '/table/v1/driving/' + router._coord_string([coordinates[i] for i in indices])
            response = router.session.get(url, params={
                'sources': ';'.join(str(local[i]) for i in sources),
                'destinations': ';'.join(str(local[i]) for i in destinations),
                'annotations': 'duration'}, timeout=router.timeout)
            response.raise_for_status()
            payload = response.json()
            if payload.get('code') != 'Ok':
                raise ValueError('Straßenmatrix nicht verfügbar; keine Luftlinien-Ersatzwerte.')
            values = payload['durations']
            if len(values) != len(sources) or any(len(row) != len(destinations) for row in values):
                raise ValueError('Unvollständige Straßenmatrix.')
            for i, source in enumerate(sources):
                for j, target in enumerate(destinations):
                    value = values[i][j]
                    matrix[source][target] = math.inf if value is None else number(value, 'Straßenfahrzeit') / 60
    return matrix


def cluster_orders(orders, vehicles, depot, durations=None):
    columns = list(orders.columns) + ['cluster_id', 'vehicle_id', 'cluster_sequence']
    empty = pd.DataFrame(columns=list(dict.fromkeys(columns)))
    if orders.empty:
        return empty, pd.DataFrame(), ['Keine Aufträge vorhanden.']
    rows = orders.to_dict('records')
    for row in rows:
        for key in ('paletten', 'gesamtgewicht_kg', 'service_min'):
            number(row[key], key)
        if float(row['paletten']) != int(row['paletten']):
            raise ValueError('Palettenzahl muss ganzzahlig sein.')
    coords = [depot] + [(float(r['lat']), float(r['lon'])) for r in rows]
    if any(not math.isfinite(lat) or not math.isfinite(lon) or abs(lat)>90 or abs(lon)>180 for lat, lon in coords):
        raise ValueError('Ungültige Koordinaten.')
    fleet = vehicles.to_dict('records')
    fleet = [v for v in fleet if str(v.get('available', True)).lower() in ('true', '1', '1.0')]
    fleet.sort(key=lambda v: (number(v['pallet_capacity'], 'Kapazität'), number(v['payload_kg'], 'Nutzlast')), reverse=True)
    matrix = durations if durations is not None else road_matrix(coords) if fleet else None
    open_ids = list(range(1, len(rows)+1))
    assigned, summaries = [], []
    warnings = ['Planung: OSRM-Straßenfahrzeiten, kein LKW-Profil und keine Verkehrsgarantie. '
                'Fahrerbudgets sind Vorgaben, keine geprüfte Fahrerhistorie. '
                'Zeitfenster und reale Pausenplätze müssen separat geprüft werden.']
    def assess(seq, v):
        path = [0] + seq + [0]
        legs = [matrix[x][y] for x, y in zip(path, path[1:])]
        if any(not math.isfinite(t) for t in legs):
            return None
        report = plan_day(legs, [rows[i-1]['service_min'] for i in seq], v)
        return report if report['feasible'] else None
    for vehicle in fleet:
        sequence = []
        used_p = used_w = 0
        report = None
        while open_ids:
            candidates = []
            for item in open_ids:
                row = rows[item-1]
                if used_p + row['paletten'] > float(vehicle['pallet_capacity']) or used_w + row['gesamtgewicht_kg'] > float(vehicle['payload_kg']):
                    continue
                if not sequence:
                    trial = assess([item], vehicle)
                    if trial:
                        candidates.append((-matrix[0][item], item, [item], trial))
                else:
                    # Cheapest feasible insertion using directed road times.
                    for position in range(len(sequence)+1):
                        seq = sequence[:position] + [item] + sequence[position:]
                        trial = assess(seq, vehicle)
                        if trial:
                            candidates.append((trial['tourdauer_min'] - report['tourdauer_min'], item, seq, trial))
            if not candidates:
                break
            _, item, sequence, report = min(candidates, key=lambda c: (c[0], c[1]))
            used_p += rows[item-1]['paletten']
            used_w += rows[item-1]['gesamtgewicht_kg']
            open_ids.remove(item)
        if not sequence:
            continue
        cid = f'REGION-{len(summaries)+1:02d}'
        for position, item in enumerate(sequence, 1):
            assigned.append({**rows[item-1], 'cluster_id': cid, 'vehicle_id': vehicle['vehicle_id'], 'cluster_sequence': position})
        summaries.append({'cluster_id': cid, 'vehicle_id': vehicle['vehicle_id'], 'stopps': len(sequence),
                          'paletten': used_p, 'gewicht_kg': used_w,
                          'paletten_auslastung_pct': round(100*used_p/float(vehicle['pallet_capacity']), 1),
                          'gewicht_auslastung_pct': round(100*used_w/float(vehicle['payload_kg']), 1) if float(vehicle['payload_kg']) else 0,
                          **{k:v for k,v in report.items() if k not in ('break_events', 'feasible')}})
    for item in open_ids:
        assigned.append({**rows[item-1], 'cluster_id': 'NICHT ZUGEWIESEN', 'vehicle_id': 'NICHT ZUGEWIESEN', 'cluster_sequence': 0})
    if open_ids:
        warnings.append(f'{len(open_ids)} Aufträge offen: keine zulässige Einfügung bei Straßen-, Kapazitäts- und Zeitprüfung gefunden. Kein Nachweis globaler Unlösbarkeit.')
    return pd.DataFrame(assigned, columns=empty.columns), pd.DataFrame(summaries), warnings
