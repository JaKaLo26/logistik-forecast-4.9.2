from __future__ import annotations
import math
import pandas as pd

PALLET_TARE_KG = 25

REQUIRED = ['auftrag','kunde','strasse','plz','ort','paletten','warengewicht_kg','service_min']

def normalize_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError('Fehlende CSV-Spalten: ' + ', '.join(missing))
    for c in ['paletten','warengewicht_kg','service_min']:
        df[c] = pd.to_numeric(df[c], errors='raise')
    df['paletten'] = df['paletten'].astype(int)
    df['palettengewicht_kg'] = df['paletten'] * PALLET_TARE_KG
    df['gesamtgewicht_kg'] = df['warengewicht_kg'] + df['palettengewicht_kg']
    df['adresse'] = df['strasse'].astype(str) + ', ' + df['plz'].astype(str) + ' ' + df['ort'].astype(str)
    return df

def summarize_orders(df: pd.DataFrame) -> dict:
    p = int(df['paletten'].sum())
    w = int(df['gesamtgewicht_kg'].sum())
    return {
        'auftraege': len(df), 'paletten': p, 'gewicht_kg': w,
        'durchschnitt_kg_pro_palette': math.floor(w / p) if p else 0,
    }

def distribute_orders(orders: pd.DataFrame, vehicles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    vehicles = vehicles.copy()
    vehicles = vehicles[vehicles['available'].astype(bool)].copy()
    vehicles['used_pallets'] = 0
    vehicles['used_weight_kg'] = 0
    orders = orders.copy().sort_values(['gesamtgewicht_kg','paletten'], ascending=False)
    assignments, warnings = [], []
    for _, order in orders.iterrows():
        candidates = []
        for idx, v in vehicles.iterrows():
            free_p = int(v['pallet_capacity'] - v['used_pallets'])
            free_w = int(v['payload_kg'] - v['used_weight_kg'])
            if order['paletten'] <= free_p and order['gesamtgewicht_kg'] <= free_w:
                projected_p = (v['used_pallets'] + order['paletten']) / v['pallet_capacity']
                projected_w = (v['used_weight_kg'] + order['gesamtgewicht_kg']) / v['payload_kg']
                candidates.append((max(projected_p, projected_w), idx))
        if not candidates:
            warnings.append(f"Auftrag {order['auftrag']} ({int(order['paletten'])} Pal., {int(order['gesamtgewicht_kg'])} kg) nicht zuweisbar")
            assignments.append({**order.to_dict(), 'vehicle_id': 'NICHT ZUGEWIESEN'})
            continue
        _, best_idx = max(candidates, key=lambda x: x[0])
        vehicles.loc[best_idx, 'used_pallets'] += int(order['paletten'])
        vehicles.loc[best_idx, 'used_weight_kg'] += int(order['gesamtgewicht_kg'])
        assignments.append({**order.to_dict(), 'vehicle_id': vehicles.loc[best_idx, 'vehicle_id']})
    vehicles['pallet_utilization_pct'] = (vehicles['used_pallets'] / vehicles['pallet_capacity'] * 100).round(1)
    vehicles['weight_utilization_pct'] = (vehicles['used_weight_kg'] / vehicles['payload_kg'] * 100).round(1)
    return pd.DataFrame(assignments), vehicles, warnings
