"""Import the existing order CSV and run the application's actual tour forecast."""
import argparse,json,math
from pathlib import Path
import pandas as pd
from src.capacity import normalize_orders
from src.ml_training import load_forecast_model
from src.forecast import calculate_tour_forecast


def run(model_path, orders_path, plan_path, output):
    bundle=load_forecast_model(model_path)
    orders=pd.read_csv(orders_path,dtype=str,sep=None,engine='python')
    # Explicit input allowlist: outputs/labels from an imported file cannot become features.
    required=['auftrag','kunde','strasse','plz','ort','paletten','warengewicht_kg','service_min']
    if not set(required)<=set(orders): raise ValueError('Unvollständige Auftrags-CSV')
    orders=orders[required].copy()
    if orders.isna().any().any() or orders.auftrag.duplicated().any():raise ValueError('Fehlende Werte/doppelte Auftrags-IDs')
    for name in ['paletten','warengewicht_kg','service_min']:
        values=pd.to_numeric(orders[name],errors='raise')
        if not values.map(math.isfinite).all() or (values<0).any():raise ValueError('Ungültige Auftragszahlen')
        if name=='paletten' and ((values%1!=0)|(values<=0)).any():raise ValueError('Ungültige Palettenzahl')
    orders=normalize_orders(orders).set_index('auftrag')
    plans=json.loads(Path(plan_path).read_text())
    assigned=[s['auftrag'] for p in plans for s in p['stops']]
    if len(assigned)!=len(set(assigned)) or set(assigned)!=set(orders.index):raise ValueError('Tourplan und CSV-Aufträge stimmen nicht überein')
    results=[]
    for p in plans:
        stops=[]
        for s in p['stops']:
            stop=orders.loc[s['auftrag']].to_dict()
            stop.update(auftrag=s['auftrag'],lat=float(s['lat']),lon=float(s['lon']))
            stops.append(stop)
        legs=p['legs']
        if len(legs)!=len(stops)+1:raise ValueError('Rückfahrt/Segmentdaten fehlen')
        for leg in legs:
            if not all(math.isfinite(float(leg[k])) and float(leg[k])>0 for k in ['duration','distance']):raise ValueError('Ungültige Routenwerte')
        result=calculate_tour_forecast({'vehicle_id':p['tour_id'],'stops':stops,'legs':legs},
            p['depot'],p['start_time'],ml_bundle=bundle,tour_id=p['tour_id'])
        if any(s['model_id']!=bundle['report']['model_id'] for s in result['segments']):raise AssertionError('Forecast did not use loaded model')
        results.append(result)
    Path(output).write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False))
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Gespeichertes Pipeline-Modell → CSV-Aufträge → Forecast')
    for key in ['model','orders','plan','out']:p.add_argument('--'+key,required=True)
    a=p.parse_args();run(a.model,a.orders,a.plan,a.out)
