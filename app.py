from __future__ import annotations
from __future__ import annotations

import json
import os
import spaces

from pathlib import Path

import pandas as pd
import gradio as gr

from dotenv import load_dotenv

from src.capacity import normalize_orders, summarize_orders, distribute_orders
from src.geocoding import Geocoder
from src.models import Vehicle
from src.routing import OSRMRouter
from src.traffic import HereTrafficProvider, AutobahnProvider, combine
from src.maps import build_map
from src.forecast import forecast_summary

load_dotenv()
TIMEOUT=int(os.getenv('REQUEST_TIMEOUT_SECONDS','20'))
DEFAULT_VEHICLES=[
    Vehicle(f'LKW-K0{i}','14 t',18,6000,c) for i,c in enumerate(['#2563eb','#7c3aed','#0891b2'],1)
]+[
    Vehicle(f'LKW-G0{i}','40 t',33,24000,c) for i,c in enumerate(['#dc2626','#ea580c','#16a34a'],1)
]

def vehicles_df(): return pd.DataFrame([v.to_dict() for v in DEFAULT_VEHICLES])

def read_csv(file):
    if not file: raise gr.Error('Bitte CSV auswählen.')
    df=normalize_orders(pd.read_csv(file))
    s=summarize_orders(df)
    return df, f"**{s['auftraege']} Aufträge · {s['paletten']} Paletten · {s['gewicht_kg']:,} kg · Ø {s['durchschnitt_kg_pro_palette']} kg/Palette**", df.to_json(orient='records', force_ascii=False)

def geocode_orders(orders_json):
    df=pd.read_json(orders_json, orient='records')
    geocoder=Geocoder(os.getenv('NOMINATIM_BASE_URL','https://nominatim.openstreetmap.org'),TIMEOUT)
    rows=[]; candidates={}
    for _,r in df.iterrows():
        try: hits=geocoder.search(r['adresse'])
        except Exception as e: hits=[]
        best=hits[0] if hits else None
        uncertain=not best or best['confidence']<0.72
        rows.append({'auftrag':r['auftrag'],'eingabe':r['adresse'],'treffer':best['display_name'] if best else '',
                     'confidence':best['confidence'] if best else 0,'status':'MANUELL PRÜFEN' if uncertain else 'OK',
                     'lat':best['lat'] if best else None,'lon':best['lon'] if best else None})
        candidates[str(r['auftrag'])]=hits
    return pd.DataFrame(rows), json.dumps(candidates,ensure_ascii=False), 'Unsichere Treffer können direkt in der Tabelle überschrieben werden.'

def save_addresses(address_table, orders_json):
    orders=pd.read_json(orders_json,orient='records')
    adr=pd.DataFrame(address_table)
    merged=orders.merge(adr[['auftrag','treffer','lat','lon']],on='auftrag',how='left')
    if merged[['lat','lon']].isna().any().any(): raise gr.Error('Mindestens eine Adresse hat keine Koordinaten.')
    return merged.to_json(orient='records',force_ascii=False), 'Adressen übernommen.'

def distribute(orders_geo_json, vehicle_table):
    orders=pd.read_json(orders_geo_json,orient='records'); vehicles=pd.DataFrame(vehicle_table)
    ass, util, warnings=distribute_orders(orders,vehicles)
    msg='✅ Alle Aufträge zugewiesen.' if not warnings else '⚠️ ' + ' | '.join(warnings)
    return ass, util, ass.to_json(orient='records',force_ascii=False), msg

def calculate(assign_json, vehicle_table, here_key):
    ass=pd.read_json(assign_json,orient='records'); vehicles=pd.DataFrame(vehicle_table)
    router=OSRMRouter(os.getenv('OSRM_BASE_URL','https://router.project-osrm.org'),TIMEOUT)
    routes=[]; summaries=[]; debug=[]
    for vid,group in ass[ass.vehicle_id!='NICHT ZUGEWIESEN'].groupby('vehicle_id'):
        v=vehicles[vehicles.vehicle_id==vid].iloc[0]
        # Demo-Depot: erster Stopp auch Start/Ende. Später als eigenes Depot-Feld ausbaubar.
        coords=[(float(x.lat),float(x.lon)) for _,x in group.iterrows()]
        if len(coords)==1: coords=coords+coords
        route=router.route(coords)
        here=HereTrafficProvider(here_key or os.getenv('HERE_API_KEY',''),TIMEOUT).analyze_route(route['geometry'],route['duration_s'])
        autobahn=AutobahnProvider(timeout=TIMEOUT).analyze_route(route['geometry'],route['duration_s'])
        traffic=combine([here,autobahn],{'HERE':0.7,'Autobahn API':0.3})
        service=float(group.service_min.sum())
        summary=forecast_summary(route['distance_m'],route['duration_s'],traffic['delay_s'],service)
        summary.update({'vehicle_id':vid,'paletten':int(group.paletten.sum()),'gewicht_kg':int(group.gesamtgewicht_kg.sum()),
                        'traffic_score':round(traffic['score'],1),'datenvertrauen_pct':round(traffic['confidence']*100)})
        summaries.append(summary)
        stops=[{'lat':float(r.lat),'lon':float(r.lon),'kunde':r.kunde,'paletten':int(r.paletten),'gesamtgewicht_kg':int(r.gesamtgewicht_kg)} for _,r in group.iterrows()]
        routes.append({'vehicle_id':vid,'color':v['color'],'geometry':route['geometry'],'stops':stops})
        debug.append({'vehicle_id':vid,'traffic':traffic,'osrm_summary':{'distance_m':route['distance_m'],'duration_s':route['duration_s']}})
    return build_map(routes), pd.DataFrame(summaries), json.dumps(debug,ensure_ascii=False,indent=2,default=str)

CSS='''
.step-title{font-size:1.35rem;font-weight:700;margin-bottom:.35rem}.muted{color:#6b7280}
'''
with gr.Blocks(title='Logistik Forecast 4.9.2',css=CSS) as demo:
    gr.Markdown('# Logistik Forecast 4.9.2\nMehrstufige Python-/Gradio-Demo')
    orders_state=gr.State('[]'); geo_state=gr.State('[]'); assignments_state=gr.State('[]'); candidates_state=gr.State('{}')
    with gr.Tabs():
        with gr.Tab('1 · CSV & Adressen'):
            gr.Markdown('<div class="step-title">Aufträge importieren und Adressen prüfen</div>')
            upload=gr.File(label='CSV-Datei',file_types=['.csv'],type='filepath')
            import_btn=gr.Button('CSV importieren',variant='primary')
            order_summary=gr.Markdown(); orders_table=gr.Dataframe(interactive=True,label='Aufträge')
            geocode_btn=gr.Button('Adressen automatisch prüfen')
            address_table=gr.Dataframe(headers=['auftrag','eingabe','treffer','confidence','status','lat','lon'],interactive=True,label='Adressprüfung')
            address_note=gr.Markdown(); save_address_btn=gr.Button('Geprüfte Adressen übernehmen',variant='primary'); address_saved=gr.Markdown()
        with gr.Tab('2 · Flotte & Kapazität'):
            gr.Markdown('<div class="step-title">Fahrzeuge skalieren und Kapazität verteilen</div>')
            gr.Markdown('Zeilen hinzufügen, löschen oder Werte ändern. Palettenplätze und Nutzlast werden parallel geprüft.')
            vehicle_table=gr.Dataframe(value=vehicles_df(),interactive=True,label='Verfügbare Fahrzeuge')
            distribute_btn=gr.Button('Kapazität automatisch verteilen',variant='primary')
            allocation_status=gr.Markdown(); assignments_table=gr.Dataframe(label='Auftragszuweisung')
            utilization_table=gr.Dataframe(label='Fahrzeugauslastung')
        with gr.Tab('3 · Forecast & Verkehr'):
            gr.Markdown('<div class="step-title">Routen, Live-Zuschlag und API-Kontrolle</div>')
            here_key=gr.Textbox(label='HERE API-Key (optional)',type='password')
            forecast_btn=gr.Button('Forecast berechnen',variant='primary')
            with gr.Row():
                with gr.Column(scale=3): map_html=gr.HTML(label='Karte')
                with gr.Column(scale=1): forecast_table=gr.Dataframe(label='Forecast je LKW')
            gr.Markdown('### API-Debug – Anfrageauswertung und Feldzuordnung')
            debug_json=gr.Code(language='json',label='Traffic-/Routing-Debug')
    import_btn.click(read_csv,upload,[orders_table,order_summary,orders_state])
    geocode_btn.click(geocode_orders,orders_state,[address_table,candidates_state,address_note])
    save_address_btn.click(save_addresses,[address_table,orders_state],[geo_state,address_saved])
    distribute_btn.click(distribute,[geo_state,vehicle_table],[assignments_table,utilization_table,assignments_state,allocation_status])
    forecast_btn.click(calculate,[assignments_state,vehicle_table,here_key],[map_html,forecast_table,debug_json])

if __name__=='__main__':
    demo.launch(server_name='0.0.0.0',server_port=int(os.getenv('PORT','7860')))
