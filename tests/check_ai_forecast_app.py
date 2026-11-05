"""Offline app check with bundled model and mocked road responses."""
import sys, json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app
from src.builtin_model import load_builtin_model, predict_road_matrix
from src.capacity import normalize_orders
import pandas as pd
bundle=load_builtin_model()
assert not any('Training & Vergleich' in str(c) for c in app.demo.config['components'])
frame=normalize_orders(pd.DataFrame([dict(auftrag='A',kunde='K',strasse='S 1',plz='00000',ort='Demo',paletten=2,warengewicht_kg=700,service_min=5,lat=48.79,lon=9.18)]))
depot=json.dumps(dict(lat=48.8,lon=9.2))
with patch.object(app,'road_matrix',return_value=([[0,10],[10,0]],[[0,5000],[5000,0]])):
    driver={"driver_id":"TEST-FAHRER",**app.DRIVER_DEFAULTS,"remaining_drive_min":80,"drive_since_break_min":260}
    clustered=app.create_clusters(frame.to_json(orient='records'),depot,'05.11.26','08:00',bundle,pd.DataFrame([driver]))
assert clustered[5]
assert clustered[5][0]["driver_id"]=="TEST-FAHRER"
assert clustered[5][0]["remaining_drive_min"]==80
assert clustered[1].iloc[0].pausen_min==45
route_data=dict(legs=[dict(duration=600,distance=5000)]*2,distance_m=10000,duration_s=1200,geometry=[(48.8,9.2),(48.79,9.18),(48.8,9.2)])
with patch.object(app.OSRMRouter,'route',return_value=route_data):
    planned=app.optimize_routes(clustered[2],clustered[5],depot,'05.11.26','08:00',bundle)
with patch.object(app,'TOMTOM_API_KEY',''),patch('src.forecast.calculate_model_forecast_s',side_effect=AssertionError('heuristic')),patch.object(app,'log_forecast',return_value={}),patch.object(app,'training_logger_status',return_value={}):
    out=app.calculate_all_forecasts(planned[3],depot,clustered[5],'05.11.26','08:00',bundle)
assert out[0]
assert json.loads(planned[3])[0]['stops'][0]['warengewicht_kg']==700
print('BUILTIN_MODEL_PAGE_2_PAGE_3_OK')
