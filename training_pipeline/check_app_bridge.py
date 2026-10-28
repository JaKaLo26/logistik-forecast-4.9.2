"""Exercise actual UI callbacks; no browser rendering and no remote logging."""
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import app
from src import training_logger
root=Path('training_pipeline/integration_result')
orders,_,_=app.read_csv(str(root/'forecast_orders.csv'))
bundle,status=app.activate_pipeline_model(str(root/'model/model.joblib'))
plans=json.loads((root/'forecast_plan.json').read_text());count=0
with tempfile.TemporaryDirectory() as d,patch.object(training_logger,'LOCAL_DATA_DIR',Path(d)),patch.object(training_logger,'HF_TOKEN',''),patch.object(app,'TOMTOM_API_KEY',''),patch.dict('os.environ',{'TOMTOM_API_KEY':''}),patch('src.forecast.calculate_model_forecast_s',side_effect=AssertionError('Heuristic used')),patch('requests.sessions.Session.request',side_effect=AssertionError('Network used')):
    for p in plans:
        stops=[]
        for s in p['stops']:
            row=orders[orders.auftrag==s['auftrag']].iloc[0].to_dict();row.update(s);stops.append(row)
        route=dict(vehicle_id=p['tour_id'],stops=stops,legs=p['legs'])
        result=app.calculate_all_forecasts(json.dumps([route]),json.dumps(p['depot']),pd.DataFrame(columns=['vehicle_id']),p['start_time'],bundle)
        forecasts=json.loads(result[3]);assert len(forecasts)==1
        for segment in next(iter(forecasts.values()))['segments']:
            assert segment['model_id']==bundle['report']['model_id'];count+=1
report=dict(csv_import_callback='passed',model_load_callback='passed',forecast_callback='passed',segments=count,model_id=bundle['report']['model_id'],browser_visual_test=False)
(root/'app_callback_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
