import sys, tempfile, json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path.cwd()))
sys.path.insert(0,str(Path.cwd()/'tests'))
import app
from test_ihk import dataset
print('APP_BUILD_OK')
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder)/'train.csv';dataset().to_csv(path,index=False)
    result=app.run_training(str(path),42,10)
    assert len(result)==9
    bundle=app.activate_model(result[5])[0]
    route={'vehicle_id':'V','stops':[dict(auftrag='A',lat=48.79,lon=9.18,service_min=5,paletten=2,warengewicht_kg=700)],'legs':[{'duration':600,'distance':5000}]*2}
    vehicles=[dict(vehicle_id='V')]
    with patch.object(app,'TOMTOM_API_KEY',''),patch('src.forecast.calculate_model_forecast_s',side_effect=AssertionError('heuristic')),patch.object(app,'log_forecast',return_value={}),patch.object(app,'training_logger_status',return_value={}):
        out=app.calculate_all_forecasts(json.dumps([route]),json.dumps(dict(lat=48.8,lon=9.2)),vehicles,'05.11.26','08:00',bundle)
        assert 'V' in out[0]
    try: app.calculate_all_forecasts(json.dumps([route]),'{"lat":48,"lon":9}',vehicles,'05.11.26','08:00')
    except Exception as exc: assert 'KI-Modell' in str(exc)
    else: raise AssertionError('Missing model accepted')
print('TRAIN_EXPORT_LOAD_APP_FORECAST_OK')
