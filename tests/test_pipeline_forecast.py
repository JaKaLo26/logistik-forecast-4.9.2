import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
from datetime import datetime
import joblib
from src.ml_training import load_forecast_model,predict_seconds
from src.forecast import calculate_tour_forecast,recalculate_tour_from_stop

ROOT=Path(__file__).resolve().parents[1]
class ForecastBridgeTests(unittest.TestCase):
    def setUp(self):
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from src.ml_training import PIPELINE_FEATURES
        frame = pd.DataFrame([[600,5000,8,2,750],[600,5000,9,2,0],[900,7000,10,3,1500]], columns=PIPELINE_FEATURES)
        model = RandomForestRegressor(n_estimators=5, random_state=42).fit(frame, [800,650,1100])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.joblib'
            joblib.dump(dict(schema_version='1.0.0',features=PIPELINE_FEATURES,data_origin='synthetic',model=model),path)
            self.bundle=load_forecast_model(path)
        self.route={'vehicle_id':'test','stops':[dict(auftrag='A',lat=48.79,lon=9.18,service_min=5,paletten=2,warengewicht_kg=700)],'legs':[{'duration':600,'distance':5000}]*2}
        self.depot={'lat':48.8,'lon':9.2}
    def test_negative_delivery_is_rejected(self):
        from src.forecast import remaining_delivery_load
        with self.assertRaises(ValueError):
            remaining_delivery_load([dict(warengewicht_kg=-10,paletten=2)])
        self.assertEqual(remaining_delivery_load([]),0)

    def test_application_model_roundtrip(self):
        from test_ihk import dataset
        from src.ml_training import train_experiment
        bundle, _, _ = train_experiment(dataset())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'app.joblib'
            joblib.dump(bundle,path)
            restored=load_forecast_model(path)
        self.assertEqual(predict_seconds(bundle,500,5000,'2026-01-10T08:00:00+01:00'),
                         predict_seconds(restored,500,5000,'2026-01-10T08:00:00+01:00'))
        report=restored['report']
        groups=[set(report[key]) for key in ('train_tours','validation_tours','test_tours')]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])

    def test_schema_rejected(self):
        bad=dict(self.bundle);bad['features']=['synthetic_target_s']
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.joblib';joblib.dump(bad,path)
            with self.assertRaises(ValueError):load_forecast_model(path)
    def test_missing_load_rejected(self):
        with self.assertRaises(ValueError):predict_seconds(self.bundle,600,5000,'2026-10-28T08:00:00+01:00')
    def test_bad_route_does_not_fall_back(self):
        self.route['legs'][0]['distance']=0
        with patch.dict('os.environ',{'TOMTOM_API_KEY':''}),patch('src.forecast.calculate_model_forecast_s',side_effect=AssertionError('fallback')):
            with self.assertRaises(ValueError):calculate_tour_forecast(self.route,self.depot,'2026-10-28T08:00:00+01:00',ml_bundle=self.bundle)
    def test_reforecast_uses_loaded_model_and_zero_return_load(self):
        with patch.dict('os.environ',{'TOMTOM_API_KEY':''}),patch('src.forecast.calculate_model_forecast_s',side_effect=AssertionError('fallback')):
            initial=calculate_tour_forecast(self.route,self.depot,'2026-10-28T08:00:00+01:00',ml_bundle=self.bundle)
            updated=recalculate_tour_from_stop(initial,self.route,self.depot,'A',ml_bundle=self.bundle,actual_service_s=480)
        seg=updated['segments'][-1]
        self.assertEqual(seg['model_id'],self.bundle['report']['model_id'])
        self.assertEqual(seg['model_forecast_s'],predict_seconds(self.bundle,600,5000,seg['departure_time'],0))
        self.assertEqual(updated['segments'][0]['prediction_created_at'],initial['segments'][0]['prediction_created_at'])
if __name__=='__main__':unittest.main()

