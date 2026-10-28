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
        self.bundle=load_forecast_model(ROOT/'training_pipeline/live_result_20/model/model.joblib')
        self.route={'vehicle_id':'test','stops':[dict(auftrag='A',lat=48.79,lon=9.18,service_min=5,paletten=2,warengewicht_kg=700)],'legs':[{'duration':600,'distance':5000}]*2}
        self.depot={'lat':48.8,'lon':9.2}
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
