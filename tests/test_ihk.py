import unittest
import tempfile
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
import pandas as pd
from src.ml_training import train_experiment, validate_data
from src.forecast import calculate_tour_forecast, recalculate_tour_from_stop
from src import training_logger
from types import SimpleNamespace

def dataset():
    rows=[]
    for tour in range(10):
        for segment in range(3):
            departure=datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(days=tour,hours=segment)
            rows.append(dict(tour_id=str(tour),segment_id=segment,prediction_created_at=(departure-timedelta(hours=1)).isoformat(),planned_departure_at=departure.isoformat(),actual_departure_at=departure.isoformat(),actual_arrival_at=(departure+timedelta(seconds=600)).isoformat(),actual_recorded_at=(departure+timedelta(seconds=601)).isoformat(),osrm_baseline_s=500,distance_m=5000,data_origin='synthetic'))
    return pd.DataFrame(rows)

class Tests(unittest.TestCase):
    def test_split(self):
        bundle, metrics, details=train_experiment(dataset())
        self.assertFalse(set(bundle['report']['train_tours']) & set(bundle['report']['test_tours']))
        self.assertEqual(len(details),6)
        self.assertEqual(len(metrics),2)
    def test_reject_leakage_and_duplicates(self):
        df=dataset(); df.loc[0,'prediction_created_at']='2027-01-01T00:00:00+00:00'
        with self.assertRaises(ValueError): validate_data(df)
        with self.assertRaises(ValueError): validate_data(pd.concat([dataset(),dataset()]))
    def test_unknown_values_and_mixed_origin(self):
        df=dataset(); df.loc[0,'distance_m']=float('nan')
        with self.assertRaises(ValueError): validate_data(df)
        df=dataset(); df.loc[0,'data_origin']='real'
        with self.assertRaises(ValueError): validate_data(df)
    def test_outputs_are_not_features_and_late_labels_purged(self):
        df=dataset(); df['model_forecast_s']=df.index * 9999
        a, _, _=train_experiment(df)
        self.assertNotIn('model_forecast_s',a['report']['features'])
        df.loc[df.tour_id=='0','actual_recorded_at']='2027-01-01T00:00:00+00:00'
        b, _, _=train_experiment(df)
        self.assertNotIn('0',b['report']['train_tours'])
    def test_no_future_service_and_reforecast_identity(self):
        response=SimpleNamespace(success=True,tomtom_no_traffic_s=500,tomtom_historic_s=500,tomtom_live_s=500,traffic_delay_s=0,distance_m=5000,error=None)
        route={'vehicle_id':'V','stops':[{'auftrag':'A','lat':48,'lon':9,'service_min':1,'actual_service_s':900}], 'legs':[{'duration':500,'distance':5000}]*2}
        with patch('src.forecast.TomTomRoutingProvider') as provider:
            provider.return_value.calculate_segment.return_value=response
            initial=calculate_tour_forecast(route,{'lat':48,'lon':8},'2026-01-01T00:00:00+00:00')
            self.assertIsNone(initial['segments'][0]['actual_service_s'])
            updated=recalculate_tour_from_stop(initial,route,{'lat':48,'lon':8},'A',actual_service_s=120)
            self.assertEqual(updated['segments'][0]['forecast_version'],1)
            self.assertIsNone(initial['segments'][0]['actual_service_s'])
            bundle,_,_=train_experiment(dataset())
            ml=calculate_tour_forecast(route,{'lat':48,'lon':8},'2026-01-01T00:00:00+00:00',ml_bundle=bundle)
            self.assertEqual(ml['segments'][0]['model_id'],bundle['report']['model_id'])
        with tempfile.TemporaryDirectory() as folder, patch.object(training_logger,'LOCAL_DATA_DIR',__import__('pathlib').Path(folder)), patch.object(training_logger,'HF_TOKEN',''):
            a=training_logger.log_forecast(updated)
            b=training_logger.log_forecast(updated)
            self.assertEqual(a['path'],b['path'])
            import json
            record=json.loads(__import__('pathlib').Path(a['path']).read_text())
            self.assertEqual(record['segments'][0]['forecast_version'],1)
            self.assertNotIn('actual_service_s',record['segments'][0]['inputs'])

if __name__=='__main__': unittest.main()
