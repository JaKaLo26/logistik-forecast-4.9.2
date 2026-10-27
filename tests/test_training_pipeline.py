import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from training_pipeline.core import Config, generate, train, predict, split_tours, validate_segments, FEATURES, route
from src.capacity import normalize_orders

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.data=generate(self.root/'data',Config(tours=20,min_stops=2,max_stops=3))
    def tearDown(self):
        self.tmp.cleanup()
    def test_reproducible_and_csv_compatible(self):
        other=generate(self.root/'other',Config(tours=20,min_stops=2,max_stops=3))
        for name in ['orders.csv','segments.csv','generation.json']:
            self.assertEqual((self.data/name).read_bytes(),(other/name).read_bytes())
        orders=normalize_orders(pd.read_csv(self.data/'orders.csv',dtype={'plz':str}))
        self.assertTrue(orders.plz.eq('00000').all())
        self.assertGreater(len(orders),0)
    def test_train_reload_predict_without_labels(self):
        report=train(self.data/'segments.csv',self.root/'model')
        splits=[set(v) for v in report['splits'].values()]
        self.assertEqual(sum(map(len,splits)),20)
        self.assertFalse(splits[0]&splits[1] or splits[0]&splits[2] or splits[1]&splits[2])
        test=pd.read_csv(self.root/'model/test.csv')
        with patch('training_pipeline.core.generate',side_effect=AssertionError('Generator used')):
            y=predict(self.root/'model/model.joblib',test[FEATURES])
        self.assertTrue(np.isfinite(y).all())
        self.assertTrue(report['reload_exact_match'])
        self.assertNotIn('synthetic_target_s',report['features'])
    def test_copied_tour_same_split(self):
        df=pd.read_csv(self.data/'segments.csv')
        copy=df[df.tour_id=='T000000'].copy()
        copy['tour_id']='copy'
        copy['segment_id']='copy-'+copy.segment_id
        split=split_tours(pd.concat([df,copy]),42)
        self.assertEqual(next(k for k,v in split.items() if 'copy' in v),next(k for k,v in split.items() if 'T000000' in v))
    def test_invalid_data_rejected(self):
        for col,val in [('baseline_s',float('nan')),('synthetic_target_s',-1),('data_origin','real'),('departure_hour',24)]:
            df=pd.read_csv(self.data/'segments.csv');df.loc[0,col]=val
            with self.assertRaises(ValueError): validate_segments(df)
    def test_target_changes_do_not_change_features(self):
        df=pd.read_csv(self.data/'segments.csv'); before=split_tours(df,42)
        df['synthetic_target_s']*=2
        self.assertEqual(before,split_tours(df,42))
    def test_no_overwrite(self):
        with self.assertRaises(FileExistsError): generate(self.data)
    def test_route_snapshot_and_no_fallback(self):
        a=pd.Series({'lon':9.,'lat':48.});b=pd.Series({'lon':9.2,'lat':48.1})
        with patch('requests.get') as get:
            get.return_value.json.return_value={'code':'Ok','routes':[{'distance':20000,'duration':1800}]}
            first=route(a,b,'osrm',self.data/'routing_snapshots','https://example.test')
            second=route(a,b,'osrm',self.data/'routing_snapshots','https://example.test')
            self.assertEqual(first,second);self.assertEqual(get.call_count,1)
            get.return_value.json.return_value={'code':'NoRoute'}
            with self.assertRaises(ValueError):route(b,a,'osrm',self.data/'routing_snapshots','https://example.test')

if __name__=='__main__':unittest.main()
