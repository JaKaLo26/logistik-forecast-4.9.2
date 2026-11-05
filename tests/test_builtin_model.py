import unittest
from unittest.mock import Mock
import math
from src.builtin_model import load_builtin_model,predict_road_matrix
class BuiltinTests(unittest.TestCase):
    def test_saved_model_contract(self):
        bundle=load_builtin_model()
        self.assertEqual(bundle['report']['data_origin'],'synthetic')
        report=bundle['report']
        groups=[set(report[k]) for k in ('train_tours','validation_tours','test_tours')]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    def test_planning_uses_model_times(self):
        model=Mock();model.predict.return_value=[1200,1800]
        result=predict_road_matrix([[0,10],[12,0]],[[0,5000],[6000,0]],'2026-11-05T08:00:00+01:00',{'model':model})
        self.assertEqual(result,[[0,20],[30,0]])
        self.assertEqual(list(model.predict.call_args.args[0].osrm_baseline_s),[600,720])
    def test_unreachable_and_coincident(self):
        model=Mock()
        result=predict_road_matrix([[0,math.inf],[0,0]],[[0,math.inf],[0,0]],'2026-11-05T08:00:00+01:00',{'model':model})
        self.assertTrue(math.isinf(result[0][1]));model.predict.assert_not_called()
    def test_bad_distance_rejected(self):
        with self.assertRaises(ValueError):
            predict_road_matrix([[0,10],[10,0]],[[0,0],[1000,0]],'2026-11-05T08:00:00+01:00',load_builtin_model())
