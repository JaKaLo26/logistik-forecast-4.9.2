import unittest
import pandas as pd
from src.clustering import plan_day, cluster_orders

class PlanningTests(unittest.TestCase):
    def test_break_boundary(self):
        self.assertEqual(plan_day([270], [], {})['pausen_min'], 0)
        self.assertEqual(plan_day([271], [], {})['pausen_min'], 45)
    def test_service_is_not_rest(self):
        p = plan_day([240, 60], [60], {})
        self.assertEqual(p['pausen_min'], 45)
        self.assertEqual(p['tourdauer_min'], 405)
    def test_other_work_break(self):
        self.assertEqual(plan_day([0, 0], [361], {})['pausen_min'], 45)
    def test_preexisting_driving(self):
        self.assertEqual(plan_day([31], [], {'drive_since_break_min':240})['pausen_min'], 45)
    def test_rest_and_week_budgets(self):
        for v in ({'remaining_duty_min':59}, {'remaining_week_drive_min':59},
                  {'remaining_fortnight_drive_min':59}, {'minutes_until_weekly_rest':59}):
            self.assertFalse(plan_day([60], [], v)['feasible'])
    def test_daily_limit(self):
        self.assertFalse(plan_day([541], [], {'remaining_work_min':1000})['feasible'])
    def test_road_assignment(self):
        orders = pd.DataFrame([{'auftrag':str(i), 'lat':48+i*.001, 'lon':9,
                               'paletten':1, 'gesamtgewicht_kg':100, 'service_min':0} for i in range(3)])
        vehicles = pd.DataFrame([{'vehicle_id':str(i), 'pallet_capacity':2, 'payload_kg':1000} for i in range(2)])
        # First two coordinates are nearest geometrically, but 100 minutes apart by road.
        matrix = [[0,30,20,20],[30,0,100,1],[20,100,0,100],[20,1,100,0]]
        result, _, _ = cluster_orders(orders, vehicles, (48,9), matrix)
        first = result[result.vehicle_id=='0'].auftrag.tolist()
        self.assertEqual(set(first), {'0','2'})
    def test_return_must_fit(self):
        orders = pd.DataFrame([{'auftrag':'a','lat':48,'lon':9,'paletten':1,'gesamtgewicht_kg':100,'service_min':0}])
        vehicles = pd.DataFrame([{'vehicle_id':'v','pallet_capacity':2,'payload_kg':1000,'remaining_drive_min':50}])
        result, _, _ = cluster_orders(orders, vehicles, (48,9), [[0,30],[30,0]])
        self.assertEqual(result.iloc[0].vehicle_id, 'NICHT ZUGEWIESEN')
    def test_no_vehicle(self):
        orders = pd.DataFrame([{'auftrag':'a','lat':48,'lon':9,'paletten':1,'gesamtgewicht_kg':100,'service_min':0}])
        result, _, _ = cluster_orders(orders, pd.DataFrame(), (48,9))
        self.assertEqual(result.iloc[0].vehicle_id, 'NICHT ZUGEWIESEN')

if __name__ == '__main__':
    unittest.main()
