import unittest
import pandas as pd
from src.driver_planning import driver_fleet, DRIVER_COLUMNS
from src.clustering import DRIVER_DEFAULTS,cluster_orders,plan_day
class DriverTests(unittest.TestCase):
    def test_driver_budget_limits_assignment(self):
        fleet=pd.DataFrame([dict(vehicle_id='V',pallet_capacity=10,payload_kg=1000)])
        driver={'driver_id':'D',**DRIVER_DEFAULTS,'remaining_drive_min':50}
        vehicles=driver_fleet(fleet,pd.DataFrame([driver]))
        orders=pd.DataFrame([dict(auftrag='A',lat=48,lon=9,paletten=1,gesamtgewicht_kg=100,service_min=0)])
        assigned,_,_=cluster_orders(orders,vehicles,(48,9),[[0,30],[30,0]])
        self.assertEqual(assigned.iloc[0].vehicle_id,'NICHT ZUGEWIESEN')
    def test_pause_and_rest_deadline(self):
        p=plan_day([20,20],[10],{**DRIVER_DEFAULTS,'drive_since_break_min':260,'remaining_duty_min':80})
        self.assertEqual(p['pausen_min'],45)
        self.assertEqual(p['tourdauer_min'],95)
        self.assertFalse(p['feasible'])
    def test_empty_duplicate_and_excess_budget(self):
        fleet=pd.DataFrame([dict(vehicle_id='V'),dict(vehicle_id='W')])
        d={'driver_id':'D',**DRIVER_DEFAULTS}
        for rows in [[],[d,d],[{**d,'remaining_drive_min':600}]]:
            with self.assertRaises(ValueError):driver_fleet(fleet,rows)
