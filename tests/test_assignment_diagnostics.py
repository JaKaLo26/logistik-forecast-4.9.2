import unittest
import pandas as pd
from src.clustering import cluster_orders, assignment_failure_reason

class AssignmentDiagnostics(unittest.TestCase):
    def test_all_feasible_orders_get_vehicles(self):
        n=160
        orders=pd.DataFrame([dict(auftrag=str(i),lat=48,lon=9,paletten=1,gesamtgewicht_kg=100,service_min=10) for i in range(n)])
        fleet=pd.DataFrame([dict(vehicle_id=str(i),pallet_capacity=1,payload_kg=1000) for i in range(n)])
        matrix=[[0 if i==j else 15 for j in range(n+1)] for i in range(n+1)]
        assigned,summary,_=cluster_orders(orders,fleet,(48,9),matrix)
        self.assertEqual(len(assigned),n)
        self.assertEqual(assigned.auftrag.nunique(),n)
        self.assertFalse(assigned.vehicle_id.eq('NICHT ZUGEWIESEN').any())
        self.assertEqual(len(summary),n)

    def test_long_single_tour_explains_time_not_vehicle_count(self):
        row=dict(paletten=1,gesamtgewicht_kg=100,service_min=10)
        fleet=[dict(pallet_capacity=33,payload_kg=24000)]*10
        reason=assignment_failure_reason(row,fleet,300,300)
        self.assertIn('600.0 > 540.0',reason)
        self.assertIn('610.0 > 480.0',reason)
        self.assertIn('zusätzliche LKW allein',reason)

    def test_missing_route_capacity_and_fleet(self):
        row=dict(paletten=1,gesamtgewicht_kg=100,service_min=10)
        self.assertIn('Kein verfügbarer',assignment_failure_reason(row,[],0,0))
        self.assertIn('Fahrzeugkapazität',assignment_failure_reason(row,[dict(pallet_capacity=0,payload_kg=1000)],0,0))
        self.assertIn('Straßenverbindung',assignment_failure_reason(row,[dict(pallet_capacity=1,payload_kg=1000)],float('inf'),0))
