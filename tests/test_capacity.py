import pandas as pd
from src.capacity import normalize_orders
from src.clustering import cluster_orders

def test_weight_and_pallet_limits():
    orders=normalize_orders(pd.DataFrame([{
        'lat':48.8,'lon':9.3,'auftrag':'A1','kunde':'K','strasse':'S 1','plz':'1','ort':'O','paletten':18,'warengewicht_kg':5900,'service_min':10
    }]))
    vehicles=pd.DataFrame([{'vehicle_id':'V1','vehicle_class':'14 t','pallet_capacity':18,'payload_kg':6000,'color':'#000','available':True}])
    ass,_,warnings=cluster_orders(orders,vehicles,(48.8,9.2))
    assert ass.iloc[0].vehicle_id=='NICHT ZUGEWIESEN'
    assert warnings
