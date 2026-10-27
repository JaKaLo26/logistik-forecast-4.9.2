import argparse
from pathlib import Path
from .core import Config, generate, train, predict
import pandas as pd

p=argparse.ArgumentParser(description='Standalone Logistik-PoC: Generator → Training → Modell')
s=p.add_subparsers(dest='command',required=True)
g=s.add_parser('generate')
g.add_argument('--out',required=True)
g.add_argument('--tours',type=int,default=100)
g.add_argument('--min-stops',type=int,default=3)
g.add_argument('--max-stops',type=int,default=8)
g.add_argument('--seed',type=int,default=42)
m=g.add_mutually_exclusive_group(required=True)
m.add_argument('--demo',action='store_true')
m.add_argument('--addresses',help='Geprüfter Adresspool mit strasse,plz,ort,lat,lon')
g.add_argument('--snapshots',help='Vorhandene routing_snapshots für reproduzierbare Wiederholung')
g.add_argument('--source',help='Quellennachweis JSON für den Adresspool')
g.add_argument('--osrm-url',default='https://router.project-osrm.org')
t=s.add_parser('train')
t.add_argument('--data',required=True)
t.add_argument('--out',required=True)
t.add_argument('--seed',type=int,default=42)
f=s.add_parser('predict')
f.add_argument('--model',required=True)
f.add_argument('--data',required=True)
f.add_argument('--out',required=True)
a=p.parse_args()
if a.command=='generate':
    if a.addresses and not a.source:
        p.error('--addresses benötigt --source')
    print(generate(a.out,Config(tours=a.tours,min_stops=a.min_stops,max_stops=a.max_stops,seed=a.seed),a.addresses,a.source,a.osrm_url,a.snapshots))
elif a.command=='train':
    print(train(a.data,a.out,a.seed)['metrics'])
else:
    data=pd.read_csv(a.data)
    data['predicted_travel_s']=predict(a.model,data)
    with Path(a.out).open('x',encoding='utf-8') as stream:
        data.to_csv(stream,index=False)
