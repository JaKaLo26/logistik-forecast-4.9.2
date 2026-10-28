"""Small real HTTP smoke test. Explicitly invoked, never part of offline CI."""
import csv,json,time,sys,subprocess,argparse
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
import requests
import pandas as pd
import numpy as np
from .core import Config, generate, train, digest
from src.capacity import normalize_orders

parser=argparse.ArgumentParser(description='Live-PoC mit echten Adressen und synthetischen Zielzeiten')
parser.add_argument('output')
parser.add_argument('--tours',type=int,default=12)
args=parser.parse_args()
Config(tours=args.tours).validate()
root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
headers={'User-Agent':'LogistikForecast-PoC-LiveCheck/1.0 (https://github.com/JaKaLo26/logistik-forecast-4.9.2)'}
rows=[]
for i,query in enumerate(['Mailänder Platz 1, Stuttgart, Deutschland','Mercedesstraße 100, Stuttgart, Deutschland','Porscheplatz 1, Stuttgart, Deutschland']):
    time.sleep(1.1)
    r=requests.get('https://nominatim.openstreetmap.org/search',params={'q':query,'format':'jsonv2','addressdetails':1,'limit':1},headers=headers,timeout=30)
    r.raise_for_status();hits=r.json()
    (root/f'geocode-{i}.json').write_text(json.dumps(dict(query=query,retrieved_at=datetime.now(timezone.utc).isoformat(),response=hits),indent=2,ensure_ascii=False))
    if not hits:raise ValueError('Adresse nicht gefunden: '+query)
    h=hits[0];a=h['address']
    rows.append(dict(strasse=a['road']+' '+a['house_number'],plz=a['postcode'],ort=a.get('city',a.get('town')),lat=h['lat'],lon=h['lon']))
    print('Adresse bestätigt:',rows[-1]['strasse'],flush=True)
with (root/'addresses.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
source=dict(author='OpenStreetMap contributors',source_url='https://nominatim.openstreetmap.org/search',data_version='not provided by service',retrieved_at=datetime.now(timezone.utc).isoformat(),license='ODbL 1.0',license_url='https://www.openstreetmap.org/copyright',terms_url='https://operations.osmfoundation.org/policies/nominatim/',transformation='Three explicit public-place queries; returned address fields and coordinates mapped to CSV; original results preserved',script_version='live_check-1.1',rows=3)
(root/'source.json').write_text(json.dumps(source,indent=2))
original=requests.get;calls=[]
def polite_get(url,**kwargs):
    time.sleep(1.1);kwargs['headers']=headers
    response=original(url,**kwargs)
    calls.append(dict(url=url,status=response.status_code))
    print('OSRM-Anfrage',len(calls),'HTTP',response.status_code,flush=True)
    return response
cfg=Config(tours=args.tours,min_stops=2,max_stops=2,seed=20261028)
with patch('requests.get',side_effect=polite_get):
    data=generate(root/'data',cfg,root/'addresses.csv',root/'source.json')
orders=normalize_orders(pd.read_csv(data/'orders.csv',dtype={'plz':str}))
report=train(data/'segments.csv',root/'model',20261028)
with patch('requests.get',side_effect=AssertionError('Unerwarteter Netzaufruf bei Snapshot-Wiederholung')):
    replay=generate(root/'replay',cfg,root/'addresses.csv',root/'source.json',snapshots=data/'routing_snapshots')
for name in ['orders.csv','segments.csv','generation.json']:
    assert digest(data/name)==digest(replay/name),name
frame=pd.read_csv(root/'model/test.csv').drop(columns=['synthetic_target_s'])
frame.to_csv(root/'inference-input.csv',index=False)
subprocess.run([sys.executable,'-m','training_pipeline','predict','--model',str(root/'model/model.joblib'),'--data',str(root/'inference-input.csv'),'--out',str(root/'predictions.csv')],check=True)
y=pd.read_csv(root/'predictions.csv').predicted_travel_s
assert np.isfinite(y).all() and (y>0).all()
summary=dict(status='passed',tested_at=datetime.now(timezone.utc).isoformat(),tours=cfg.tours,unique_addresses=len(rows),orders=len(orders),live_osrm_requests=len(calls),http_statuses=sorted(set(c['status'] for c in calls)),split_tours={k:len(v) for k,v in report['splits'].items()},test_segments=len(y),snapshot_replay_identical=True,inference_fresh_process_without_labels=True,model_bytes=(root/'model/model.joblib').stat().st_size,model_sha256=digest(root/'model/model.joblib'),metrics=report['metrics'],limitations=['Synthetic labels, not real truck accuracy','OSRM driving profile, not truck restrictions','No TomTom calls','Not a deployed Hugging Face/browser test'],transport='Real requests with identification header and 1.1 second throttle; no response mocks')
(root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
