"""Reproducible PoC. Targets are simulations, never measured truck travel times."""
from __future__ import annotations
import hashlib
import json
import platform
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

VERSION = '1.0.0'
FEATURES = ['baseline_s', 'distance_m', 'departure_hour', 'weekday', 'remaining_load_kg']
ORDER_COLUMNS = ['auftrag', 'kunde', 'strasse', 'plz', 'ort', 'paletten', 'warengewicht_kg', 'service_min']
ADDRESS_COLUMNS = ['strasse', 'plz', 'ort', 'lat', 'lon']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


@dataclass
class Config:
    tours: int = 100
    min_stops: int = 3
    max_stops: int = 8
    seed: int = 42
    min_pallets: int = 1
    max_pallets: int = 3
    min_service: int = 5
    max_service: int = 20
    pallet_goods_kg: float = 350

    def validate(self):
        for name in ('tours','min_stops','max_stops','seed','min_pallets','max_pallets','min_service','max_service'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name}: nicht negative ganze Zahl erforderlich')
        if self.tours < 10 or not 1 <= self.min_stops <= self.max_stops:
            raise ValueError('Mindestens 10 Touren und 1 ≤ Stopps von ≤ bis erforderlich')
        if not 1 <= self.min_pallets <= self.max_pallets or self.min_service > self.max_service:
            raise ValueError('Ungültige Paletten-/Servicebereiche')
        if not np.isfinite(self.pallet_goods_kg) or self.pallet_goods_kg <= 0:
            raise ValueError('Warengewicht je Palette muss positiv sein')


def demo_addresses():
    # Deliberately fictional, never presented as geocoded delivery addresses.
    return pd.DataFrame([dict(strasse=f'DEMO-NICHT-GEOCODIEREN {i+1}', plz='00000',
        ort='Synthetischer Testort', lat=48.5+i*.015, lon=9.0+(i%7)*.025) for i in range(40)])


def read_addresses(path):
    df = pd.read_csv(path, dtype={'plz':str}, sep=None, engine='python')
    if not set(ADDRESS_COLUMNS) <= set(df):
        raise ValueError('Adresspool benötigt: '+', '.join(ADDRESS_COLUMNS))
    if df[ADDRESS_COLUMNS].isna().any().any():
        raise ValueError('Unvollständiger Adresspool')
    for c, low, high in [('lat',47,56),('lon',5,16)]:
        df[c] = pd.to_numeric(df[c], errors='raise')
        if not (np.isfinite(df[c]) & df[c].between(low,high)).all():
            raise ValueError('Koordinaten außerhalb des Deutschland-Prüfbereichs')
    if not df.plz.str.fullmatch(r'\d{5}').all():
        raise ValueError('PLZ muss fünf Ziffern enthalten')
    return df.drop_duplicates(ADDRESS_COLUMNS).reset_index(drop=True)


def provenance(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    required = ['author','source_url','data_version','retrieved_at','license','license_url','terms_url']
    if any(not data.get(k) for k in required):
        raise ValueError('Quellennachweis benötigt: '+', '.join(required))
    if datetime.fromisoformat(data['retrieved_at'].replace('Z','+00:00')).tzinfo is None:
        raise ValueError('Abrufzeit benötigt Zeitzone')
    return data


def route(a, b, mode, cache, endpoint):
    coords = f"{a.lon},{a.lat};{b.lon},{b.lat}"
    key = hashlib.sha256((endpoint+coords).encode()).hexdigest()
    if mode == 'demo':
        # Explicit synthetic test geometry, not road routing or OSRM data.
        km = max(.1, float(np.hypot((a.lat-b.lat)*111, (a.lon-b.lon)*74))*1.4)
        return km*1000, km/55*3600, 'synthetic-geometry'
    path = cache / (key+'.json')
    if not path.exists():
        import requests
        response = requests.get(endpoint.rstrip('/')+'/route/v1/driving/'+coords,
                                params={'overview':'false'}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get('code') != 'Ok' or not payload.get('routes'):
            raise ValueError('Keine Straßenroute; kein Luftlinien-Fallback')
        write_json(path, {'retrieved_at':datetime.now(timezone.utc).isoformat(),
                         'coordinates':coords, 'endpoint':endpoint, 'profile':'driving',
                         'api_version':'v1', 'data_version':'not provided by service', 'response':payload})
    r = json.loads(path.read_text())['response']['routes'][0]
    distance, seconds = float(r['distance']), float(r['duration'])
    if not np.isfinite([distance,seconds]).all() or min(distance,seconds)<=0:
        raise ValueError('Ungültige Straßenroute')
    return distance, seconds, key


def generate(out, config=Config(), addresses=None, source=None, endpoint='https://router.project-osrm.org', snapshots=None):
    config.validate()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite an experiment.
    cache = out/'routing_snapshots'
    cache.mkdir()
    if snapshots is not None:
        for snapshot in Path(snapshots).glob('*.json'):
            shutil.copy2(snapshot, cache/snapshot.name)
    mode = 'demo' if addresses is None else 'osrm'
    pool = demo_addresses() if addresses is None else read_addresses(addresses)
    src = {'author':'Project generator', 'data_origin':'synthetic', 'external_data_used':False} if mode=='demo' else provenance(source)
    if len(pool) < config.max_stops+1:
        raise ValueError('Adresspool benötigt max_stopps + 1 unterschiedliche Adressen')
    pool.to_csv(out/'address_pool.csv', index=False)
    rng = np.random.default_rng(config.seed)
    orders, segments = [], []
    for tour in range(config.tours):
        tid = f'T{tour:06d}'
        count = int(rng.integers(config.min_stops, config.max_stops+1))
        ids = rng.choice(len(pool), size=count+1, replace=False)
        pallets = rng.integers(config.min_pallets, config.max_pallets+1, count)
        load = float(pallets.sum()*(config.pallet_goods_kg+25))
        hour = float(rng.integers(5,18))
        weekday = int(rng.integers(0,7))
        for j in range(count):
            addr = pool.iloc[ids[j+1]]
            oid = f'{tid}-S{j+1:03d}'
            service = int(rng.integers(config.min_service, config.max_service+1))
            orders.append(dict(auftrag=oid, kunde=f'Synthetischer Kunde {j+1}',
                **{k:str(addr[k]) for k in ['strasse','plz','ort']}, paletten=int(pallets[j]),
                warengewicht_kg=float(pallets[j]*config.pallet_goods_kg), service_min=service))
            distance, baseline, rid = route(pool.iloc[ids[j]],addr,mode,cache,endpoint)
            rush = .25 if (7<=hour<9 or 16<=hour<18) and weekday<5 else 0
            # Scenario equation, not a learned forecast. Random effects remain unobservable.
            target = max(1., baseline*(1.08+rush+load/200000+rng.normal(0,.06)))
            segments.append(dict(tour_id=tid,segment_id=oid,baseline_s=baseline,distance_m=distance,
                departure_hour=hour,weekday=weekday,remaining_load_kg=load,synthetic_target_s=target,
                data_origin='synthetic',route_ref=rid,from_pool_index=int(ids[j]),to_pool_index=int(ids[j+1])))
            # Planned chronology only: no previous target determines future features.
            next_hour = hour+(baseline+service*60)/3600
            weekday = (weekday+int(next_hour//24))%7
            hour = next_hour%24
            load -= float(pallets[j]*(config.pallet_goods_kg+25))
    pd.DataFrame(orders)[ORDER_COLUMNS].to_csv(out/'orders.csv', index=False)
    pd.DataFrame(segments).to_csv(out/'segments.csv',index=False)
    manifest = dict(generator_version=VERSION, parameters=asdict(config), mode=mode,
        source=src, endpoint=endpoint if mode=='osrm' else None,
        routing_source=None if mode=='demo' else dict(author='OpenStreetMap contributors; OSRM contributors',
            source_url=endpoint, api_version='v1', data_version='not provided by service',
            retrieved_at='See individual routing_snapshots/*.json',
            license='OSM: ODbL 1.0; OSRM software: BSD-2-Clause',
            license_url='https://www.openstreetmap.org/copyright',
            software_license_url='https://github.com/Project-OSRM/osrm-backend/blob/master/LICENSE.TXT',
            terms_url='https://github.com/Project-OSRM/osrm-backend/wiki/Api-usage-policy',
            profile='driving; not verified for trucks'),
        target_definition='Synthetic multiplicative baseline, rush-hour/load effects and Gaussian noise; never IST',
        transformations=[dict(method='Sample without replacement per tour; synthetic orders and labels',
            reason='Technical feasibility demonstration',script_version=VERSION,rows=len(segments))],
        hashes={p.name:digest(p) for p in out.rglob('*') if p.is_file()})
    write_json(out/'generation.json',manifest)
    return out


def validate_segments(df):
    required = FEATURES+['tour_id','segment_id','synthetic_target_s','data_origin']
    if not set(required)<=set(df) or df[required].isna().any().any():
        raise ValueError('Unvollständiges Segmentschema')
    if not df.data_origin.eq('synthetic').all():
        raise ValueError('Diese Pipeline erwartet eindeutig synthetische Zielzeiten')
    if df.duplicated(['tour_id','segment_id']).any():
        raise ValueError('Doppelte Segmente')
    for key in ['tour_id','segment_id']:
        if df[key].astype(str).str.strip().eq('').any():
            raise ValueError('Leere IDs')
    for key in FEATURES+['synthetic_target_s']:
        df[key] = pd.to_numeric(df[key], errors='raise')
        if not np.isfinite(df[key]).all():
            raise ValueError('Nicht endliche Zahlen')
    if (df[['baseline_s','distance_m','synthetic_target_s']]<=0).any().any() or (df.remaining_load_kg<0).any():
        raise ValueError('Ungültige Fahrzeiten/Distanz/Last')
    if not df.departure_hour.between(0,24,inclusive='left').all() or not df.weekday.isin(range(7)).all():
        raise ValueError('Ungültige Zeitmerkmale')
    return df


def split_tours(df, seed):
    # Content fingerprints ignore IDs and labels: re-imported copies cannot cross splits.
    groups = {}
    for tid, rows in df.groupby('tour_id', sort=True):
        signature = hashlib.sha256(rows[FEATURES].sort_values(FEATURES).to_csv(index=False).encode()).hexdigest()
        groups.setdefault(signature, []).append(str(tid))
    keys = sorted(groups)
    if len(keys)<10:
        raise ValueError('Mindestens 10 unabhängige Tourgruppen erforderlich')
    keys = np.random.default_rng(seed).permutation(keys)
    a,b = int(len(keys)*.7),int(len(keys)*.85)
    return {name:sorted(t for k in part for t in groups[k]) for name,part in
            zip(['train','validation','test'],[keys[:a],keys[a:b],keys[b:]])}


def predict(model_path, frame):
    # Only trusted local artifacts: joblib files can execute Python on load.
    bundle = joblib.load(model_path)
    if bundle['features'] != FEATURES or bundle['schema_version'] != VERSION:
        raise ValueError('Inkompatibles Modellschema')
    return bundle['model'].predict(frame[FEATURES])


def train(dataset, output, seed=42):
    dataset, output = Path(dataset), Path(output)
    df = validate_segments(pd.read_csv(dataset, dtype={'tour_id':str,'segment_id':str}))
    splits = split_tours(df,seed)
    parts = {k:df[df.tour_id.isin(v)].copy() for k,v in splits.items()}
    output.mkdir(parents=True,exist_ok=False)
    candidates=[]
    for depth in [6,12]:
        model = RandomForestRegressor(n_estimators=80,max_depth=depth,min_samples_leaf=2,random_state=seed,n_jobs=1)
        model.fit(parts['train'][FEATURES],parts['train'].synthetic_target_s)
        score = float(mean_absolute_error(parts['validation'].synthetic_target_s,model.predict(parts['validation'][FEATURES])))
        candidates.append((score,depth,model))
    score,depth,model = min(candidates,key=lambda x:x[0])
    joblib.dump(dict(model=model,features=FEATURES,schema_version=VERSION,data_origin='synthetic'),output/'model.joblib', compress=3)
    test = parts['test']
    prediction = predict(output/'model.joblib',test[FEATURES])
    if not np.allclose(prediction,model.predict(test[FEATURES]),rtol=0,atol=0):
        raise AssertionError('Modell-Roundtrip verändert Vorhersagen')
    def metrics(values):
        y=test.synthetic_target_s
        return dict(MAE_s=float(mean_absolute_error(y,values)),RMSE_s=float(np.sqrt(mean_squared_error(y,values))),
                    bias_s=float(np.mean(values-y)))
    report=dict(schema_version=VERSION,dataset_sha256=digest(dataset),seed=seed,features=FEATURES,splits=splits,
        validation_candidates=[dict(max_depth=d,MAE_s=s) for s,d,_ in candidates],selected_max_depth=depth,
        metrics=dict(baseline=metrics(test.baseline_s),model=metrics(prediction)),reload_exact_match=True,
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
        limitation='Synthetic technical PoC only. No real truck accuracy claim. Random grouped split, not future generalization.')
    for name,part in parts.items():
        part.to_csv(output/f'{name}.csv',index=False)
    result=test[['tour_id','segment_id','synthetic_target_s']].copy()
    result['predicted_travel_s']=prediction
    result.to_csv(output/'test_predictions.csv',index=False)
    report['model_sha256']=digest(output/'model.joblib')
    write_json(output/'report.json',report)
    return report
