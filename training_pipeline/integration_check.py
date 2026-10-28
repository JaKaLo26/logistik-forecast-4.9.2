"""Train -> disk -> fresh process -> order CSV -> production forecast -> assertions."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
from unittest.mock import patch
import pandas as pd
from .core import train,digest,FEATURES
from .forecast_csv import run
from src.ml_training import load_forecast_model,predict_seconds


def worker(root):
    # Both generator and heuristic are forbidden during inference; no external calls.
    with patch('training_pipeline.core.generate',side_effect=AssertionError('Generator used in forecast')), \
         patch('src.forecast.calculate_model_forecast_s',side_effect=AssertionError('Heuristic used')), \
         patch('requests.sessions.Session.request',side_effect=AssertionError('Unexpected live request')):
        result=run(root/'model/model.joblib',root/'forecast_orders.csv',root/'forecast_plan.json',root/'forecasts.json')
    return result


def check(source,root):
    root.mkdir(parents=True,exist_ok=False)
    report=train(source/'data/segments.csv',root/'model',20261028)
    orders=pd.read_csv(source/'data/orders.csv',dtype={'auftrag':str,'plz':str})
    pool=pd.read_csv(source/'data/address_pool.csv',dtype={'plz':str})
    # Forecast preparation reads only features/identities, never target values.
    fields=['tour_id','segment_id','from_pool_index','to_pool_index','baseline_s','distance_m','weekday','departure_hour']
    segments=pd.read_csv(source/'data/segments.csv',usecols=fields)
    segments=segments[segments.tour_id.isin(report['splits']['test'])]
    orders[orders.auftrag.isin(segments.segment_id)].to_csv(root/'forecast_orders.csv',index=False)
    snapshots=[json.loads(p.read_text()) for p in (source/'data/routing_snapshots').glob('*.json')]
    plans=[]
    for tid,group in segments.groupby('tour_id'):
        group=group.sort_values('segment_id');first=group.iloc[0];last=group.iloc[-1]
        depot=pool.iloc[int(first.from_pool_index)]
        stops=[dict(auftrag=r.segment_id,lat=float(pool.iloc[int(r.to_pool_index)].lat),lon=float(pool.iloc[int(r.to_pool_index)].lon)) for r in group.itertuples()]
        legs=[dict(duration=float(r.baseline_s),distance=float(r.distance_m)) for r in group.itertuples()]
        origin=pool.iloc[int(last.to_pool_index)]
        coords=f'{origin.lon},{origin.lat};{depot.lon},{depot.lat}'
        snapshot=next(s for s in snapshots if s['coordinates']==coords)
        r=snapshot['response']['routes'][0];legs.append(dict(duration=r['duration'],distance=r['distance']))
        start=datetime(2026,10,19,tzinfo=ZoneInfo('Europe/Berlin'))+timedelta(days=int(first.weekday),hours=float(first.departure_hour))
        plans.append(dict(tour_id=tid,depot=dict(lat=float(depot.lat),lon=float(depot.lon)),stops=stops,legs=legs,start_time=start.isoformat()))
    (root/'forecast_plan.json').write_text(json.dumps(plans,indent=2))
    env=dict(os.environ);env['TOMTOM_API_KEY']=''
    subprocess.run([sys.executable,'-m','training_pipeline.integration_check','--worker',str(root)],check=True,env=env)
    results=json.loads((root/'forecasts.json').read_text())
    bundle=load_forecast_model(root/'model/model.joblib')
    imported=pd.read_csv(root/'forecast_orders.csv').set_index('auftrag')
    checks=[]
    for plan,result in zip(plans,results):
        prior=None
        for i,segment in enumerate(result['segments']):
            load=sum(float(imported.loc[s['auftrag'],'warengewicht_kg'])+25*float(imported.loc[s['auftrag'],'paletten']) for s in plan['stops'][i:])
            expected=predict_seconds(bundle,segment['osrm_baseline_s'],plan['legs'][i]['distance'],segment['departure_time'],load)
            assert segment['model_forecast_s']==expected
            departure=datetime.fromisoformat(segment['departure_time']);arrival=datetime.fromisoformat(segment['arrival_time_forecast'])
            assert (arrival-departure).total_seconds()==expected
            if prior is not None: assert departure==prior
            prior=arrival+timedelta(seconds=segment['planned_service_s'])
            checks.append(dict(tour_id=plan['tour_id'],segment=i,remaining_load_kg=load,predicted_s=expected,return_to_depot=i==len(plan['stops'])))
    summary=dict(status='passed',source_dataset_sha256=digest(source/'data/segments.csv'),
        train_tours=len(report['splits']['train']),validation_tours=len(report['splits']['validation']),test_tours=len(plans),
        csv_orders=len(imported),forecast_segments=len(checks),model_id=bundle['report']['model_id'],
        fresh_process=True,heuristic_forbidden=True,generator_forbidden=True,labels_absent_from_forecast_input=True,
        all_model_predictions_match=True,arrival_service_chain_verified=True,return_legs_included=True,
        source='OSRM live snapshots from live_result_20; no new service calls',
        limitation='Technical integration PoC, synthetic targets. No real truck accuracy, TomTom live test or browser interaction test.')
    (root/'checks.json').write_text(json.dumps(checks,indent=2));(root/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source');p.add_argument('--out');p.add_argument('--worker');a=p.parse_args()
    if a.worker:worker(Path(a.worker))
    elif a.source and a.out:check(Path(a.source),Path(a.out))
    else:p.error('Provide --source and --out')
