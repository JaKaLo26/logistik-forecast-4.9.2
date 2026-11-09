"""Offline synthetic PoC training, separate from application startup."""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import joblib, sklearn
from src.synthetic_training import generate_training_data
from src.ml_training import train_experiment, load_forecast_model, features, FEATURES
from sklearn.metrics import mean_absolute_error
ROOT=Path(__file__).resolve().parents[1]
orders=pd.read_csv(ROOT/'data/training_stops_de.csv',dtype={'plz':str,'hausnummer':str})
frame,generator=generate_training_data(orders,seed=42,tour_count=1000)
bundle,metrics,details=train_experiment(frame)
report=bundle['report']
report.update(generator=generator,script_version='2.1',sklearn_version=sklearn.__version__,
              source='Fully synthetic offline scenario; addresses not geocoded; no external routing data')
report['state_counts']={state: {'tours': int(frame[frame.bundesland==state].tour_id.nunique()),
                                'stops': int((frame.bundesland==state).sum())}
                        for state in sorted(frame.bundesland.unique())}
# The test set remains untouched during training. Summaries show whether
# errors vary by state, but do not establish real geographic accuracy.
state_lookup=frame.drop_duplicates('tour_id').set_index('tour_id').bundesland
details['bundesland']=details.tour_id.map(state_lookup)
details['abs_error_min']=(details.ML_min-details.Synthetische_Zielzeit_min).abs()
report['test_mae_by_state_min']={state: float(group.abs_error_min.mean())
                                  for state,group in details.groupby('bundesland')}
report['test_mae_by_distance_band_min']={}
test_dist=frame.set_index(['tour_id','segment_id']).distance_m
details['distance_m']=[test_dist.loc[(row.tour_id,row.segment_id)] for row in details.itertuples()]
for name,group in details.groupby(pd.cut(details.distance_m,[0,10000,25000],labels=['0-10 km','10-25 km']),observed=True):
    report['test_mae_by_distance_band_min'][str(name)]=float(group.abs_error_min.mean())
training=frame[frame.tour_id.isin(report['train_tours'])]
train_x=pd.DataFrame([features(row.osrm_baseline_s,row.distance_m,row.planned_departure_at)
                      for row in training.itertuples()],columns=FEATURES)
train_target=(pd.to_datetime(training.actual_arrival_at,utc=True)-pd.to_datetime(training.actual_departure_at,utc=True)).dt.total_seconds()
report['training_mae_min']=float(mean_absolute_error(train_target,bundle['model'].predict(train_x))/60)
report['test_minus_training_mae_min']=float(report['metrics'][1]['MAE_min']-report['training_mae_min'])

# A separate diagnostic keeps two entire states out of training. Its model is
# never deployed; the production artifact above uses all 16 states.
held_out=['Sachsen','Schleswig-Holstein']
regional_train=frame[~frame.bundesland.isin(held_out)]
regional_test=frame[frame.bundesland.isin(held_out)]
regional_bundle,_,_=train_experiment(regional_train)
regional_x=pd.DataFrame([features(row.osrm_baseline_s,row.distance_m,row.planned_departure_at)
                         for row in regional_test.itertuples()],columns=FEATURES)
regional_target=(pd.to_datetime(regional_test.actual_arrival_at,utc=True)-pd.to_datetime(regional_test.actual_departure_at,utc=True)).dt.total_seconds()
report['unseen_state_diagnostic']={
    'states': held_out, 'rows': len(regional_test),
    'model_mae_min': float(mean_absolute_error(regional_target,regional_bundle['model'].predict(regional_x))/60),
    'baseline_mae_min': float(mean_absolute_error(regional_target,regional_test.osrm_baseline_s)/60),
    'scope': 'Separate synthetic-only model; not the bundled model; state is not a model feature.'}
report['interpretation']='Regional balance and held-out synthetic tours do not prove geographic generalization: distances and labels are generated without road or traffic data.'
folder=ROOT/'models';folder.mkdir(exist_ok=True)
path=folder/'builtin_forecast.joblib';joblib.dump(bundle,path,compress=3)
restored=load_forecast_model(path)
(folder/'builtin_report.json').write_text(json.dumps(restored['report'],indent=2,ensure_ascii=False))
orders.to_csv(folder/'synthetic_orders.csv',index=False)
frame.to_csv(folder/'synthetic_segments.csv',index=False)
details.to_csv(folder/'test_predictions.csv',index=False)
print(path.stat().st_size)
