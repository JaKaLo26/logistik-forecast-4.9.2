"""Offline PoC training, separate from application startup. Version 1.0."""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import joblib, sklearn
from src.synthetic_training import generate_training_data
from src.ml_training import train_experiment, load_forecast_model
ROOT=Path(__file__).resolve().parents[1]
orders=pd.DataFrame([dict(auftrag=f'DEMO-{i}',kunde=f'Synthetischer Kunde {i}',strasse=f'Demoweg {i}',plz='00000',ort='Synthetischer Ort',paletten=2,warengewicht_kg=700,service_min=10) for i in range(1,6)])
frame,generator=generate_training_data(orders,seed=42,tour_count=100)
bundle,metrics,details=train_experiment(frame)
bundle['report'].update(generator=generator,script_version='1.0',sklearn_version=sklearn.__version__,source='Fully synthetic offline scenario; no external routing data')
folder=ROOT/'models';folder.mkdir(exist_ok=True)
path=folder/'builtin_forecast.joblib';joblib.dump(bundle,path,compress=3)
restored=load_forecast_model(path)
(folder/'builtin_report.json').write_text(json.dumps(restored['report'],indent=2,ensure_ascii=False))
orders.to_csv(folder/'synthetic_orders.csv',index=False)
frame.to_csv(folder/'synthetic_segments.csv',index=False)
details.to_csv(folder/'test_predictions.csv',index=False)
print(path.stat().st_size)
