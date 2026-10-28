"""IHK experiment: immutable feature snapshots, tour holdout and measured labels."""
from __future__ import annotations
import hashlib
import json
import math
import uuid
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = ['osrm_baseline_s', 'distance_m', 'departure_hour', 'weekday']
REQUIRED = ['tour_id', 'segment_id', 'prediction_created_at', 'planned_departure_at',
            'actual_departure_at', 'actual_arrival_at', 'actual_recorded_at',
            'osrm_baseline_s', 'distance_m', 'data_origin']

def features(baseline, distance, departure):
    dt = datetime.fromisoformat(str(departure).replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Zeitangaben benötigen eine Zeitzone, z. B. +02:00.')
    if not all(math.isfinite(float(v)) and float(v) > 0 for v in (baseline, distance)):
        raise ValueError('OSRM-Zeit und Distanz müssen positiv und endlich sein.')
    local = pd.Timestamp(dt).tz_convert('Europe/Berlin')
    return [float(baseline), float(distance), local.hour + local.minute / 60, local.weekday()]

def validate_data(df):
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError('Fehlende Spalten: ' + ', '.join(sorted(missing)))
    df = df.copy()
    if df[REQUIRED].isna().any().any():
        raise ValueError('Pflichtfelder enthalten fehlende Werte.')
    if not df.data_origin.isin(['real', 'synthetic', 'simulation']).all() or df.data_origin.nunique() != 1:
        raise ValueError('Genau eine Datenherkunft je Experiment: real, synthetic oder simulation.')
    if df.tour_id.astype(str).str.strip().eq('').any() or df.segment_id.astype(str).str.strip().eq('').any():
        raise ValueError('Tour- und Segment-ID dürfen nicht leer sein.')
    if df.duplicated(['tour_id', 'segment_id']).any():
        raise ValueError('Pro Tour/Segment genau einen ursprünglichen Forecast verwenden; keine Reforecast-Duplikate.')
    for name in ['prediction_created_at', 'planned_departure_at', 'actual_departure_at', 'actual_arrival_at', 'actual_recorded_at']:
        for value in df[name]:
            if datetime.fromisoformat(str(value).replace('Z', '+00:00')).tzinfo is None:
                raise ValueError(f'{name}: Zeitzone fehlt.')
        df[name] = pd.to_datetime(df[name], utc=True)
    if not ((df.prediction_created_at <= df.actual_departure_at) &
            (df.actual_arrival_at > df.actual_departure_at) &
            (df.actual_recorded_at >= df.actual_arrival_at)).all():
        raise ValueError('Ungültige Zeitfolge: Prognose ≤ Abfahrt < Ankunft ≤ IST-Erfassung.')
    for col in ['osrm_baseline_s', 'distance_m']:
        df[col] = pd.to_numeric(df[col], errors='raise')
        if not (np.isfinite(df[col]) & (df[col] > 0)).all():
            raise ValueError(f'{col} muss positiv und endlich sein.')
    df['target_s'] = (df.actual_arrival_at - df.actual_departure_at).dt.total_seconds()
    return df

def train_experiment(df):
    df = validate_data(df)
    tours = df.groupby('tour_id').prediction_created_at.min().sort_values(kind='stable')
    if len(tours) < 5:
        raise ValueError('Mindestens fünf unabhängige Touren erforderlich; für belastbare Ergebnisse deutlich mehr.')
    cut = max(1, int(len(tours) * .8))
    test_ids = set(tours.index[cut:])
    test = df[df.tour_id.isin(test_ids)].copy()
    train = df[~df.tour_id.isin(test_ids)].copy()
    cutoff = test.prediction_created_at.min()
    # Entire tour is purged if any of its labels was unavailable at evaluation cutoff.
    eligible = train.groupby('tour_id').actual_recorded_at.max()
    train = train[train.tour_id.isin(eligible[eligible < cutoff].index)]
    if train.tour_id.nunique() < 2:
        raise ValueError('Zu wenige abgeschlossene Trainingstouren vor dem Testzeitraum.')
    x = lambda frame: pd.DataFrame([features(r.osrm_baseline_s, r.distance_m, r.planned_departure_at.isoformat()) for r in frame.itertuples()], columns=FEATURES)
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, max_depth=10, random_state=42, n_jobs=1)
    model.fit(x(train), train.target_s)
    prediction = model.predict(x(test))
    baseline = test.osrm_baseline_s.to_numpy()
    scores = []
    for name, values in [('OSRM-Planbasis', baseline), ('Random Forest', prediction)]:
        scores.append({'Verfahren': name, 'MAE_min': mean_absolute_error(test.target_s, values)/60,
                       'RMSE_min': math.sqrt(mean_squared_error(test.target_s, values))/60,
                       'Bias_min': float(np.mean(values-test.target_s))/60})
    report = {'model_id': str(uuid.uuid4()), 'features': FEATURES, 'data_origin': str(df.data_origin.iloc[0]),
              'train_tours': sorted(map(str, train.tour_id.unique())), 'test_tours': sorted(map(str, test.tour_id.unique())),
              'train_rows': len(train), 'test_rows': len(test), 'cutoff': cutoff.isoformat(),
              'dataset_sha256': hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
              'metrics': scores, 'parameters': model.get_params(),
              'limitation': 'Ein zeitlicher Holdout, keine Hyperparameterwahl am Test. Segmentfehler sind keine End-to-End-ETA-Bewertung.'}
    details = test[['tour_id','segment_id']].copy()
    details['IST_min'] = test.target_s/60
    details['Plan_min'] = baseline/60
    details['ML_min'] = prediction/60
    return {'model': model, 'report': report}, pd.DataFrame(scores), details

PIPELINE_FEATURES = ['baseline_s', 'distance_m', 'departure_hour', 'weekday', 'remaining_load_kg']

def load_forecast_model(path):
    """Load a trusted pipeline artifact; never load untrusted joblib files."""
    import joblib
    from pathlib import Path
    raw = Path(path).read_bytes()
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get('schema_version') != '1.0.0':
        raise ValueError('Unsupported pipeline model schema')
    if bundle.get('features') != PIPELINE_FEATURES or bundle.get('data_origin') != 'synthetic':
        raise ValueError('Incompatible pipeline feature/origin contract')
    model = bundle.get('model')
    if list(getattr(model, 'feature_names_in_', [])) != PIPELINE_FEATURES:
        raise ValueError('Fitted model features differ from declared schema')
    bundle['report'] = {'model_id': 'pipeline-' + hashlib.sha256(raw).hexdigest(),
                        'features': PIPELINE_FEATURES, 'data_origin': 'synthetic'}
    return bundle

def predict_seconds(bundle, baseline, distance, departure, remaining_load_kg=None):
    if bundle.get('schema_version') is not None:
        if bundle.get('schema_version') != '1.0.0' or bundle.get('features') != PIPELINE_FEATURES:
            raise ValueError('Incompatible pipeline model schema')
        values = features(baseline, distance, departure)
        local = pd.Timestamp(departure).tz_convert('Europe/Berlin')
        values[2] = local.hour + local.minute/60 + local.second/3600 + local.microsecond/3600000000
        if remaining_load_kg is None or not math.isfinite(float(remaining_load_kg)) or float(remaining_load_kg) < 0:
            raise ValueError('Pipeline forecast requires nonnegative remaining load')
        row = pd.DataFrame([values + [float(remaining_load_kg)]], columns=PIPELINE_FEATURES)
        result = float(bundle['model'].predict(row)[0])
        if not math.isfinite(result) or result <= 0:
            raise ValueError('Model produced invalid travel time')
        return max(1, int(round(result)))

    row = pd.DataFrame([features(baseline, distance, departure)], columns=FEATURES)
    return max(1, int(round(bundle['model'].predict(row)[0])))

