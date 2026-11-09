"""Reproducible, tour-disjoint temporal evaluation. Run python -m src.ml_training --help."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml_features import INPUTS, SCHEMA_VERSION, features


def prepare(frame):
    required = INPUTS + ['tour_id', 'segment_id', 'departure_time', 'actual_arrival',
                         'actual_travel_s', 'prediction_recorded_at', 'label_source']
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    data = frame.copy()
    if data[['tour_id', 'segment_id']].isna().any().any():
        raise ValueError('Tour and segment identifiers must not be missing.')
    if data.duplicated(['tour_id', 'segment_id']).any():
        raise ValueError('Duplicate tour/segment: select one pre-departure forecast snapshot first.')
    if not data.label_source.eq('measured').all():
        raise ValueError('Only measured IST labels are accepted; no synthetic or provider estimates.')
    for col in ['departure_time', 'actual_arrival', 'prediction_recorded_at']:
        data[col] = pd.to_datetime(data[col], utc=True, errors='coerce')
    for col in INPUTS + ['actual_travel_s']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    valid = np.isfinite(data[INPUTS + ['actual_travel_s']]).all(axis=1)
    valid &= (data[INPUTS[:-1]] >= 0).all(axis=1)
    valid &= data.osrm_baseline_s.gt(0) & data.actual_travel_s.gt(0)
    valid &= data.data_confidence.between(0, 1)
    valid &= data.actual_arrival.gt(data.departure_time)
    valid &= data.prediction_recorded_at.le(data.departure_time)
    rejected = data.loc[~valid].copy()
    data = data.loc[valid].sort_values(['departure_time', 'tour_id', 'segment_id']).reset_index(drop=True)
    if data.tour_id.nunique() < 12:
        raise ValueError('At least 12 valid independent tours are required for temporal evaluation.')
    return data, rejected


def tour_order(data):
    return data.groupby('tour_id').departure_time.min().sort_values(kind='stable').index.to_numpy()


def purge(data, earlier, later):
    """All labels of an earlier tour must be available before the later period starts."""
    cutoff = data.loc[later, 'departure_time'].min()
    ends = data.groupby('tour_id').actual_arrival.max()
    allowed = ends[ends < cutoff].index
    return np.asarray([i for i in earlier if data.loc[i, 'tour_id'] in allowed], dtype=int)


def split(data):
    tours = tour_order(data)
    n = len(tours)
    a, b = int(n * .7), int(n * .85)
    parts = [data.index[data.tour_id.isin(ids)].to_numpy() for ids in
             [tours[:a], tours[a:b], tours[b:]]]
    parts[0] = purge(data, parts[0], parts[1])
    parts[1] = purge(data, parts[1], parts[2])
    if any(len(p) == 0 for p in parts):
        raise ValueError('Overlapping tours leave an empty temporal partition; collect more history.')
    return parts


def folds(data):
    tours = tour_order(data)
    if len(tours) < 6:
        raise ValueError('At least six training tours remain necessary after overlap purging.')
    result = []
    for a, b in TimeSeriesSplit(n_splits=3).split(tours):
        train = data.index[data.tour_id.isin(tours[a])].to_numpy()
        valid = data.index[data.tour_id.isin(tours[b])].to_numpy()
        train = purge(data, train, valid)
        if not len(train):
            raise ValueError('Insufficient non-overlapping history for backtesting.')
        result.append((train, valid))
    return result


def metrics(actual, predicted):
    y, p = np.asarray(actual, float), np.maximum(0, np.asarray(predicted, float))
    error = p - y
    mask = np.abs(y) >= 1  # avoid undefined or unstable percentages below one second
    return {'mae_s': float(np.abs(error).mean()), 'rmse_s': float(np.sqrt((error ** 2).mean())),
            'mape_pct': float((np.abs(error[mask] / y[mask])).mean() * 100) if mask.any() else None,
            'mape_n': int(mask.sum()), 'n': len(y)}


def train(frame):
    data, rejected = prepare(frame)
    tr, va, te = split(data)
    training = data.loc[tr].reset_index(drop=True)
    cv = folds(training)
    x, y = features(training), training.actual_travel_s
    # Statistical extremes are flagged, never deleted: real congestion remains learnable.
    ratio = y / training.osrm_baseline_s
    q1, q3 = ratio.quantile([.25, .75])
    bounds = [float(q1 - 1.5 * (q3-q1)), float(q3 + 1.5 * (q3-q1))]
    specs = {
        'ridge': (make_pipeline(StandardScaler(), Ridge()), {'ridge__alpha': [.1, 10, 100]}),
        'forest': (RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=1),
                   {'max_depth': [5, None], 'min_samples_leaf': [2, 5]}),
        # Explicit iteration selection on temporal folds avoids random early-stopping splits.
        'boosting': (GradientBoostingRegressor(random_state=42, loss='huber'),
                     {'n_estimators': [50, 100], 'learning_rate': [.03, .1], 'max_depth': [2]}),
    }
    candidates, report = {}, {}
    for name, (model, grid) in specs.items():
        search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=1,
                              error_score='raise').fit(x, y)
        candidates[name] = search.best_estimator_
        report[name] = {'parameters': search.best_params_, 'cv_mae_s': float(-search.best_score_),
                       'backtest_mae_s': [float(-search.cv_results_[f'split{i}_test_score'][search.best_index_])
                                         for i in range(len(cv))]}
    candidates['ensemble'] = VotingRegressor([(n, clone(m)) for n, m in candidates.items()]).fit(x, y)
    report['ensemble'] = {}
    for name, model in candidates.items():
        report[name]['validation'] = metrics(data.loc[va, 'actual_travel_s'], model.predict(features(data.loc[va])))
    # OSRM participates in selection so a worse ML model is not automatically activated.
    baseline_validation = metrics(data.loc[va, 'actual_travel_s'], data.loc[va, 'osrm_baseline_s'])
    winner = min(report, key=lambda name: report[name]['validation']['mae_s'])
    selected = winner if report[winner]['validation']['mae_s'] < baseline_validation['mae_s'] else 'osrm'
    # Refit only after validation selection; the test set is evaluated once, never used for tuning.
    fit_idx = np.concatenate([tr, va])
    model = clone(candidates[winner]).fit(features(data.loc[fit_idx]), data.loc[fit_idx, 'actual_travel_s']) if selected != 'osrm' else None
    test = data.loc[te]
    prediction = model.predict(features(test)) if model is not None else test.osrm_baseline_s.to_numpy()
    data['partition'] = 'purged_overlap'
    for name, indices in zip(['train', 'validation', 'test'], [tr, va, te]):
        data.loc[indices, 'partition'] = name
    data['outlier_flag'] = ~((data.actual_travel_s / data.osrm_baseline_s).between(*bounds))
    result = {'selected': selected, 'candidates': report, 'baseline_validation': baseline_validation,
              'test': metrics(test.actual_travel_s, prediction),
              'baseline_test': metrics(test.actual_travel_s, test.osrm_baseline_s),
              'split_rows': data.partition.value_counts().to_dict(), 'rejected_rows': len(rejected),
              'outlier_ratio_bounds_train_only': bounds, 'outlier_rows_retained': int(data.outlier_flag.sum()),
              'sklearn_version': sklearn.__version__, 'schema_version': SCHEMA_VERSION,
              'backtesting': '3 expanding tour windows within training; overlapping label intervals purged'}
    predictions = test[['tour_id', 'segment_id', 'actual_travel_s', 'osrm_baseline_s']].copy()
    predictions['prediction_s'] = np.maximum(0, prediction)
    return {'model': model, 'schema_version': SCHEMA_VERSION}, result, data, rejected, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', required=True, help='Source, author/version, rights/license and preparation reference')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory to preserve previous runs.')
    bundle, report, assignments, rejected, predictions = train(pd.read_csv(args.csv))
    report.update(source=args.source, input_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest())
    args.output.mkdir(parents=True)
    joblib.dump(bundle, args.output / 'model.joblib')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    assignments.to_csv(args.output / 'assignments.csv', index=False)
    rejected.to_csv(args.output / 'rejected.csv', index=False)
    predictions.to_csv(args.output / 'test_predictions.csv', index=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
