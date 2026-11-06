import joblib
import numpy as np
import pandas as pd
import pytest

from src.ml_features import INPUTS, features
from src.ml_predict import predict_if_configured
from src.ml_training import prepare, split, folds, metrics, train


def fixture():
    rows = []
    for tour in range(24):
        for segment in range(2):
            departure = pd.Timestamp('2026-01-01', tz='UTC') + pd.Timedelta(days=tour, hours=segment)
            base = 300 + tour * 13 + segment * 50
            rows.append(dict(tour_id=f't{tour:02d}', segment_id=segment,
                departure_time=departure.isoformat(),
                prediction_recorded_at=(departure - pd.Timedelta(minutes=5)).isoformat(),
                actual_arrival=(departure + pd.Timedelta(seconds=base)).isoformat(),
                actual_travel_s=base, label_source='measured', osrm_baseline_s=base,
                tomtom_historic_s=base+10, tomtom_live_s=base+20,
                tomtom_traffic_delay_s=20, data_confidence=1))
    return pd.DataFrame(rows)


def assert_boundary(data, a, b):
    assert set(data.loc[a, 'tour_id']).isdisjoint(data.loc[b, 'tour_id'])
    assert data.loc[a, 'actual_arrival'].max() < data.loc[b, 'departure_time'].min()


def test_temporal_tour_partitions_and_cv():
    data, _ = prepare(fixture())
    a, b, c = split(data)
    assert_boundary(data, a, b)
    assert_boundary(data, b, c)
    training = data.loc[a].reset_index(drop=True)
    for earlier, later in folds(training):
        assert_boundary(training, earlier, later)


def test_overlap_purges_whole_tour():
    data, _ = prepare(fixture())
    data.loc[data.tour_id.eq('t15'), 'actual_arrival'] = pd.Timestamp('2026-01-20', tz='UTC')
    a, b, _ = split(data)
    assert 't15' not in set(data.loc[a, 'tour_id'])
    assert_boundary(data, a, b)


def test_rejects_duplicates_and_unmeasured_labels():
    frame = fixture()
    with pytest.raises(ValueError, match='Duplicate'):
        prepare(pd.concat([frame, frame.iloc[:1]]))
    frame.loc[0, 'label_source'] = 'synthetic'
    with pytest.raises(ValueError, match='measured'):
        prepare(frame)


def test_invalid_and_future_information_quarantined():
    frame = fixture()
    frame['actual_travel_s'] = frame.actual_travel_s.astype(float)
    frame.loc[0, 'actual_travel_s'] = np.inf
    frame.loc[1, 'prediction_recorded_at'] = '2027-01-01T00:00:00Z'
    frame.loc[2, 'data_confidence'] = 2
    _, rejected = prepare(frame)
    assert len(rejected) == 3


def test_metrics_zero_and_seconds():
    result = metrics([0, 10], [2, 8])
    assert result['mae_s'] == result['rmse_s'] == 2
    assert result['mape_pct'] == 20
    assert result['mape_n'] == 1
    assert metrics([0], [0])['mape_pct'] is None


def test_baseline_can_win_and_artifact_is_usable(tmp_path, monkeypatch):
    bundle, report, assigned, rejected, predictions = train(fixture())
    assert report['selected'] == 'osrm'
    assert report['test']['mae_s'] == 0
    assert report['test']['n'] == len(predictions)
    assert set(assigned.partition) == {'train', 'validation', 'test'}
    assert rejected.empty
    artifact = tmp_path / 'model.joblib'
    joblib.dump(bundle, artifact)
    monkeypatch.setenv('FORECAST_MODEL_PATH', str(artifact))
    assert predict_if_configured(600, 650, 700, 20, 1) == 600


def test_inference_uses_training_features(tmp_path, monkeypatch):
    from sklearn.linear_model import LinearRegression
    frame = fixture()
    model = LinearRegression().fit(features(frame), frame.osrm_baseline_s * 1.2)
    artifact = tmp_path / 'model.joblib'
    joblib.dump({'schema_version': 1, 'model': model}, artifact)
    monkeypatch.setenv('FORECAST_MODEL_PATH', str(artifact))
    row = frame.iloc[8]
    expected = int(round(model.predict(features(frame.iloc[[8]]))[0]))
    assert predict_if_configured(*row[INPUTS].tolist()) == expected


def test_missing_or_incompatible_artifact_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv('FORECAST_MODEL_PATH', raising=False)
    assert predict_if_configured(600, 650, 700, 20, 1) is None
    monkeypatch.setenv('FORECAST_MODEL_PATH', str(tmp_path / 'missing.joblib'))
    assert predict_if_configured(600, 650, 700, 20, 1) is None
    path = tmp_path / 'bad.joblib'
    joblib.dump({'schema_version': -1, 'model': None}, path)
    monkeypatch.setenv('FORECAST_MODEL_PATH', str(path))
    assert predict_if_configured(600, 650, 700, 20, 1) is None


def test_forecast_existing_behavior_and_activation(tmp_path, monkeypatch):
    from src.forecast import calculate_model_forecast_s
    monkeypatch.delenv('FORECAST_MODEL_PATH', raising=False)
    assert calculate_model_forecast_s(100, 200, 300, 20, 1) == 260
    path = tmp_path / 'baseline.joblib'
    joblib.dump({'schema_version': 1, 'model': None}, path)
    monkeypatch.setenv('FORECAST_MODEL_PATH', str(path))
    assert calculate_model_forecast_s(100, 200, 300, 20, 1) == 100


def test_holdout_labels_do_not_change_selection_or_outlier_bounds():
    frame = fixture()
    frame['actual_travel_s'] = frame.osrm_baseline_s * 1.2
    _, first, _, _, _ = train(frame)
    assert first['selected'] != 'osrm'
    frame.loc[frame.tour_id.ge('t20'), 'actual_travel_s'] *= 10
    _, second, _, _, _ = train(frame)
    assert first['selected'] == second['selected']
    assert first['candidates'] == second['candidates']
    assert first['outlier_ratio_bounds_train_only'] == second['outlier_ratio_bounds_train_only']
    assert second['test']['mae_s'] > first['test']['mae_s']
