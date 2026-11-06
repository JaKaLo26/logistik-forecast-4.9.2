"""Opt-in loading of a trusted, locally trained artifact; safe heuristic fallback."""
from functools import lru_cache
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _load(path, modified):
    import joblib
    from src.ml_features import SCHEMA_VERSION
    bundle = joblib.load(path)  # Only load artifacts you created/trust, never arbitrary uploads.
    if bundle.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported forecast feature schema')
    return bundle


_warned = set()


def predict_if_configured(osrm, historic, live, delay, confidence):
    path = os.environ.get('FORECAST_MODEL_PATH')
    if not path:
        return None
    try:
        import numpy as np
        import pandas as pd
        from src.ml_features import INPUTS, features
        values = [osrm, historic, live, delay, confidence]
        if not np.isfinite(values).all() or osrm <= 0:
            return None
        artifact = Path(path).resolve()
        bundle = _load(str(artifact), artifact.stat().st_mtime_ns)
        model = bundle['model']
        value = osrm if model is None else model.predict(features(pd.DataFrame([values], columns=INPUTS)))[0]
        if not np.isfinite(value):
            raise ValueError('Non-finite model prediction')
        return max(0, int(round(value)))
    except Exception as exc:
        key = (path, type(exc).__name__, str(exc))
        if key not in _warned:
            log.warning('ML forecast unavailable; using heuristic: %s', exc)
            _warned.add(key)
        return None
