"""One feature contract for offline training and online inference (seconds)."""
import numpy as np
import pandas as pd

INPUTS = ['osrm_baseline_s', 'tomtom_historic_s', 'tomtom_live_s',
          'tomtom_traffic_delay_s', 'data_confidence']
SCHEMA_VERSION = 1


def features(frame):
    x = frame[INPUTS].astype(float).copy()
    base = x.osrm_baseline_s.clip(lower=1)
    for name in ['tomtom_historic_s', 'tomtom_live_s']:
        x[name + '_missing'] = (x[name] <= 0).astype(float)
        x[name + '_ratio'] = x[name] / base
    x['live_minus_historic_s'] = x.tomtom_live_s - x.tomtom_historic_s
    x['log_baseline'] = np.log1p(base)
    return x
