"""Bundled synthetic PoC model, loaded without training at app startup."""
from pathlib import Path
from functools import lru_cache
from src.ml_training import load_forecast_model, features, FEATURES, guard_prediction
import numpy as np
import pandas as pd

@lru_cache(maxsize=1)
def load_builtin_model():
    return load_forecast_model(Path(__file__).resolve().parents[1]/'models/builtin_forecast.joblib')

def predict_road_matrix(baselines_min, distances_m, departure, bundle):
    matrix=np.asarray(baselines_min,dtype=float).copy()
    distances=np.asarray(distances_m,dtype=float)
    if matrix.shape != distances.shape:
        raise ValueError('Unvollständige Streckenmatrix.')
    indices=[]; rows=[]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i==j or not np.isfinite(matrix[i,j]):
                continue
            if matrix[i,j]==0 and distances[i,j]==0:
                continue
            rows.append(features(matrix[i,j]*60,distances[i,j],departure))
            indices.append((i,j))
    if rows:
        predictions=np.asarray(bundle['model'].predict(pd.DataFrame(rows,columns=FEATURES)),dtype=float)
        if predictions.shape != (len(rows),):
            raise ValueError('Ungültige Anzahl KI-Fahrzeiten.')
        if not (np.isfinite(predictions)&(predictions>0)).all():
            raise ValueError('Ungültige KI-Fahrzeiten.')
        for (i,j), value in zip(indices,predictions):
            warning = guard_prediction(bundle, matrix[i,j]*60, distances[i,j], value)
            if not warning:
                matrix[i,j]=value/60
            # Otherwise retain the matching OSRM cell, in minutes.

    return matrix.tolist()
