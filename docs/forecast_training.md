# Forecast training and evaluation

The existing forecast is a weighted OSRM/TomTom heuristic, not a trained neural
network. This change adds an opt-in supervised regression pipeline. No dataset
is downloaded, licensed implicitly, or trained at application startup.

## Input contract

One CSV row represents one pre-departure prediction snapshot and its **measured**
segment travel time. Required columns:

| Column | Meaning |
| --- | --- |
| tour_id, segment_id | Unique physical tour/segment; use a new tour ID on each date |
| departure_time | Actual departure, ISO timestamp with timezone |
| actual_arrival | Actual arrival; used to purge overlapping outcome periods |
| prediction_recorded_at | When the input snapshot was captured; no later than departure |
| actual_travel_s | Measured travel seconds, positive; exclude unloading/service time |
| label_source | `measured`; synthetic/provider-derived targets are rejected |
| osrm_baseline_s | Positive planned OSRM seconds |
| tomtom_historic_s, tomtom_live_s | Snapshot provider seconds; zero means unavailable |
| tomtom_traffic_delay_s | Nonnegative snapshot traffic delay seconds |
| data_confidence | Snapshot confidence, 0 through 1 |

Keep raw logs unchanged. To assemble the CSV, join the existing forecast segment
and actual segment logs on physical tour/segment IDs, verifying stop endpoints
and route version. Select exactly one snapshot captured **before** actual departure;
map its `logged_at` to `prediction_recorded_at` and the actual log's
`actual_departure` to `departure_time`. Forecast logs alone are not measured labels.
Do not join re-routed segments solely by their sequence number. The CLI deliberately
rejects duplicate IDs rather than silently picking an ambiguous version.

Confirm data provenance and storage/training rights before preparing the CSV.
The source argument records the dataset, author/version, license/permission and
preparation notes or a reference to them; it is not automatic license verification.
No Uni Wuppertal data is included or presumed available in this repository.

## Run

```bash
pip install -r requirements.txt
python -m src.ml_training measured_segments.csv --output training_runs/run_001 --source "Dataset; author; version; license/permission; preparation reference"
```

At least 12 valid tours are required (a software minimum, not statistical proof).
Tour start order defines approximately 70/15/15 train/validation/test partitions.
Tours remain intact; earlier tours whose labels overlap the next period are
excluded. Three expanding temporal folds inside training repeat this purging.
Coincident/overlapping histories may require more tours and fail explicitly.
Scaling is fitted inside each fold. Test data is never used for model selection.

Features use only the five online inputs: provider durations, delay and confidence,
plus missing-provider flags, ratios, live/historic difference and log baseline.
Invalid numeric/timestamp rows go to `rejected.csv`. Extreme actual/baseline ratios
are flagged using training-only IQR bounds but **retained**: real congestion must
not be removed simply because it is rare. No target clipping or test-set cleaning.

Grid search compares Ridge, Random Forest and robust Gradient Boosting using MAE.
A voting ensemble is also compared on validation. OSRM participates in validation
selection; if it wins, the artifact predicts OSRM directly. The selected ML model
is refitted on train plus validation and evaluated once on the held-out test set.
`report.json` includes MAE/RMSE in seconds, MAPE in percent (excluding targets below
one second), counts, best parameters, training CV fold errors, validation scores,
test and OSRM test scores, source, input SHA-256 and sklearn version.
`assignments.csv` records partitions, purged rows and outlier flags;
`test_predictions.csv` preserves segment-level errors for inspection.
Output directories must be new, preserving previous experiment evidence.

## Activate explicitly

```bash
export FORECAST_MODEL_PATH=training_runs/run_001/model.joblib
python app.py
```

Use the same dependency versions as training. Joblib uses pickle: load only your
own trusted artifacts. A missing/unreadable/incompatible artifact emits a warning
and falls back to the existing heuristic; without the variable behavior is unchanged.
An artifact selected as `osrm` returns the baseline rather than the heuristic.
Changing the file reloads it by modification time. The feature transformation is
shared between training and inference. Validation selection does not guarantee
better accuracy on new routes: inspect the held-out test report before activation.

Early stopping/patience and learning-rate scheduling are not added to the current
non-neural application. Boosting iteration count and learning rate are selected
through temporal CV, avoiding sklearn's random internal early-stopping holdout.
Gradient clipping applies to neural training, which is absent here. There is no
RNN/LSTM introduced merely to enable these techniques.

## Checks

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use generated fixtures only to verify mechanics; their scores are not evidence
of real-world forecast quality. Real accuracy requires independently measured tours.

## Datenfluss und Referenzwerte

Alle internen Fahrzeiten bleiben in Sekunden. Tabellen wandeln erst bei der
Anzeige in Minuten um. Ankunft, nächste Abfahrt und Tourensumme verwenden denselben
Segment-Forecast; Servicezeiten werden separat addiert.

Fehlende oder falsch zugeordnete OSRM-Legs führen zu einer verständlichen
Fehlermeldung und erfordern eine neue Routenberechnung. Die gesamte Tourdauer wird
nicht mehr gleichmäßig auf Stopps verteilt. OSRM-Trip-Waypoints müssen eine
vollständige, eindeutige Rundreise ab Depot bilden.

Fehlende TomTom-Felder bleiben als Null-Sentinel markiert. OSRM und generische
TomTom-Reisezeiten werden nicht als beobachtete Live-/Historienwerte ausgegeben.
Ein Tour-Livevergleich ist nur bei vollständigen Livewerten aller Segmente verfügbar;
Teilwerte bleiben in der Segmenttabelle sichtbar. Live ist eine Prognose, keine Ist-Zeit.
Verkehrsverzögerungen werden nicht zusätzlich auf bereits verkehrsabhängige Zeiten addiert.

Modellprognosen unter der Hälfte der kleinsten positiven Referenzzeit oder über
dem Doppelten der größten Referenzzeit werden durch die Referenzgewichtung ersetzt
und sichtbar markiert. Diese breite technische Plausibilitätsregel ist kein
statistisch kalibriertes Konfidenzintervall. Quelle und Warnung werden protokolliert.
Regression: OSRM 262 min, historisch/live jeweils 341 min ergeben bei Confidence 1
333,1 min; ein injizierter Modellwert von 40 min wird abgewiesen.

Die Screenshot-Oberfläche mit „KI-Fahrzeit … Fahrerbudget“ und „TTLive= Ist-Zeit“
ist im untersuchten Repository nicht vorhanden. Ein Fix dieser konkreten laufenden
Oberfläche kann erst mit ihrer tatsächlichen Code-/Deployment-Version verifiziert werden.
