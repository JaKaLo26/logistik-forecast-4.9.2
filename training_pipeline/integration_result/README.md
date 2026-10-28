# Technischer End-to-End-PoC: Training → Modell → CSV → Forecast

Prüfung am 28.10.2026 erfolgreich. Grundlage: 20 synthetische Touren mit realen Stuttgarter Adressen und am 28.10.2026 live abgefragten OSRM-Routen aus `../live_result_20`. Keine erneuten externen API-Aufrufe für diesen Integrationstest.

1. Modell aus `live_result_20/data/segments.csv` neu trainiert: 14 Trainingstouren, 3 Validierungstouren. Modellwahl nur auf Validierung.
2. Als `model/model.joblib` gespeichert, in neuem Python-Prozess über `src.ml_training.load_forecast_model` geladen.
3. Sechs Aufträge der drei zurückgehaltenen Testtouren über `forecast_orders.csv` im bestehenden Acht-Spalten-Format eingelesen. `forecast_plan.json` ergänzt die getrennt benötigten Koordinaten, Straßenabschnitte und Abfahrtszeiten; enthält keine Zielwerte.
4. Der tatsächliche `src.forecast.calculate_tour_forecast` erzeugt neun Segmentprognosen einschließlich drei Rückfahrten. Alle Werte stimmen mit direkter Inferenz des geladenen Modells überein. Ankunft + geplante Servicezeit = nächste Abfahrt.
5. Heuristik, Generator und externe HTTP-Aufrufe sind während dieses Test-Forecasts gesperrt. Jede Ausgabe trägt eine aus der Modelldatei abgeleitete Modellkennung.
6. Tatsächliche Gradio-Callbacks `read_csv`, `activate_pipeline_model` und `calculate_all_forecasts` erfolgreich mit demselben Modell geprüft; neun Prognosen. Logging ausschließlich in temporäre lokale Testablage, kein HF-Upload.

Nachweise: `summary.json`, `checks.json`, `app_callback_check.json`, `forecasts.json` und `model/report.json`.
Zusätzlich 16 Unit-Tests bestanden: 7 Pipeline-, 5 bestehende IHK- und 4 neue Integrations-/Fehlerfalltests.

## Wiederholung

Vom Repository-Verzeichnis, nach Installation von `requirements.txt` und `training_pipeline/requirements.txt`:

```bash
python -m training_pipeline.integration_check --source training_pipeline/live_result_20 --out neuer_integrationstest
python -m training_pipeline.forecast_csv --model neuer_integrationstest/model/model.joblib --orders neuer_integrationstest/forecast_orders.csv --plan neuer_integrationstest/forecast_plan.json --out weitere_forecasts.json
python -m training_pipeline.check_app_bridge
```

Der letzte Befehl prüft die mitgelieferten Dateien in `training_pipeline/integration_result`.

## Anwendung

Forecast-App starten: `python app.py`. Das selbst erzeugte vertrauenswürdige `model.joblib` oben auswählen und „Modell für Forecast und Reforecast laden“ klicken. Danach Auftrags-CSV importieren, Adressen bestätigen, Depot/Flotte konfigurieren, Routen bilden und Forecast berechnen. Für das Pipeline-Modell ist kein TomTom-Schlüssel erforderlich; gültige OSRM-Segmentzeiten und -distanzen bleiben erforderlich. Ein fehlerhaftes Modell/fehlende Merkmale führen zum Fehler statt zum unbemerkten Heuristik-Ersatz.

Die Trainingspipeline bleibt eigenständig. Die Forecast-App erhält lediglich den Modell-Loader und die Übergabe der fünf Merkmale. Alles bleibt im Branch `training-pipeline`, kein Merge in `main`.

## Aussagegrenzen und Datenherkunft

Nachweis der technischen Integration, keine echte LKW-Genauigkeit. Zielzeiten synthetisch. Gleiches Straßennetz kann in verschiedenen Touren vorkommen. Rückfahrten sind zusätzlich funktional geprüft, haben in diesem Datensatz keine synthetischen Bewertungslabels. Beim Forecast werden spätere Abfahrten aus Modellankunft und Service berechnet; beim Generator aus Planzeiten. Dies ist ein Anwendungstest, kein Gleichheitsnachweis zu isolierten Testsegment-Scores.

OSRM driving berücksichtigt keine geprüften LKW-Beschränkungen. Kein TomTom-Livetest, keine visuelle Browserprüfung, kein Deployment.
Quellen, Abrufzeiten, Lizenz-/Nutzungslinks und unveränderte Antworten: `../live_result_20/source.json`, `../live_result_20/geocode-*.json` und `../live_result_20/data/routing_snapshots/`.
© OpenStreetMap-Mitwirkende, ODbL 1.0: https://www.openstreetmap.org/copyright . OSRM-Bedingungen: https://github.com/Project-OSRM/osrm-backend/wiki/Api-usage-policy .
Änderung: Integrationsadapter Version 1; bestehendes Modell-/Feature-Schema 1.0.0 bleibt erhalten. Restladung = noch auszulieferndes Warengewicht + 25 kg je Palette. Auftrags-IDs und bestehende CSV-Spalten unverändert; Testauswahl anhand des Split-Manifests. Test-Planaufbereitung ergänzt die Rückfahrt aus vorhandenem OSRM-Snapshot. Umfang: 6 Aufträge, 3 Touren, 9 Prognosen. Quellhash und Modellhash stehen im Prüfbericht.
