# Eigenständiges Trainingslabor · Version 1.0.0

Ziel: technischer Machbarkeitsnachweis vom Tourengenerator bis zum erneut geladenen Modell.
Dieser Ordner bleibt im Branch `training-pipeline`; keine Integration oder Zusammenführung in `main` vorgesehen.
Die bestehende Forecast-App wird nicht geändert. Startpunkt: `Ai-workflow`, Commit c193a7a.

## Start auf dem PC

Python 3.12 empfohlen. Im Repository-Verzeichnis:

```bash
python -m pip install -r training_pipeline/requirements.txt
python -m training_pipeline.ui
```

Browser: http://localhost:7861 . Die eigenständige Oberfläche erzeugt Touren und trainiert nach einem Klick.
Datenpaket und Modellpaket stehen danach zum Download bereit. Es wurde kein zusätzlicher Hugging-Face-Space eingerichtet.
CPU genügt; kein LLM und keine GPU erforderlich.

## Nachvollziehbarer CLI-Durchlauf

```bash
python -m training_pipeline generate --demo --tours 100 --min-stops 3 --max-stops 8 --seed 42 --out pipeline_runs/demo-data
python -m training_pipeline train --data pipeline_runs/demo-data/segments.csv --out pipeline_runs/demo-model
python -m training_pipeline predict --model pipeline_runs/demo-model/model.joblib --data pipeline_runs/demo-model/test.csv --out pipeline_runs/demo-predictions.csv
python -m unittest discover -s tests -p test_training_pipeline.py -v
```

Vorhandene Ausgabeordner werden nicht überschrieben. Für neue Experimente neue Namen wählen.
Die **Offline-Demo enthält erfundene Adressen und synthetische geometrische Strecken**. Diese Auftrags-CSV ist schema-kompatibel,
aber nicht zum Geocodieren bestimmt. Sie ist kein Straßen- oder LKW-Routennachweis.

## Deutschland-Adressen und Straßenrouten

Einen geprüften Adresspool mit `strasse,plz,ort,lat,lon` und einen JSON-Quellennachweis bereitstellen.
Der Generator erfindet hier keine Hausnummern und geocodiert nicht selbst. Pro Tour werden Start und Stopps ohne Wiederholung gezogen.
Der Start ist ein zufälliger Poolpunkt, kein festes Depot. Keine Kapazitätsoptimierung, Rückfahrt oder Fahrerzeitplanung in diesem Datengenerator.
Die separat bestehende Planungsanwendung übernimmt diese Aufgaben. `from_pool_index` / `to_pool_index` dokumentieren die Wege.

Quellennachweis: `author`, `source_url`, `data_version`, `retrieved_at` (ISO mit Zeitzone), `license`, `license_url`, `terms_url`.
Bei unbekanntem Datenstand ausdrücklich `not provided by service` eintragen, keine Version erfinden.

```bash
python -m training_pipeline generate --addresses adresspool.csv --source quelle.json --tours 100 --out pipeline_runs/road-data
```

OSRM Route API v1 / driving liefert Straßenlängen und Basiszeiten. Kein LKW-Profil und keine garantierte Befahrbarkeit.
Bei Dienstausfall oder fehlender Route wird abgebrochen; keine stillen Luftlinien-Ersatzwerte.
Antworten werden mit Abrufdatum, Endpoint, Profil und SHA-Referenz zwischengespeichert.
Erneute Generierung mit gleichem Pool, Parametern, Seed und `--snapshots pipeline_runs/road-data/routing_snapshots`
verwendet dieselben Antworten. Ohne diese Snapshots kann ein aktualisierter Routingdienst andere Daten liefern.
Für größere Datenmengen einen eigenen/geeigneten OSRM-Server über `--osrm-url` verwenden; dessen Nutzungsbedingungen dokumentieren.

## Dateien und Modellvertrag

- `orders.csv`: unveränderte acht Pflichtspalten `auftrag,kunde,strasse,plz,ort,paletten,warengewicht_kg,service_min`.
- `segments.csv`: separate Tour-/Segment-IDs, Feature-Snapshot, Routenreferenz und `synthetic_target_s`; Auftrags-ID = Segment-ID.
- `address_pool.csv`, `routing_snapshots/`, `generation.json`: reproduzierbare Grundlage und Herkunft/Parameter/Hashes.
- `train.csv`, `validation.csv`, `test.csv`, `report.json`: feste Tourzuordnung und Modellvergleich.
- `model.joblib`: Random Forest mit Schema und Feature-Liste; nur vertrauenswürdige eigene Joblib-Dateien laden.
- `test_predictions.csv`: Vorhersagen des gespeicherten, erneut geladenen Modells mit synthetischen Vergleichswerten.

Die Inferenz benötigt ausschließlich `baseline_s,distance_m,departure_hour,weekday,remaining_load_kg`.
Sie ruft weder Generator noch Zielzeitformel auf. Das Artefakt hat einen eigenen versionierten Vertrag;
es ist kein ungeprüfter Ersatz für das bisherige `src.ml_training`-Bundle. Die Forecast-App importiert es noch nicht.
Andere Eingabedaten lassen sich über dasselbe `segments.csv`-Schema trainieren; reale Labels benötigen einen gesonderten Adapter und Validierungsregeln.

## Methodik und Grenzen

Default 100 Touren; 3–8 Stopps, 1–3 Paletten/Stopp, 350 kg Ware/Palette, 25 kg Palettenleergewicht, 5–20 Minuten Service.
Die Python-Konfiguration `Config` erlaubt diese Bereiche zu ändern. Alle Parameter landen im Manifest.
Synthetisches Ziel: Basiszeit × (1,08 + Rushhourfaktor + Restladung/200000 + Normalrauschen mit σ=0,06), mindestens 1 Sekunde.
Rushhour: 7–9 bzw. 16–18 Uhr werktags, Faktor 0,25. Dies sind Szenarioannahmen, keine empirisch belegten Verkehrseffekte.
Folgesegment-Eingaben nutzen geplante Basis-/Servicezeiten; synthetische Zielzeiten fließen nicht in Eingaben ein.

Vollständige Tourgruppen werden seed-basiert ungefähr 70/15/15 aufgeteilt. Exakte Kopien der Feature-Inhalte werden zusammengehalten,
auch wenn IDs geändert wurden. Semantisch ähnliche oder geänderte Kopien werden dadurch nicht vollständig erkannt.
Kein Anspruch auf zeitliche Zukunftsvalidierung. Zwei Random-Forest-Tiefen werden ausschließlich auf der Validierung ausgewählt.
Keine Vorverarbeitung mit Fit erforderlich; keine Testverwendung für Auswahl. Abschlusstest vergleicht Basiszeit und ML mit MAE/RMSE/Bias.
Wiederholte Entscheidungen anhand des Tests erfordern einen neuen, unabhängigen Abschlusstest.

Nachgewiesen wird technische Funktion, **keine Genauigkeit auf realen LKW-Fahrten**. Service/Pausen/Ankunftskette sind nicht das Lernziel.
Datenbasis der Universität Wuppertal wird nicht verwendet.

## Quellen und Nutzungsbedingungen · Dokumentationsabruf 27.10.2026

Offline-Demo: keine externen Adress-/Verkehrsdaten. Generator ist Projektcode, Version 1.0.0.

Für den optionalen Straßenmodus:

- Urheber: © OpenStreetMap-Mitwirkende. Datenlizenz ODbL 1.0; Attribution und Bedingungen:
  https://www.openstreetmap.org/copyright und https://opendatacommons.org/licenses/odbl/1-0/ .
- Routingsoftware: OSRM-Mitwirkende; API v1, Referenzdokumentation v5.24.0:
  https://project-osrm.org/docs/v5.24.0/api/ . Dokumentationsversion ist nicht die Serverversion.
- OSRM-Softwarelizenz BSD-2-Clause: https://github.com/Project-OSRM/osrm-backend/blob/master/LICENSE.TXT .
  Serverbedingungen: https://github.com/Project-OSRM/osrm-backend/wiki/Api-usage-policy .
- Adressdatenstand/Abruf kommen aus dem verpflichtenden Quellennachweis, Routing-Abruf aus jedem Snapshot.
  Bei abweichendem Server dessen konkrete Bedingungen ergänzen. Änderungen und betroffene Zeilen stehen in `generation.json`.

TomTom wird **von dieser Trainingspipeline nicht abgefragt und nicht als Trainingsquelle ausgegeben**.
Die bestehende Forecast-App verwendet TomTom International BV, Routing API v1 / Calculate Route:
https://docs.tomtom.com/routing-api/documentation/tomtom-maps/v1/calculate-route (Dokumentationsstand 07.08.2026).
Bedingungen: https://docs.tomtom.com/legal/terms-and-conditions . Das ist keine pauschale Erlaubnis für Training oder dauerhafte Speicherung.
Vor einer späteren Einbindung konkrete Vertrags-/Speicher-/ML-Nutzungsrechte und Antwort-Abrufdaten dokumentieren.

## Änderungsnachweis 1.0.0

Grund: eigenständiger reproduzierbarer IHK-PoC. Methode: neuer Generator, getrennte Datensplits, CPU-Random-Forest,
Datei-Roundtrip, separate CLI/UI und gezielte Tests. Keine Änderung an Forecast-App oder bisherigen Daten.
Laufabhängiger Umfang, Generatorparameter, Prüfsummen und Modellbibliotheksversionen werden automatisch protokolliert.

## Mitgeliefertes geprüftes Beispiel

`example/data/` enthält 100 Offline-Demotouren, Seed 42. `example/model/model.joblib` ist das daraus trainierte,
gespeicherte und erneut geprüfte Modell; `report.json` enthält alle Tourzuordnungen und Bibliotheksversionen.
Die Testprognosen stehen daneben. Nur synthetische Daten, kein externer Dienstabruf.

Prüfung am 27.10.2026: sieben gezielte Tests erfolgreich, CLI-Durchlauf einschließlich separatem Inferenzprozess erfolgreich,
Gradio-Aufbau und UI-Callback erfolgreich. OSRM wurde im automatischen Test ersetzt; kein Live-Straßendienst- oder visueller Browsertest.
