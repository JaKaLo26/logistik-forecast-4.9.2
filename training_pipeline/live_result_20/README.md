# Live-Test mit 20 Touren · 28.10.2026

Bestanden: 20 generierte Touren aus drei realen, erneut über Nominatim bestätigten Stuttgarter Adressen.
Je zwei Stopps, insgesamt 40 Segmente. Sechs gerichtete OSRM-Routen live abgefragt, alle HTTP 200.
Keine 20 gemessenen LKW-Fahrten: Lernzielzeiten sind explizit synthetisch.

Split: 14 Training, 3 Validierung, 3 Test. Das Modell in model/model.joblib wurde trainiert, gespeichert,
in einem neuen Prozess geladen und auf sechs Testsegmenten ohne Zielwertspalte angewendet.
Snapshot-Wiederholung ergibt bytegleiche Aufträge, Segmente und Generierungsmanifest.
Sieben Offline-Tests zusätzlich bestanden. Keine Änderung der Forecast-App.

Aufruf: `python -m training_pipeline.live_check NEUER_ORDNER --tours 20`
Testskript Version live_check-1.1: Tourenzahl als Parameter statt fest 12; sonst unveränderte Testmethodik.
Modellgröße und SHA-256: summary.json. Quellenabrufe und Originalantworten: geocode-*.json und data/routing_snapshots/.
Transformation, Umfang und Quellenbedingungen: source.json und data/generation.json. Modellparameter/Splits: model/report.json.

© OpenStreetMap-Mitwirkende, ODbL 1.0: https://www.openstreetmap.org/copyright
Lizenz: https://opendatacommons.org/licenses/odbl/1-0/
Nominatim: https://nominatim.openstreetmap.org/search
Nutzungsbedingungen: https://operations.osmfoundation.org/policies/nominatim/
OSRM Route API v1 / driving: https://router.project-osrm.org
Nutzungsbedingungen: https://github.com/Project-OSRM/osrm-backend/wiki/Api-usage-policy
Datenversion vom Dienst nicht geliefert, konkrete Abrufzeiten stehen in den Snapshots.
Keine TomTom-Aufrufe, keine Prüfung von LKW-Straßenbeschränkungen oder einer veröffentlichten Browseroberfläche.
