# Live-Prüfung vom 28.10.2026

Ergebnis: bestanden. Drei öffentliche Adressen in Stuttgart über Nominatim live aufgelöst,
sechs gerichtete Straßenrouten über OSRM live abgefragt (alle HTTP 200).
Zwölf synthetische Touren mit je zwei Stopps, 24 Aufträge. Split: 8 Training, 2 Validierung, 2 Test.

`model/model.joblib` ist das tatsächlich trainierte Random-Forest-Modell (13.517 Bytes).
Ein separater Python-Prozess lädt es und prognostiziert vier Testsegmente aus `inference-input.csv`,
das keine synthetischen Zielwerte enthält. Ergebnisse: `predictions.csv`.
`summary.json` enthält Prüfergebnisse und SHA-256; `model/report.json` Modellwahl und Split-IDs.
Sieben zusätzliche Offline-Tests bestanden.

Auftragsimport wurde mit `normalize_orders` geprüft. Identische Neugenerierung mit gespeicherten Routenantworten
wurde ohne Netzverkehr geprüft (orders.csv, segments.csv und generation.json bytegleich).

Live-Testskript: `python -m training_pipeline.live_check NEUER_AUSGABEORDNER` vom Repository-Verzeichnis.
Es verwendet echte HTTP-Antworten; der Request-Wrapper ergänzt nur User-Agent und 1,1 Sekunden Abstand.
Der Lauf erzeugt einen neuen Ordner und überschreibt keine früheren Tests.

## Herkunft und Grenzen

© OpenStreetMap-Mitwirkende, ODbL 1.0: https://www.openstreetmap.org/copyright
Lizenz: https://opendatacommons.org/licenses/odbl/1-0/
Geocoding: https://nominatim.openstreetmap.org/search
Bedingungen: https://operations.osmfoundation.org/policies/nominatim/
Straßenrouting: OSRM Route API v1, driving, https://router.project-osrm.org
Bedingungen: https://github.com/Project-OSRM/osrm-backend/wiki/Api-usage-policy
Konkrete Abrufzeiten, Koordinaten und Originalantworten stehen in geocode-*.json und data/routing_snapshots/.
Server-Datenversion wurde nicht geliefert und daher nicht erfunden.
Adressfeld-Auswahl und Transformation: source.json; Generatorversion, Parameter, Umfang und Hashes: data/generation.json.

Die Zielzeiten bleiben synthetisch. Der Test belegt technische Funktion, keine Genauigkeit echter LKW-Fahrten.
Das OSRM-Profil prüft keine LKW-Beschränkungen. TomTom ist nicht Bestandteil dieses Trainingslaufs.
Kein Test einer veröffentlichten Hugging-Face-Oberfläche oder der visuellen Browserdarstellung.
