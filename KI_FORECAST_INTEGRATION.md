# Eingebauter KI-Forecast · Adapter 1.2 · 05.11.2026

Das vortrainierte Modell `models/builtin_forecast.joblib` wird beim App-Start
geladen. Kein Training, Upload oder manuelles Aktivieren in der Oberfläche.
Die bisherige vierte Seite entfällt. Training bleibt ein separater Offline-Schritt:
`python scripts/train_builtin_model.py` (Skriptversion 1.0).

Seite 2: Depot und Startzeit angeben. Die gerichtete OSRM-Matrix liefert
Basiszeiten und Distanzen. Das Modell prognostiziert daraus die Fahrzeiten
zur gewählten Startzeit. Diese KI-Zeitmatrix steuert die bestehende zulässige
Einfügeplanung: Gruppierung, Reihenfolge und benötigte Fahrzeuge.
Kapazitäts-/Fahrerbudgetregeln bleiben explizite Prüfungen; das Modell selbst
ist ein Fahrzeitregressor, kein trainierter Fahrzeugzuordnungsalgorithmus.
Die Matrix ist eine Startzeit-Näherung, keine zeitabhängige globale Optimierung.

Seite 3: Straßenroute der gewählten Reihenfolge abrufen, Modellzeiten entlang
der Tour mit fortgeschriebener Abfahrt und Service berechnen und Fahrerbudget
nochmals prüfen. Fehlgeschlagene Touren werden nicht als zulässig ausgegeben.
Forecast-Ankünfte enthalten wie bisher Fahrzeit und Service, jedoch keine
vollständig eingearbeitete Pausenplanung. Pausen sind im separaten Fahrerplan
berücksichtigt. Diese Grenze bleibt für lange Touren bestehen.
Bei geänderter Startzeit werden Tourgruppen und Forecast verworfen.

## Modell und Reproduzierbarkeit

Random Forest, 100 Bäume, feste Parameter und Seed 42. 100 synthetische Touren
mit je 5 Aufträgen; Zeitlabels ausdrücklich synthetisch. Vollständige Touren
für Training, Validierung und Test getrennt. Generator 1.0, Parameter,
Datensatzhash, Modellparameter, Bibliotheksversion, Split-IDs und Testmetriken:
`models/builtin_report.json`. Rohaufträge, Segmente und Testprognosen liegen
neben dem Modell. Die Adressen sind erfunden und nicht zum Geocodieren geeignet.
Modellspeicherung und erneutes Laden vor Nutzung; kein Training beim Start.
Scikit-learn-Version für das mitgelieferte Artefakt in requirements.txt fixiert.

Trainingsbasis ist vollständig offline synthetisch: zufällige Segmentdistanzen
2–25 km und angenommene Basisgeschwindigkeit 45 km/h. Kein Straßen-/LKW-
Genauigkeitsnachweis und keine belastbare Extrapolation außerhalb dieses
Szenarios. Der Wuppertal-Datensatz und TomTom werden nicht fürs Training benutzt.

## Änderungen und Prüfungen

Grund: automatische Nutzung eines mitgelieferten Modells auf Seite 2 und 3.
Methode: vortrainiertes Artefakt, Ladeadapter, gebündelte Matrixinferenz,
KI-basierte erneute Budgetprüfung der Straßenroute, Entfernung der Trainings-UI.
Warengewicht wird beim Übergeben der Route erhalten. Deployment lädt exakt das
geprüfte Artefakt hoch; andere Pickle-Dateien bleiben ausgeschlossen.

24 gezielte Unit-Tests sowie `python tests/check_ai_forecast_app.py` prüfen
Modellvertrag, getrennte Touren, Speichern/Laden, KI-Matrix, ungültige Daten,
Forecast/Reforecast, Seite-2-Zuordnung und Seite-3-Route/Forecast ohne Heuristik.
Straßendienste und externe Protokollierung sind im App-Test ersetzt.
Kein Live-API- oder visueller Browsertest. Der bereits veraltete Kapazitätstest
importiert weiterhin die entfernte Funktion distribute_orders.

## Externe Grundlagen

Keine externen Daten beim Offline-Training oder bei diesen Tests abgerufen.
Im Betrieb: OSRM Table API v1/driving mit annotations=duration,distance
(statt bisher nur duration), Route API v1/driving unverändert.
OSRM: https://project-osrm.org/docs/v5.24.0/api/ (Dokumentationsversion,
nicht bestätigte Serverversion), BSD-2-Clause:
https://github.com/Project-OSRM/osrm-backend/blob/master/LICENSE.TXT
Bedingungen: https://github.com/Project-OSRM/osrm-backend/wiki/Api-usage-policy
© OpenStreetMap-Mitwirkende, ODbL 1.0:
https://www.openstreetmap.org/copyright
TomTom bleibt der optionale bestehende Livevergleich; kein neuer TomTom-Dienst.
Datenstand/Abrufzeit der späteren Betriebsanfragen sind nicht mit dem Datum
dieser Codeänderung gleichzusetzen. Vollständige laufbezogene Quellenprotokolle
sind weiterhin eine gesonderte Anforderung; dieser Änderungstest erzeugt keine.
