# KI-Forecast – Integrationsadapter 1.1 · 05.11.2026

In `Ai-workflow` ist für die Forecast-Aktion ein trainiertes Modell erforderlich.
Auf Seite 4 entweder eine Trainings-CSV trainieren oder ein selbst erzeugtes,
vertrauenswürdiges `model.joblib` laden. Joblib darf nur aus eigener/vertrauter
Quelle stammen. Das App-Training bietet das Modell zum Download an und lädt es
vor Aktivierung erneut. Die Aktivierung gilt für die aktuelle Sitzung.
Anschließend auf Seite 3 Routen und Forecast berechnen.

Unterstützt werden das bestehende Vier-Merkmal-App-Modell und das separate
Pipeline-Schema 1.0.0 mit fünf Merkmalen (einschließlich Restladung).
Die Modellkennung wird aus dem Dateiinhalt abgeleitet. Beide verwenden die
bestehenden OSRM-Segmentzeiten und Distanzen. TomTom ist ein optionaler Vergleich;
fehlende Live-Daten werden weiterhin ausdrücklich als fehlend angezeigt.
Die Benutzeraktion fällt nicht unbemerkt auf heuristische Zeiten zurück.

## Begründung, Methode und Umfang

Gezielte Adapteränderungen an app.py, src/ml_training.py und src/forecast.py;
joblib als direkte Abhängigkeit. Bestehende Trainingssplits bleiben erhalten.
Geprüftes Merkmalsschema beim Laden; ungültige Streckenwerte oder Modellzeiten
führen zum Fehler. Paletten und Warengewicht werden vor Summierung geprüft;
Rückfahrt hat Restladung null. Neue Abfahrt folgt Modellankunft plus Servicezeit.
Bei Modellwechsel werden bisherige Forecast-Karten gelöscht.
Die untergeordnete Forecast-API behält den alten Heuristikmodus für bestehende
Aufrufer; die App-Aktion lässt diesen Modus nicht mehr zu.

## Prüfung

6 Adaptertests, 5 IHK-Tests und 9 Planungs-/Clusteringtests erfolgreich.
`python tests/check_ai_forecast_app.py` prüft den Gradio-Aufbau und reale Callbacks:
CSV-Training, Export, Laden, Forecast ohne TomTom-Schlüssel bei gesperrter
Heuristik, sowie Ablehnung einer Forecast-Anfrage ohne Modell.
Externe Forecast-Protokollierung ist im Callbacktest ersetzt.
Kein Live-API-Test, kein visueller Browser-/Deployment-Test.

Bestehender separater Fehler: tests/test_capacity.py importiert die nicht mehr
vorhandene Funktion distribute_orders aus src.capacity. Vollständige allgemeine
Testdiscovery ist deshalb nicht grün; die oben genannten Tests sind separat geprüft.

## Aussagegrenzen und Quellen

Synthetische Modellziele sind ein technischer Funktionsnachweis, kein
Genauigkeitsnachweis für reale LKW-Fahrten. Der Wuppertal-Datensatz wird nicht
verwendet. Diese Änderung führt keine neue Datenquelle ein und ruft keine
externen Karten-/Verkehrsdaten für das Training ab. Generatorparameter und
Trainingsmanifest verbleiben bei den bisherigen Trainingsverfahren. Vorhandene
Quellen-/Lizenznachweise der tatsächlich eingesetzten Routingdienste gelten
weiter; dieses Integrationsprotokoll ersetzt diese nicht.
