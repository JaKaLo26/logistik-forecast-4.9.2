# Eingebauter KI-Forecast · Adapter 1.2 · 05.11.2026

Das vortrainierte Modell `models/builtin_forecast.joblib` wird beim App-Start
geladen. Kein Training, Upload oder manuelles Aktivieren in der Oberfläche.
Die bisherige vierte Seite entfällt. Training bleibt ein separater Offline-Schritt:
`python scripts/train_builtin_model.py` (aktuelle Skriptversion 2.1).

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

Random Forest, 100 Bäume, feste Parameter und Seed 42. Aktuell 1.000 synthetische
Touren mit insgesamt 8.000 Stopps, genau 500 je Bundesland; Zeitlabels ausdrücklich
synthetisch. Vollständige Touren
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
Die Datei `data/training_stops_de.csv` enthält zehn Teststopps je Bundesland.
Die Hausnummer `1` dient jeweils als synthetischer, ungeprüfter Platzhalter.
Der Generator übernimmt Hausnummern aus einer eigenen Spalte oder aus vorhandenen
Straßenangaben und schreibt die vollständige Zieladresse in die Segment-CSV.
Diese Angaben werden nicht geocodiert. Das Modell sieht
keine Region als Feature; die Segmentdistanz wird unabhängig vom Ort gezogen.
Die gleichmäßige Verteilung und der separate Test mit zwei ungesehenen Bundesländern
belegen daher keine deutschlandweite Genauigkeit auf realen Straßen. Der Bericht
enthält die synthetischen Testfehler nach Land und Distanzband sowie den Abstand
zwischen Trainings- und Testfehler. Für einen belastbaren Straßennachweis wären
geprüfte Koordinaten, echte Routing-Basiszeiten und unabhängige Zielzeiten nötig.

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

## Fahrerbudgets auf Seite 2 · Adapter 1.3 · 05.11.2026

Ursache: Die vorhandene Pausen-/Budgetprüfung nutzte ausschließlich automatisch
angelegte Standardfahrer. Nun bearbeitbare Fahrerliste in Minuten, eine Zeile je
verfügbarem Fahrer. Tages-/Wochen-/Doppelwochenrestlenkzeit, Arbeits-/Dienstrest,
Lenk-/Arbeitsvorzeit seit Pause und Zeit bis Wochenruhe werden übernommen.
Kein neuer Fahrer wird bei Budgetmangel erfunden; offene Aufträge bleiben offen.
Bei Umbenennung/Größenanpassung des LKW bleiben die Fahrerwerte erhalten.
Änderungen an Fahrerbudgets verwerfen die bisherige Planung.
Pausensumme, Tourdauer und Pausenereignisse sind auf Seite 2 sichtbar.
Standardprofil: 9h tägliche Lenkzeit, volle 45-Minuten-Pausen, 11h tägliche
Ruhe geschützt durch verbleibendes Dienstzeitfenster. Keine Ausnahmeprüfung
für verlängerte Lenkzeit, geteilte Pausen oder reduzierte Ruhezeiten.
Vorzeiten und Fristen sind manuelle Vorgaben, kein Tachographenimport.
Vorhandene Grenze bleibt: Forecast-Ankunftskette auf Seite 3 enthält bislang
keine vollständige Pausenzeitfortschreibung; der separate Fahrerplan enthält sie.

Quellenprüfung 05.11.2026: Europäische Kommission, Driving time and rest periods:
https://transport.ec.europa.eu/transport-modes/road/social-provisions/driving-time-and-rest-periods_en
Rechtsgrundlage VO (EG) 561/2006, Art. 6–8, konsolidierter Stand 31.12.2024:
https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02006R0561-20241231
Kein Anspruch auf vollständige rechtliche Compliance-Prüfung.

Prüfung: 3 neue Fahrertests und 9 vorhandene Planungstests bestanden;
App-Callbacktest ergänzt um Fahrerzuordnung, erhaltene Vorzeit und 45min Pause.
Erneute App-Callbackprüfung nach Installation der lokalen Proxy-Abhängigkeit erfolgreich.

## Datenflusskorrektur: lange Strecken

Das eingebaute Modell wurde auf synthetischen Segmenten von 2–25 km bei
45 km/h Basisgeschwindigkeit trainiert. Ein Test mit 262 min / 300 km ergibt
roh rund 39 min: der Random Forest extrapoliert die absolute Fahrzeit nicht.

Planungsmatrix und Segmentforecast prüfen jetzt dieselben Modellgrenzen aus den
Artefakt-Metadaten (Distanz und daraus folgende Basisfahrzeit). Außerhalb davon
bleibt OSRM in der Planung maßgeblich; im Segmentforecast hat ein vorhandener
TomTom-Livewert Vorrang, anschließend Historie, dann OSRM. Auch Modellzeiten unter
50 % der OSRM-Basis werden als unplausibel verworfen. Diese technische Regel
ist kein statistisches Konfidenzintervall. Innerhalb des Bereichs bleibt das
Modell unverändert. Es findet kein neues Training statt.

Verwendeter Wert, Quelle, Warnung und roher Modellwert werden getrennt geführt.
Die Karte zeigt Ersatzwerte ausdrücklich an; Live wird nicht mehr als Ist-Zeit
bezeichnet. Fehlende TomTom-Felder werden nicht mit OSRM/generischen Zeiten
aufgefüllt. Ein vollständiger Tour-Livevergleich erfordert Werte aller Segmente.

Regressionstests mit dem tatsächlich ausgelieferten Modell prüfen den langen
Streckenfall sowie Planung, Fallback, Ankunftskette und Tourensumme.
