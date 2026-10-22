---
title: LagerForecast 4.9.3
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
pinned: false
---

# LagerForecast 4.9.3

Version 4.9.3 setzt die neue verbindliche Prozessstruktur um:

1. CSV importieren.
2. Adressen automatisch prüfen.
3. Nur unsichere Adressen nacheinander bestätigen.
4. Depot und verfügbare Flotte festlegen.
5. Aufträge geografisch clustern.
6. Cluster unter Paletten- und Nutzlastgrenzen LKW zuweisen.
7. Erst danach die Stopp-Reihenfolge je LKW über OSRM Trip optimieren.
8. Erst die fertige Route auf Verkehr/Störungen prüfen.
9. Forecast und Kartenansicht je LKW anzeigen.

## Wesentliche Änderung zu 4.9.2

Die Auftragsverteilung arbeitet nicht mehr nach dem Prinzip „was noch hineinpasst“.

Stattdessen werden räumlich nahe Stopps zu zusammenhängenden Liefergebieten gebündelt. Dadurch sollen sich Touren weniger überschneiden und jeder LKW möglichst wenig unnötige Strecke zwischen Regionen fahren.

## Clusterheuristik

- Farthest-first Seed ab Depot
- Near-neighbour Region Growing
- harte Grenzen für Palettenplätze und Nutzlast
- leichte Berücksichtigung der Zeitfenster-Mittelpunkte
- kompakte, möglichst nicht überlappende Liefergebiete

## Routenoptimierung

Nach der Clusterbildung wird pro Fahrzeug der OSRM-Trip-Service verwendet, um die Reihenfolge der Stopps als Rundtour ab/bis Depot zu optimieren.

## Verkehr / Forecast

Verkehr wird bewusst erst nach der Routenoptimierung ausgewertet.

Die offizielle Autobahn-API ist als Adapter vorhanden. Die produktive räumliche Zuordnung von Autobahn-Ereignissen zur optimierten Route ist der nächste Entwicklungsschritt und wird nicht durch künstliche Baustellenindizes ersetzt.

## Start

```bash
pip install -r requirements.txt
python app.py
```

## IHK-Fachgespräch: Planbasis, ML und IST (Testbranch)

Umsetzung des Themenblatts „Entwicklung eines KI-gestützten Logistik-Forecasts für LKW-Liefertouren“:

| Anforderung | Umsetzung |
|---|---|
| Einfache Vergleichsberechnung | OSRM-Leg-Fahrzeit als unveränderte Baseline |
| ML-Modell | Random Forest mit vorab festgelegten Parametern, CPU genügt |
| Datenqualität | Pflichtfelder, positive endliche Werte, Zeitzonen, chronologische IST-Prüfung |
| Training/Test trennen | Zeitliche 80/20-Aufteilung ganzer Touren; nur vor Testbeginn verfügbare Trainingslabels |
| Fehler und Grenzen | MAE, RMSE, Bias und Plan/ML/IST je Testsegment in Tab 6 |
| Aktualisierung unterwegs | Sitzungsmodell wird in Forecast und Reforecast verwendet |
| Nachvollziehbarkeit | Feature-Liste, Modell-ID, Datensatz-Hash, Split-IDs und Parameter im Manifest |

### CSV für Tab 6

Eine Zeile je ursprünglicher Segmentprognose; Reforecast-Kopien sind nicht zugelassen.
Pflichtspalten:

```csv
tour_id,segment_id,prediction_created_at,planned_departure_at,actual_departure_at,actual_arrival_at,actual_recorded_at,osrm_baseline_s,distance_m,data_origin
```

Zeiten sind ISO 8601 mit Zeitzone. `actual_travel_s` wird ausschließlich aus tatsächlicher Abfahrt und Ankunft berechnet, einschließlich unterwegs anfallender Wartezeit, ohne Service am Ziel. Alle Features müssen beim Erstellen der ursprünglichen Prognose vorgelegen haben. Der CSV-Lieferant muss diese Herkunft sicherstellen; die Zeitprüfung allein beweist keine korrekte Quelle. `data_origin` ist `real`, `synthetic` oder `simulation`; Herkünfte werden nicht gemischt. Mindestens fünf Touren sind ein technisches Minimum, keine ausreichende Stichprobengröße für einen allgemeinen Qualitätsnachweis.

`data/ihk_synthetic_example.csv` ist ein ausdrücklich synthetischer Funktionstest, kein Beleg realer Modellgüte. Training aktiviert das Modell nur in der aktuellen Gradio-Sitzung. Es wird kein fremdes Pickle-Modell geladen. Eine neue Sitzung beginnt mit der bestehenden Heuristik. Die Testtouren werden nicht anschließend zum Nachtrainieren verwendet.

Die Features sind ausschließlich OSRM-Basiszeit, OSRM-Distanz, geplante lokale Abfahrtsstunde und Wochentag. IST-Zeiten, prognostizierte Ankunft, Kundennamen, IDs und Forecast-Ausgaben sind keine Features. Weitere Faktoren wie Verkehr, Baustellen, Ladung und Wetter sind erst nach zuverlässiger zeitlicher Quellenzuordnung aufzunehmen. Der produktive Verkehrsdatenabgleich ist weiterhin offen.

### Prognoseprotokoll V2

Neue Forecasts werden unter `forecasts/runs/<run_id>.json` zuerst lokal, danach bei Konfiguration in HF gespeichert. Eingaben, Provider-Schätzungen, Ausgaben und bekannter Betriebszustand sind getrennt. Wiederverwendete Segmente behalten ihre ursprüngliche Version und Erstellungszeit. Wiederholungen desselben Runs sind lokal idempotent; abweichende Inhalte unter gleicher ID werden abgewiesen. HF-Fehler bleiben sichtbar. Lokale Daten brauchen auf Hugging Face ein dauerhaftes Volume; automatischer Nachversand ist noch nicht implementiert.

Die alten JSONL-Dateien bleiben unverändert. Sie werden nicht automatisch als Training akzeptiert. Für echte Auswertungen müssen unabhängig erfasste IST-Ereignisse mit den ursprünglichen Eingabe-Snapshots zu obigem CSV-Schema verbunden werden. Die App verfügt noch nicht über GPS-/Telematik-Import. Alte `log_actual_*`-Funktionen sind Legacy und werden vom neuen Trainingsworkflow nicht verwendet.

### Bewertung und praktische Grenzen

Positive Differenz `MAE Plan − MAE ML` bedeutet nur im konkreten Holdout einen Vorteil. Negative Werte sind ein reguläres Ergebnis und werden nicht ausgeblendet. Wiederholte Entscheidungen am Test erfordern einen neuen unberührten Abschlusstest. Segmentmetriken sind keine Ende-zu-Ende-ETA-Metriken: Serviceabweichungen und Fehlerfortpflanzung müssen dafür separat anhand kompletter Touren untersucht werden. Spätere geplante Segmentabfahrten stammen aus den vorherigen Prognosen; entsprechende Trainings-Snapshots müssen nach demselben Verfahren erzeugt sein.

Bei fehlender OSRM-Leg-Distanz/-Zeit bleibt die Heuristik aktiv; die konkrete `model_id` wird pro Segment gespeichert. Synthetische Daten können den Ablauf demonstrieren, aber keine Aussage über reale Einsparungen liefern. Ein wirtschaftlicher Nutzen muss später über gemessene Dispositionszeit, Pünktlichkeit und Betriebskosten bewertet werden.

Tests: `python -m unittest discover -s tests -p 'test_ihk.py' -v`
