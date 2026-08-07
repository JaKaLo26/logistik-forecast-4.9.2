---
title: LagerForecast v4
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
pinned: false
---

# LagerForecast v4 – Logistik Forecast 4.9.2

Modulare Python-/Gradio-Anwendung für:

- CSV-Import von Aufträgen
- automatische und manuelle Adressprüfung
- skalierbare Fahrzeugflotte
- Kapazitätsverteilung nach Palettenplätzen und Nutzlast
- OSRM-Routing
- HERE-Live-Traffic
- Verkehrszuschläge
- API-Debug-Ausgabe
- Forecast je LKW
- farblich getrennte Routen
- Paletten- und Gewichtsanzeige je Auftrag

---

## Projektziel

LagerForecast v4 unterstützt die Planung von Lieferfahrten mit mehreren LKW.

Vor der Forecast-Berechnung werden zuerst die verfügbaren Fahrzeuge auf die Aufträge verteilt. Dabei werden gleichzeitig berücksichtigt:

- Palettenstellplätze
- Nutzlast
- Auftragsgewicht
- Anzahl der Paletten
- Servicezeiten

Anschließend werden die Routen berechnet und aktuelle Verkehrsdaten ausgewertet.

---

## Ablauf der Anwendung

Die Oberfläche ist in drei Schritte unterteilt.

### 1. CSV und Adressprüfung

Aufträge werden per CSV importiert.

Danach werden die Adressen automatisch geocodiert.

Unsichere Adressen werden mit:

`MANUELL PRÜFEN`

markiert.

Diese Treffer können direkt in der Tabelle manuell angepasst werden.

---

### 2. Flotte und Kapazitätsverteilung

Standardmäßig stehen zur Verfügung:

- 3 × 14-t-LKW
- 3 × 40-t-LKW

Die Fahrzeugliste ist vollständig editierbar.

Fahrzeuge können:

- ergänzt
- entfernt
- deaktiviert
- angepasst

werden.

Für jedes Fahrzeug werden geprüft:

- Palettenkapazität
- Nutzlast
- aktuelle Auslastung
- zugewiesene Aufträge

Ein Auftrag wird nur zugewiesen, wenn sowohl genügend Stellplätze als auch genügend freie Nutzlast vorhanden sind.

---

## Standardfahrzeuge

### 14-t-LKW

Standardwerte:

- 18 Palettenstellplätze
- 6.000 kg Nutzlast

### 40-t-LKW

Standardwerte:

- 33 Palettenstellplätze
- 24.000 kg Nutzlast

Wichtig:

Die Bezeichnungen 14 t und 40 t beziehen sich nicht direkt auf die Nutzlast.

Die tatsächliche Nutzlast hängt vom Fahrzeug, Aufbau und Leergewicht ab.

Deshalb können die Werte in der Anwendung angepasst werden.

---

## 3. Routing und Forecast

Nach erfolgreicher Kapazitätsverteilung wird die Route für jedes Fahrzeug separat berechnet.

Die Routen werden auf der Karte farblich unterschieden.

Jeder LKW besitzt eine eigene Farbe.

Zusätzlich wird die Route mit einer dunkleren Umrandung dargestellt, damit sich mehrere Routen besser voneinander unterscheiden lassen.

---

## Verkehrsdaten

Aktuell können folgende Datenquellen verwendet werden:

### HERE Traffic API

Primäre Quelle für:

- Verkehrsgeschwindigkeit
- Verkehrsfluss
- Stauintensität
- Verzögerung
- Jam Factor

### Autobahn API

Zusätzliche Quelle für:

- Baustellen
- Verkehrswarnungen
- Sperrungen

Die Autobahn-API ist als eigener Provider eingebunden und wird in kommenden Versionen noch genauer räumlich mit der Route abgeglichen.

---

## Live-Verkehrszuschlag

Der Forecast berücksichtigt nicht nur die reine OSRM-Fahrzeit.

Zusätzlich wird ein Verkehrszuschlag berechnet.

Dieser basiert unter anderem auf:

- aktueller Geschwindigkeit
- normaler Geschwindigkeit
- Jam Factor
- Verkehrsbelastung
- verfügbaren Traffic-Daten

Der berechnete Zuschlag wird in Sekunden bzw. Minuten zum Forecast addiert.

---

## API-Debug

Die Anwendung besitzt eine Debug-Ausgabe für Routing und Verkehr.

Dort wird angezeigt:

- welche Traffic-Quelle verwendet wurde
- welche HERE-Werte empfangen wurden
- Jam Factor
- Geschwindigkeiten
- API-Status
- berechnete Verzögerung
- Traffic Score
- Datenvertrauen
- OSRM-Distanz
- OSRM-Fahrzeit

Damit kann kontrolliert werden, ob die API-Daten korrekt interpretiert werden.

---

## CSV-Format

Die CSV-Datei sollte folgende Spalten enthalten:

```csv
auftrag,kunde,strasse,plz,ort,paletten,warengewicht_kg,service_min,zeitfenster_von,zeitfenster_bis