# Logistik Forecast 4.9.2

Modulare Gradio-Demo für CSV-Import, manuelle Adressprüfung, skalierbare Fahrzeugflotte, Paletten-/Traglastverteilung, OSRM-Routing und HERE-Traffic-Debug.

## Schnellstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Danach: `http://localhost:7860`

## CSV-Spalten

`auftrag,kunde,strasse,plz,ort,paletten,warengewicht_kg,service_min,zeitfenster_von,zeitfenster_bis`

## Ablauf

1. CSV importieren.
2. Adressen automatisch prüfen; unsichere Treffer manuell überschreiben.
3. Fahrzeugliste beliebig erweitern oder reduzieren.
4. Kapazität nach Palettenplätzen und Nutzlast verteilen.
5. Forecast berechnen.
6. Im API-Debug prüfen, welche HERE-Felder empfangen und wie sie interpretiert wurden.

## Wichtige Hinweise

- Ohne HERE-Key wird kein echter Live-Zuschlag berechnet; die App markiert dies im Debug ausdrücklich.
- Öffentliche OSRM- und Nominatim-Dienste sind für Entwicklung/Demos gedacht. Für Produktion sind eigener Dienst, Caching, Rate-Limits und ein korrektes Kontakt-User-Agent nötig.
- Die Autobahn-API ist als Adapter vorbereitet. Für räumlich korrekte Meldungen müssen im nächsten Schritt Autobahnnummern aus OSRM-Schritten extrahiert und die Ereignisse gegen die Routengeometrie gematcht werden.
- Ein 14-t- oder 40-t-Wert bezeichnet nicht automatisch die Nutzlast. Fahrzeugwerte bleiben deshalb editierbar.

## GitHub / Hugging Face Spaces

Das Repository kann direkt zu GitHub gepusht werden. Für Hugging Face Spaces einen **Gradio Space** anlegen und diese Dateien in den Repository-Root kopieren. Das Secret `HERE_API_KEY` im Space hinterlegen.
