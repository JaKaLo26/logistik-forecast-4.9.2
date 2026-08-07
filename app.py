from __future__ import annotations

import json
import os
from io import StringIO

import gradio as gr
import pandas as pd
import spaces
from dotenv import load_dotenv

from src.capacity import normalize_orders, summarize_orders, distribute_orders
from src.geocoding import Geocoder
from src.models import Vehicle
from src.routing import OSRMRouter
from src.traffic import HereTrafficProvider, AutobahnProvider, combine
from src.maps import build_map
from src.forecast import forecast_summary


@spaces.GPU(duration=1)
def zerogpu_startup_check():
    """ZeroGPU-Kompatibilitätsfunktion. Wird von der Logistik-App nicht verwendet."""
    return True


load_dotenv()
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))

DEFAULT_VEHICLES = [
    Vehicle(f"LKW-K0{i}", "14 t", 18, 6000, c)
    for i, c in enumerate(["#2563eb", "#7c3aed", "#0891b2"], 1)
] + [
    Vehicle(f"LKW-G0{i}", "40 t", 33, 24000, c)
    for i, c in enumerate(["#dc2626", "#ea580c", "#16a34a"], 1)
]


def vehicles_df():
    return pd.DataFrame([v.to_dict() for v in DEFAULT_VEHICLES])


def _read_json_records(payload: str) -> pd.DataFrame:
    if not payload or payload.strip() in ("", "[]", "null"):
        return pd.DataFrame()
    return pd.read_json(StringIO(payload), orient="records")


def _normalize_column_name(name: str) -> str:
    value = str(name).strip().lower()
    value = value.replace(" ", "_").replace("-", "_")
    aliases = {
        "straße": "strasse",
        "str.": "strasse",
        "gewicht": "warengewicht_kg",
        "gewicht_kg": "warengewicht_kg",
        "servicezeit": "service_min",
        "servicezeit_min": "service_min",
        "entladezeit": "service_min",
        "entladezeit_min": "service_min",
        "palettenanzahl": "paletten",
        "anzahl_paletten": "paletten",
        "auftragsnummer": "auftrag",
        "auftrag_id": "auftrag",
        "kundenname": "kunde",
        "postleitzahl": "plz",
        "stadt": "ort",
    }
    return aliases.get(value, value)


def read_csv(file):
    if not file:
        raise gr.Error("Bitte zuerst eine CSV-Datei auswählen.")

    try:
        try:
            df = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig")
            used_encoding = "utf-8-sig"
        except UnicodeDecodeError:
            df = pd.read_csv(file, sep=None, engine="python", encoding="cp1252")
            used_encoding = "cp1252"

        if df.empty:
            raise gr.Error("Die CSV-Datei ist leer oder enthält keine Datensätze.")

        original_columns = [str(c) for c in df.columns]
        df.columns = [_normalize_column_name(c) for c in df.columns]

        required = [
            "auftrag", "kunde", "strasse", "plz", "ort",
            "paletten", "warengewicht_kg", "service_min",
        ]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise gr.Error(
                "CSV-Struktur nicht erkannt.\n\n"
                f"Fehlende Spalten:\n{', '.join(missing)}\n\n"
                f"Original erkannte Spalten:\n{', '.join(original_columns)}\n\n"
                f"Normalisierte Spalten:\n{', '.join(df.columns)}\n\n"
                "Erwartet werden mindestens:\n" + ", ".join(required)
            )

        for col in ["paletten", "warengewicht_kg", "service_min"]:
            try:
                cleaned = df[col].astype(str).str.strip()
                cleaned = cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
                cleaned = cleaned.replace({"nan": None, "None": None, "": None})
                df[col] = pd.to_numeric(cleaned, errors="raise")
            except Exception as e:
                raise gr.Error(
                    f"Ungültige Zahlenwerte in der Spalte '{col}'.\n\n"
                    f"Technischer Fehler: {type(e).__name__}: {e}"
                )

        df["auftrag"] = df["auftrag"].astype(str).str.strip()
        df["kunde"] = df["kunde"].astype(str).str.strip()
        df["strasse"] = df["strasse"].astype(str).str.strip()
        df["plz"] = df["plz"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        df["ort"] = df["ort"].astype(str).str.strip()

        if (df["paletten"] < 0).any():
            raise gr.Error("Die Spalte 'paletten' enthält negative Werte.")
        if (df["warengewicht_kg"] < 0).any():
            raise gr.Error("Die Spalte 'warengewicht_kg' enthält negative Werte.")
        if (df["service_min"] < 0).any():
            raise gr.Error("Die Spalte 'service_min' enthält negative Werte.")

        df = normalize_orders(df)
        s = summarize_orders(df)
        summary = (
            "✅ **CSV erfolgreich importiert**\n\n"
            f"- Aufträge: {s['auftraege']}\n"
            f"- Paletten: {s['paletten']}\n"
            f"- Gesamtgewicht: {s['gewicht_kg']:,} kg\n"
            f"- Ø Gewicht/Palette: {s['durchschnitt_kg_pro_palette']} kg\n"
            f"- Encoding: {used_encoding}\n\n"
            "**Erkannte Spalten:**\n" + ", ".join(df.columns)
        )
        return df, summary, df.to_json(orient="records", force_ascii=False)

    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(
            "CSV konnte nicht importiert werden.\n\n"
            f"Fehlertyp: {type(e).__name__}\n"
            f"Fehlermeldung: {e}"
        )


def geocode_orders(orders_json):
    df = _read_json_records(orders_json)
    if df.empty:
        raise gr.Error("Bitte zuerst eine CSV-Datei erfolgreich importieren.")

    geocoder = Geocoder(
        os.getenv("NOMINATIM_BASE_URL", "https://nominatim.openstreetmap.org"),
        TIMEOUT,
    )
    rows, candidates = [], {}

    for _, r in df.iterrows():
        try:
            hits = geocoder.search(r["adresse"])
            geocode_error = ""
        except Exception as e:
            hits = []
            geocode_error = f"{type(e).__name__}: {e}"

        best = hits[0] if hits else None
        uncertain = not best or best["confidence"] < 0.72
        rows.append({
            "auftrag": r["auftrag"],
            "eingabe": r["adresse"],
            "treffer": best["display_name"] if best else "",
            "confidence": best["confidence"] if best else 0,
            "status": "MANUELL PRÜFEN" if uncertain else "OK",
            "lat": best["lat"] if best else None,
            "lon": best["lon"] if best else None,
            "fehler": geocode_error,
        })
        candidates[str(r["auftrag"])] = hits

    return (
        pd.DataFrame(rows),
        json.dumps(candidates, ensure_ascii=False),
        "Unsichere Treffer können direkt in der Tabelle überschrieben werden. Bei 'MANUELL PRÜFEN' bitte Treffer, Latitude und Longitude kontrollieren.",
    )


def save_addresses(address_table, orders_json):
    orders = _read_json_records(orders_json)
    if orders.empty:
        raise gr.Error("Keine Aufträge vorhanden.")

    adr = pd.DataFrame(address_table)
    required_cols = ["auftrag", "treffer", "lat", "lon"]
    missing = [c for c in required_cols if c not in adr.columns]
    if missing:
        raise gr.Error("Adressprüfung unvollständig. Fehlende Tabellenspalten: " + ", ".join(missing))

    adr["auftrag"] = adr["auftrag"].astype(str)
    orders["auftrag"] = orders["auftrag"].astype(str)
    merged = orders.merge(adr[required_cols], on="auftrag", how="left")

    if merged[["lat", "lon"]].isna().any().any():
        bad = merged[merged[["lat", "lon"]].isna().any(axis=1)]["auftrag"].tolist()
        raise gr.Error("Mindestens eine Adresse hat keine gültigen Koordinaten.\nBetroffene Aufträge: " + ", ".join(map(str, bad)))

    merged["lat"] = pd.to_numeric(merged["lat"], errors="raise")
    merged["lon"] = pd.to_numeric(merged["lon"], errors="raise")
    return merged.to_json(orient="records", force_ascii=False), "✅ Adressen übernommen."


def distribute(orders_geo_json, vehicle_table):
    orders = _read_json_records(orders_geo_json)
    if orders.empty:
        raise gr.Error("Bitte zuerst die geprüften Adressen übernehmen.")
    vehicles = pd.DataFrame(vehicle_table)
    if vehicles.empty:
        raise gr.Error("Es sind keine Fahrzeuge vorhanden.")

    ass, util, warnings = distribute_orders(orders, vehicles)
    msg = "✅ Alle Aufträge zugewiesen." if not warnings else "⚠️ " + " | ".join(warnings)
    return ass, util, ass.to_json(orient="records", force_ascii=False), msg


def calculate(assign_json, vehicle_table, here_key):
    ass = _read_json_records(assign_json)
    if ass.empty:
        raise gr.Error("Bitte zuerst die Kapazität verteilen.")
    vehicles = pd.DataFrame(vehicle_table)
    if vehicles.empty:
        raise gr.Error("Keine Fahrzeugdaten vorhanden.")

    router = OSRMRouter(os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org"), TIMEOUT)
    routes, summaries, debug = [], [], []
    active = ass[ass.vehicle_id != "NICHT ZUGEWIESEN"]

    if active.empty:
        raise gr.Error("Es gibt keine zugewiesenen Aufträge für den Forecast.")

    for vid, group in active.groupby("vehicle_id"):
        matching_vehicle = vehicles[vehicles.vehicle_id == vid]
        if matching_vehicle.empty:
            debug.append({"vehicle_id": vid, "error": "Fahrzeug ist in der Fahrzeugtabelle nicht mehr vorhanden."})
            continue
        v = matching_vehicle.iloc[0]
        coords = [(float(x.lat), float(x.lon)) for _, x in group.iterrows()]
        if len(coords) == 1:
            coords = coords + coords

        try:
            route = router.route(coords)
        except Exception as e:
            debug.append({"vehicle_id": vid, "provider": "OSRM", "error": f"{type(e).__name__}: {e}"})
            continue

        try:
            here = HereTrafficProvider(here_key or os.getenv("HERE_API_KEY", ""), TIMEOUT).analyze_route(route["geometry"], route["duration_s"])
        except Exception as e:
            here = {"provider": "HERE", "available": False, "score": 0, "confidence": 0, "delay_s": 0, "error": f"{type(e).__name__}: {e}"}

        try:
            autobahn = AutobahnProvider(timeout=TIMEOUT).analyze_route(route["geometry"], route["duration_s"])
        except Exception as e:
            autobahn = {"provider": "Autobahn API", "available": False, "score": 0, "confidence": 0, "delay_s": 0, "error": f"{type(e).__name__}: {e}"}

        traffic = combine([here, autobahn], {"HERE": 0.7, "Autobahn API": 0.3})
        service = float(group.service_min.sum())
        summary = forecast_summary(route["distance_m"], route["duration_s"], traffic["delay_s"], service)
        summary.update({
            "vehicle_id": vid,
            "paletten": int(group.paletten.sum()),
            "gewicht_kg": int(group.gesamtgewicht_kg.sum()),
            "traffic_score": round(traffic["score"], 1),
            "datenvertrauen_pct": round(traffic["confidence"] * 100),
        })
        summaries.append(summary)

        stops = [{
            "lat": float(r.lat),
            "lon": float(r.lon),
            "kunde": r.kunde,
            "paletten": int(r.paletten),
            "gesamtgewicht_kg": int(r.gesamtgewicht_kg),
        } for _, r in group.iterrows()]

        routes.append({"vehicle_id": vid, "color": v["color"], "geometry": route["geometry"], "stops": stops})
        debug.append({
            "vehicle_id": vid,
            "traffic": traffic,
            "here_raw": here,
            "autobahn_raw": autobahn,
            "osrm_summary": {"distance_m": route["distance_m"], "duration_s": route["duration_s"]},
        })

    if not routes:
        raise gr.Error("Keine Route konnte erfolgreich berechnet werden. Bitte API-Debug bzw. Logs prüfen.")

    return build_map(routes), pd.DataFrame(summaries), json.dumps(debug, ensure_ascii=False, indent=2, default=str)


CSS = """
.step-title{font-size:1.35rem;font-weight:700;margin-bottom:.35rem}.muted{color:#6b7280}
"""

with gr.Blocks(title="Logistik Forecast 4.9.2", css=CSS) as demo:
    gr.Markdown("# Logistik Forecast 4.9.2\nMehrstufige Python-/Gradio-Demo")
    orders_state = gr.State("[]")
    geo_state = gr.State("[]")
    assignments_state = gr.State("[]")
    candidates_state = gr.State("{}")

    with gr.Tabs():
        with gr.Tab("1 · CSV & Adressen"):
            gr.Markdown('<div class="step-title">Aufträge importieren und Adressen prüfen</div>')
            upload = gr.File(label="CSV-Datei", file_types=[".csv"], type="filepath")
            import_btn = gr.Button("CSV importieren", variant="primary")
            order_summary = gr.Markdown()
            orders_table = gr.Dataframe(interactive=True, label="Aufträge")
            geocode_btn = gr.Button("Adressen automatisch prüfen")
            address_table = gr.Dataframe(
                headers=["auftrag", "eingabe", "treffer", "confidence", "status", "lat", "lon", "fehler"],
                interactive=True,
                label="Adressprüfung",
            )
            address_note = gr.Markdown()
            save_address_btn = gr.Button("Geprüfte Adressen übernehmen", variant="primary")
            address_saved = gr.Markdown()

        with gr.Tab("2 · Flotte & Kapazität"):
            gr.Markdown('<div class="step-title">Fahrzeuge skalieren und Kapazität verteilen</div>')
            gr.Markdown("Zeilen hinzufügen, löschen oder Werte ändern. Palettenplätze und Nutzlast werden parallel geprüft.")
            vehicle_table = gr.Dataframe(value=vehicles_df(), interactive=True, label="Verfügbare Fahrzeuge")
            distribute_btn = gr.Button("Kapazität automatisch verteilen", variant="primary")
            allocation_status = gr.Markdown()
            assignments_table = gr.Dataframe(label="Auftragszuweisung")
            utilization_table = gr.Dataframe(label="Fahrzeugauslastung")

        with gr.Tab("3 · Forecast & Verkehr"):
            gr.Markdown('<div class="step-title">Routen, Live-Zuschlag und API-Kontrolle</div>')
            here_key = gr.Textbox(label="HERE API-Key (optional)", type="password")
            forecast_btn = gr.Button("Forecast berechnen", variant="primary")
            with gr.Row():
                with gr.Column(scale=3):
                    map_html = gr.HTML(label="Karte")
                with gr.Column(scale=1):
                    forecast_table = gr.Dataframe(label="Forecast je LKW")
            gr.Markdown("### API-Debug – Anfrageauswertung und Feldzuordnung")
            debug_json = gr.Code(language="json", label="Traffic-/Routing-Debug")

    import_btn.click(read_csv, upload, [orders_table, order_summary, orders_state])
    geocode_btn.click(geocode_orders, orders_state, [address_table, candidates_state, address_note])
    save_address_btn.click(save_addresses, [address_table, orders_state], [geo_state, address_saved])
    distribute_btn.click(distribute, [geo_state, vehicle_table], [assignments_table, utilization_table, assignments_state, allocation_status])
    forecast_btn.click(calculate, [assignments_state, vehicle_table, here_key], [map_html, forecast_table, debug_json])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )
