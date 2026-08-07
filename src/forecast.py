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
from src.traffic import AutobahnProvider, combine
from src.maps import build_map
from src.forecast import forecast_summary


@spaces.GPU(duration=1)
def zerogpu_startup_check():
    """ZeroGPU-Kompatibilitätsfunktion. Wird von der Logistik-App nicht verwendet."""
    return True


load_dotenv()
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))

SMALL_TEMPLATE = {
    "class": "14 t",
    "pallet_capacity": 18,
    "payload_kg": 6000,
}
LARGE_TEMPLATE = {
    "class": "40 t",
    "pallet_capacity": 33,
    "payload_kg": 24000,
}
COLORS = [
    "#2563eb", "#7c3aed", "#0891b2", "#dc2626", "#ea580c",
    "#16a34a", "#9333ea", "#0f766e", "#be123c", "#a16207",
    "#1d4ed8", "#15803d", "#b91c1c", "#6d28d9", "#0369a1",
]

def _read_json_records(payload: str) -> pd.DataFrame:
    if not payload or payload.strip() in ("", "[]", "null"):
        return pd.DataFrame()
    return pd.read_json(StringIO(payload), orient="records")

def _normalize_column_name(name: str) -> str:
    value = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "kundenname": "kunde",
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
        "postleitzahl": "plz",
        "stadt": "ort",
    }
    return aliases.get(value, value)

def _make_vehicle_rows(n_small: int, n_large: int) -> pd.DataFrame:
    rows = []
    idx = 0
    for i in range(1, int(n_small) + 1):
        rows.append({
            "vehicle_id": f"LKW-K{i:02d}",
            "class": SMALL_TEMPLATE["class"],
            "pallet_capacity": SMALL_TEMPLATE["pallet_capacity"],
            "payload_kg": SMALL_TEMPLATE["payload_kg"],
            "color": COLORS[idx % len(COLORS)],
            "available": True,
        })
        idx += 1
    for i in range(1, int(n_large) + 1):
        rows.append({
            "vehicle_id": f"LKW-G{i:02d}",
            "class": LARGE_TEMPLATE["class"],
            "pallet_capacity": LARGE_TEMPLATE["pallet_capacity"],
            "payload_kg": LARGE_TEMPLATE["payload_kg"],
            "color": COLORS[idx % len(COLORS)],
            "available": True,
        })
        idx += 1
    return pd.DataFrame(rows)

def update_fleet(n_small, n_large):
    n_small = max(0, int(n_small or 0))
    n_large = max(0, int(n_large or 0))
    df = _make_vehicle_rows(n_small, n_large)
    total_p = int(df["pallet_capacity"].sum()) if not df.empty else 0
    total_w = int(df["payload_kg"].sum()) if not df.empty else 0
    return df, f"**Flotte: {len(df)} Fahrzeuge · {total_p} Palettenplätze · {total_w:,} kg Nutzlast**"

def read_csv(file):
    if not file:
        raise gr.Error("Bitte zuerst eine CSV-Datei auswählen.")
    try:
        try:
            df = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(file, sep=None, engine="python", encoding="cp1252")

        if df.empty:
            raise gr.Error("Die CSV-Datei ist leer.")

        df.columns = [_normalize_column_name(c) for c in df.columns]
        required = ["auftrag","kunde","strasse","plz","ort","paletten","warengewicht_kg","service_min"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise gr.Error(
                "CSV-Struktur nicht erkannt.\n\nFehlende Spalten:\n"
                + ", ".join(missing)
                + "\n\nErkannte Spalten:\n"
                + ", ".join(df.columns)
            )

        for col in ["paletten","warengewicht_kg","service_min"]:
            cleaned = (
                df[col].astype(str).str.strip()
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(cleaned, errors="raise")

        df["auftrag"] = df["auftrag"].astype(str).str.strip()
        df["kunde"] = df["kunde"].astype(str).str.strip()
        df["strasse"] = df["strasse"].astype(str).str.strip()
        df["plz"] = df["plz"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        df["ort"] = df["ort"].astype(str).str.strip()

        df = normalize_orders(df)
        s = summarize_orders(df)

        return (
            df,
            f"✅ **{s['auftraege']} Aufträge · {s['paletten']} Paletten · {s['gewicht_kg']:,} kg · Ø {s['durchschnitt_kg_pro_palette']} kg/Palette**",
            df.to_json(orient="records", force_ascii=False),
        )
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"CSV konnte nicht importiert werden.\n\n{type(e).__name__}: {e}")

def geocode_orders(orders_json):
    df = _read_json_records(orders_json)
    if df.empty:
        raise gr.Error("Bitte zuerst eine CSV-Datei erfolgreich importieren.")

    geocoder = Geocoder(timeout=TIMEOUT)
    rows = []
    suggestion_rows = []
    candidates = {}

    for _, r in df.iterrows():
        try:
            hits = geocoder.search(r["adresse"], limit=5)
            err = ""
        except Exception as e:
            hits = []
            err = f"{type(e).__name__}: {e}"

        candidates[str(r["auftrag"])] = hits
        best = hits[0] if hits else None
        uncertain = not best or float(best.get("confidence", 0)) < 0.72

        rows.append({
            "auftrag": r["auftrag"],
            "eingabe": r["adresse"],
            "treffer": best["display_name"] if best else "",
            "confidence": best["confidence"] if best else 0,
            "status": "MANUELL PRÜFEN" if uncertain else "OK",
            "lat": best["lat"] if best else None,
            "lon": best["lon"] if best else None,
            "provider": best.get("provider", "") if best else "",
            "fehler": err,
        })

        for rank, hit in enumerate(hits, start=1):
            suggestion_rows.append({
                "auftrag": r["auftrag"],
                "rang": rank,
                "vorschlag": hit["display_name"],
                "confidence": hit.get("confidence", 0),
                "lat": hit["lat"],
                "lon": hit["lon"],
                "provider": hit.get("provider", ""),
            })

    note = (
        "Unsichere Treffer werden mit **MANUELL PRÜFEN** markiert. "
        "Unter der Haupttabelle stehen bis zu fünf Vorschläge je Auftrag. "
        "Kopiere bei Bedarf Vorschlag, lat und lon in die Haupttabelle."
    )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(suggestion_rows),
        json.dumps(candidates, ensure_ascii=False),
        note,
    )

def apply_suggestion(address_table, suggestion_table, selected_row):
    adr = pd.DataFrame(address_table)
    sug = pd.DataFrame(suggestion_table)
    if adr.empty or sug.empty:
        raise gr.Error("Keine Adressvorschläge vorhanden.")
    try:
        row_idx = int(selected_row)
    except Exception:
        raise gr.Error("Bitte die Zeilennummer des gewünschten Vorschlags eingeben.")

    if row_idx < 0 or row_idx >= len(sug):
        raise gr.Error(f"Zeilennummer muss zwischen 0 und {len(sug)-1} liegen.")

    chosen = sug.iloc[row_idx]
    mask = adr["auftrag"].astype(str) == str(chosen["auftrag"])
    if not mask.any():
        raise gr.Error("Passender Auftrag wurde in der Adresstabelle nicht gefunden.")

    adr.loc[mask, "treffer"] = chosen["vorschlag"]
    adr.loc[mask, "confidence"] = chosen["confidence"]
    adr.loc[mask, "lat"] = chosen["lat"]
    adr.loc[mask, "lon"] = chosen["lon"]
    adr.loc[mask, "provider"] = chosen["provider"]
    adr.loc[mask, "status"] = "MANUELL BESTÄTIGT"
    adr.loc[mask, "fehler"] = ""
    return adr, f"✅ Vorschlag für Auftrag {chosen['auftrag']} übernommen."


def geocode_depot(depot_address):
    depot_address = str(depot_address or "").strip()
    if not depot_address:
        raise gr.Error("Bitte eine Depotadresse eingeben.")

    geocoder = Geocoder(timeout=TIMEOUT)
    hits = geocoder.search(depot_address, limit=5)
    if not hits:
        raise gr.Error(
            "Depotadresse konnte nicht gefunden werden. "
            "Bitte Adresse genauer eingeben."
        )

    best = hits[0]
    state = json.dumps(
        {
            "address": depot_address,
            "display_name": best["display_name"],
            "lat": best["lat"],
            "lon": best["lon"],
            "confidence": best.get("confidence", 0),
            "provider": best.get("provider", ""),
        },
        ensure_ascii=False,
    )

    text = (
        f"✅ **Depot erkannt:** {best['display_name']}  \n"
        f"Provider: {best.get('provider','')} · "
        f"Confidence: {best.get('confidence',0)} · "
        f"Koordinaten: {best['lat']:.6f}, {best['lon']:.6f}"
    )
    return state, text


def _forecast_markdown(df):
    if df.empty:
        return "⚠️ Keine Forecast-Ergebnisse vorhanden."

    lines = ["## Forecast-Ergebnis", ""]
    for _, r in df.iterrows():
        lines.extend([
            f"### 🚚 {r['vehicle_id']}",
            f"- Distanz: **{r['distanz_km']} km**",
            f"- Basis-Fahrzeit: **{r['basis_fahrzeit_min']} min**",
            f"- Live-/Störungszuschlag: **{r['live_zuschlag_min']} min**",
            f"- Servicezeit: **{r['servicezeit_min']} min**",
            f"- Gesamtzeit: **{r['gesamtzeit_min']} min**",
            f"- Paletten: **{r['paletten']}**",
            f"- Gewicht: **{r['gewicht_kg']:,} kg**",
            f"- Traffic Score: **{r['traffic_score']}**",
            f"- Datenvertrauen: **{r['datenvertrauen_pct']} %**",
            "",
        ])
    return "\n".join(lines)


def save_addresses(address_table, orders_json):
    orders = _read_json_records(orders_json)
    adr = pd.DataFrame(address_table)
    if orders.empty:
        raise gr.Error("Keine Aufträge vorhanden.")
    required_cols = ["auftrag","treffer","lat","lon"]
    missing = [c for c in required_cols if c not in adr.columns]
    if missing:
        raise gr.Error("Adressprüfung unvollständig: " + ", ".join(missing))

    adr["auftrag"] = adr["auftrag"].astype(str)
    orders["auftrag"] = orders["auftrag"].astype(str)

    merged = orders.merge(
        adr[["auftrag","treffer","lat","lon"]],
        on="auftrag",
        how="left",
    )
    if merged[["lat","lon"]].isna().any().any():
        bad = merged[merged[["lat","lon"]].isna().any(axis=1)]["auftrag"].tolist()
        raise gr.Error("Fehlende Koordinaten bei: " + ", ".join(map(str, bad)))

    merged["lat"] = pd.to_numeric(merged["lat"], errors="raise")
    merged["lon"] = pd.to_numeric(merged["lon"], errors="raise")
    return merged.to_json(orient="records", force_ascii=False), "✅ Adressen übernommen."

def distribute(orders_geo_json, vehicle_table):
    orders = _read_json_records(orders_geo_json)
    vehicles = pd.DataFrame(vehicle_table)

    if orders.empty:
        raise gr.Error("Bitte zuerst die geprüften Adressen übernehmen.")
    if vehicles.empty:
        raise gr.Error("Keine Fahrzeuge vorhanden.")

    ass, util, warnings = distribute_orders(orders, vehicles)
    msg = "✅ Alle Aufträge zugewiesen." if not warnings else "⚠️ " + " | ".join(warnings)
    return ass, util, ass.to_json(orient="records", force_ascii=False), msg

def calculate(assign_json, vehicle_table, depot_json):
    ass = _read_json_records(assign_json)
    vehicles = pd.DataFrame(vehicle_table)
    if ass.empty:
        raise gr.Error("Bitte zuerst die Kapazität verteilen.")

    if not depot_json:
        raise gr.Error("Bitte zuerst die Depotadresse prüfen.")

    try:
        depot = json.loads(depot_json)
        depot_coord = (float(depot["lat"]), float(depot["lon"]))
    except Exception:
        raise gr.Error("Depotdaten sind ungültig. Bitte Depot erneut prüfen.")

    router = OSRMRouter(
        os.getenv("OSRM_BASE_URL","https://router.project-osrm.org"),
        TIMEOUT
    )

    routes, summaries, debug = [], [], []

    for vid, group in ass[ass.vehicle_id != "NICHT ZUGEWIESEN"].groupby("vehicle_id"):
        vmatch = vehicles[vehicles.vehicle_id == vid]
        if vmatch.empty:
            continue
        v = vmatch.iloc[0]

        stop_coords = [(float(x.lat), float(x.lon)) for _, x in group.iterrows()]
        coords = [depot_coord] + stop_coords + [depot_coord]

        try:
            route = router.route(coords)
        except Exception as e:
            debug.append({"vehicle_id": vid, "provider": "OSRM", "error": str(e)})
            continue

        try:
            autobahn = AutobahnProvider(timeout=TIMEOUT).analyze_route(
                route["geometry"], route["duration_s"]
            )
        except Exception as e:
            autobahn = {
                "provider": "Autobahn API",
                "available": False,
                "score": 0,
                "confidence": 0,
                "delay_s": 0,
                "error": str(e),
            }

        traffic = combine([autobahn], {"Autobahn API": 1.0})
        service = float(group.service_min.sum())
        summary = forecast_summary(
            route["distance_m"], route["duration_s"], traffic["delay_s"], service
        )
        summary.update({
            "vehicle_id": vid,
            "paletten": int(group.paletten.sum()),
            "gewicht_kg": int(group.gesamtgewicht_kg.sum()),
            "traffic_score": round(traffic["score"],1),
            "datenvertrauen_pct": round(traffic["confidence"]*100),
        })
        summaries.append(summary)

        stops = [{
            "lat": float(r.lat),
            "lon": float(r.lon),
            "kunde": r.kunde,
            "paletten": int(r.paletten),
            "gesamtgewicht_kg": int(r.gesamtgewicht_kg),
        } for _, r in group.iterrows()]

        routes.append({
            "vehicle_id": vid,
            "color": v["color"],
            "geometry": route["geometry"],
            "stops": stops,
        })

        debug.append({
            "vehicle_id": vid,
            "traffic": traffic,
            "autobahn_raw": autobahn,
            "osrm_summary": {
                "distance_m": route["distance_m"],
                "duration_s": route["duration_s"],
            },
            "note": "Aktuell kein kommerzieller Live-Traffic-Key konfiguriert.",
        })

    if not routes:
        raise gr.Error("Keine Route konnte erfolgreich berechnet werden.")

    forecast_df = pd.DataFrame(summaries)

    return (
        build_map(routes),
        forecast_df,
        _forecast_markdown(forecast_df),
        json.dumps(debug, ensure_ascii=False, indent=2, default=str),
    )


CSS = """
.step-title{font-size:1.35rem;font-weight:700;margin-bottom:.35rem}
.muted{color:#6b7280}
"""

with gr.Blocks(title="Logistik Forecast 4.9.2", css=CSS) as demo:
    gr.Markdown("# Logistik Forecast 4.9.2\nMehrstufige Python-/Gradio-Demo")

    orders_state = gr.State("[]")
    geo_state = gr.State("[]")
    assignments_state = gr.State("[]")
    candidates_state = gr.State("{}")
    depot_state = gr.State("")

    with gr.Tabs():
        with gr.Tab("1 · CSV & Adressen"):
            gr.Markdown('<div class="step-title">Aufträge importieren und Adressen prüfen</div>')
            upload = gr.File(label="CSV-Datei", file_types=[".csv"], type="filepath")
            import_btn = gr.Button("CSV importieren", variant="primary")
            order_summary = gr.Markdown()
            orders_table = gr.Dataframe(interactive=True, label="Aufträge")

            geocode_btn = gr.Button("Adressen automatisch prüfen")
            address_table = gr.Dataframe(
                headers=["auftrag","eingabe","treffer","confidence","status","lat","lon","provider","fehler"],
                interactive=True,
                label="Adressprüfung",
            )
            address_note = gr.Markdown()

            gr.Markdown("### Adressvorschläge")
            suggestion_table = gr.Dataframe(
                headers=["auftrag","rang","vorschlag","confidence","lat","lon","provider"],
                interactive=False,
                label="Vorschläge je Auftrag",
            )
            with gr.Row():
                suggestion_row = gr.Number(
                    label="Zeilennummer des Vorschlags (0 = erste Zeile)",
                    value=0,
                    precision=0,
                )
                apply_suggestion_btn = gr.Button("Vorschlag übernehmen")
            suggestion_status = gr.Markdown()

            save_address_btn = gr.Button("Geprüfte Adressen übernehmen", variant="primary")
            address_saved = gr.Markdown()

        with gr.Tab("2 · Flotte & Kapazität"):
            gr.Markdown('<div class="step-title">Fahrzeuge skalieren und Kapazität verteilen</div>')
            with gr.Row():
                small_count = gr.Number(label="Anzahl 14-t-LKW", value=3, precision=0, minimum=0)
                large_count = gr.Number(label="Anzahl 40-t-LKW", value=3, precision=0, minimum=0)
                update_fleet_btn = gr.Button("Flotte aktualisieren", variant="secondary")

            fleet_summary = gr.Markdown()
            vehicle_table = gr.Dataframe(
                value=_make_vehicle_rows(3,3),
                interactive=True,
                label="Verfügbare Fahrzeuge",
            )

            gr.Markdown(
                "Fahrzeuge können zusätzlich direkt in der Tabelle angepasst werden. "
                "Palettenplätze und Nutzlast werden parallel geprüft."
            )
            distribute_btn = gr.Button("Kapazität automatisch verteilen", variant="primary")
            allocation_status = gr.Markdown()
            assignments_table = gr.Dataframe(label="Auftragszuweisung")
            utilization_table = gr.Dataframe(label="Fahrzeugauslastung")

        with gr.Tab("3 · Forecast & Verkehr"):
            gr.Markdown('<div class="step-title">Routen, Verkehr und API-Kontrolle</div>')
            gr.Markdown(
                "OSRM liefert die Straßenroute. Die Autobahn-API ergänzt verfügbare "
                "offizielle Meldungen. Ohne zusätzliche Live-Traffic-Quelle bleibt "
                "der echte Verkehrszuschlag teilweise 0."
            )

            gr.Markdown("### Depot / Tourstart")
            depot_address = gr.Textbox(
                label="Depotadresse",
                placeholder="z. B. Mercedesstraße 1, 70372 Stuttgart",
            )
            depot_btn = gr.Button("Depotadresse prüfen")
            depot_status = gr.Markdown()

            forecast_btn = gr.Button("Forecast berechnen", variant="primary")

            forecast_summary_md = gr.Markdown()

            with gr.Row():
                with gr.Column(scale=3):
                    map_html = gr.HTML(label="Karte")
                with gr.Column(scale=1):
                    forecast_table = gr.Dataframe(label="Forecast je LKW")

            with gr.Accordion("API-Debug – technische Details", open=False):
                debug_json = gr.Code(language="json", label="Traffic-/Routing-Debug")

    import_btn.click(
        read_csv,
        upload,
        [orders_table, order_summary, orders_state],
    )
    geocode_btn.click(
        geocode_orders,
        orders_state,
        [address_table, suggestion_table, candidates_state, address_note],
    )
    apply_suggestion_btn.click(
        apply_suggestion,
        [address_table, suggestion_table, suggestion_row],
        [address_table, suggestion_status],
    )
    save_address_btn.click(
        save_addresses,
        [address_table, orders_state],
        [geo_state, address_saved],
    )
    update_fleet_btn.click(
        update_fleet,
        [small_count, large_count],
        [vehicle_table, fleet_summary],
    )
    distribute_btn.click(
        distribute,
        [geo_state, vehicle_table],
        [assignments_table, utilization_table, assignments_state, allocation_status],
    )
    depot_btn.click(
        geocode_depot,
        depot_address,
        [depot_state, depot_status],
    )

    forecast_btn.click(
        calculate,
        [assignments_state, vehicle_table, depot_state],
        [map_html, forecast_table, forecast_summary_md, debug_json],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT","7860")),
        show_error=True,
    )
