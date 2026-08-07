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

def _address_visible_df(full_df: pd.DataFrame) -> pd.DataFrame:
    """Nur die für Disposition sichtbaren Spalten – keine Koordinaten."""
    if full_df.empty:
        return pd.DataFrame(
            columns=["auftrag", "kunde", "adresse", "status", "treffer", "sicherheit"]
        )

    out = pd.DataFrame({
        "auftrag": full_df["auftrag"].astype(str),
        "kunde": full_df["kunde"].astype(str),
        "adresse": full_df["eingabe"].astype(str),
        "status": full_df["status"].astype(str),
        "treffer": full_df["treffer"].fillna("").astype(str),
        "sicherheit": (
            pd.to_numeric(full_df["confidence"], errors="coerce")
            .fillna(0)
            .mul(100)
            .round()
            .astype(int)
            .astype(str)
            + " %"
        ),
    })
    return out


def _address_status_text(full_df: pd.DataFrame) -> str:
    total = len(full_df)
    if total == 0:
        return "Noch keine Adressen geprüft."

    ok = int(full_df["status"].isin(["OK", "MANUELL BESTÄTIGT"]).sum())
    manual = int((full_df["status"] == "MANUELL PRÜFEN").sum())
    failed = int((full_df["status"] == "NICHT GEFUNDEN").sum())

    return (
        f"**{total} Adressen geprüft**  \n"
        f"✅ {ok} bestätigt · ⚠️ {manual} manuell prüfen"
        + (f" · ❌ {failed} ohne Treffer" if failed else "")
    )


def geocode_orders(orders_json):
    orders = _read_json_records(orders_json)
    if orders.empty:
        raise gr.Error("Bitte zuerst eine CSV-Datei erfolgreich importieren.")

    geocoder = Geocoder(timeout=TIMEOUT)
    rows = []
    candidates = {}

    for _, r in orders.iterrows():
        try:
            hits = geocoder.search(r["adresse"], limit=5)
            error_text = ""
        except Exception as exc:
            hits = []
            error_text = f"{type(exc).__name__}: {exc}"

        order_id = str(r["auftrag"])
        candidates[order_id] = hits
        best = hits[0] if hits else None

        if best:
            confidence = float(best.get("confidence", 0) or 0)
            status = "OK" if confidence >= 0.72 else "MANUELL PRÜFEN"
        else:
            confidence = 0.0
            status = "NICHT GEFUNDEN"

        rows.append({
            "auftrag": order_id,
            "kunde": str(r["kunde"]),
            "eingabe": str(r["adresse"]),
            "treffer": best.get("display_name", "") if best else "",
            "confidence": confidence,
            "status": status,
            "lat": best.get("lat") if best else None,
            "lon": best.get("lon") if best else None,
            "provider": best.get("provider", "") if best else "",
            "fehler": error_text,
        })

    full_df = pd.DataFrame(rows)

    return (
        _address_visible_df(full_df),
        full_df.to_json(orient="records", force_ascii=False),
        json.dumps(candidates, ensure_ascii=False),
        _address_status_text(full_df),
        "Tippe auf eine Zeile, um sie unten zu prüfen oder einen Vorschlag zu bestätigen.",
    )

def open_address_suggestions(
    visible_table,
    address_state_json,
    candidates_json,
    evt: gr.SelectData,
):
    """
    HTML-Ablauf:
    Zeile antippen -> 'Ausgewählte Zeile' -> Adressvorschläge als Auswahl.
    """
    visible = pd.DataFrame(visible_table)
    full = _read_json_records(address_state_json)

    if visible.empty or full.empty:
        raise gr.Error("Keine geprüften Adressen vorhanden.")

    try:
        row_index = evt.index[0] if isinstance(evt.index, (tuple, list)) else int(evt.index)
    except Exception:
        raise gr.Error("Die ausgewählte Zeile konnte nicht erkannt werden.")

    if row_index < 0 or row_index >= len(visible):
        raise gr.Error("Ungültige Zeilenauswahl.")

    visible_row = visible.iloc[row_index]
    order_id = str(visible_row["auftrag"])

    full_match = full[full["auftrag"].astype(str) == order_id]
    if full_match.empty:
        raise gr.Error("Die ausgewählte Adresse wurde intern nicht gefunden.")
    current = full_match.iloc[0]

    try:
        all_candidates = json.loads(candidates_json or "{}")
    except Exception:
        all_candidates = {}

    hits = all_candidates.get(order_id, [])
    choices = [str(h.get("display_name", "")).strip() for h in hits]
    choices = [c for c in choices if c]
    choices = list(dict.fromkeys(choices))

    # Wenn der aktuelle Treffer vorhanden ist, oben anzeigen.
    current_hit = str(current.get("treffer", "") or "").strip()
    if current_hit and current_hit not in choices:
        choices.insert(0, current_hit)

    selected_info = (
        f"### Ausgewählte Zeile\n"
        f"**Auftrag:** {order_id}  \n"
        f"**Kunde:** {current.get('kunde', '')}  \n"
        f"**CSV-Adresse:** {current.get('eingabe', '')}  \n"
        f"**Status:** {current.get('status', '')}"
    )

    if not choices:
        return (
            order_id,
            json.dumps(hits, ensure_ascii=False),
            selected_info,
            gr.update(
                choices=[],
                value=None,
                label="Welche Adresse ist richtig?",
                interactive=False,
            ),
            "Für diese Adresse wurde kein Vorschlag gefunden.",
        )

    return (
        order_id,
        json.dumps(hits, ensure_ascii=False),
        selected_info,
        gr.update(
            choices=choices,
            value=choices[0],
            label="Welche Adresse ist richtig?",
            interactive=True,
        ),
        "",
    )


def confirm_address_suggestion(
    address_state_json,
    selected_order_id,
    selected_hits_json,
    selected_address,
):
    """
    Nutzer bestätigt ausschließlich die lesbare Adresse.
    lat/lon/provider werden anhand des gewählten Vorschlags intern übernommen.
    """
    full = _read_json_records(address_state_json)

    if full.empty:
        raise gr.Error("Keine geprüften Adressen vorhanden.")

    if not selected_order_id:
        raise gr.Error("Bitte zuerst eine Zeile in der Adressliste antippen.")

    selected_address = str(selected_address or "").strip()
    if not selected_address:
        raise gr.Error("Bitte zuerst einen Adressvorschlag auswählen.")

    try:
        hits = json.loads(selected_hits_json or "[]")
    except Exception:
        hits = []

    chosen = next(
        (
            h for h in hits
            if str(h.get("display_name", "")).strip() == selected_address
        ),
        None,
    )

    # Der aktuelle automatische Treffer kann schon in der Liste stehen,
    # selbst wenn er nicht mehr in hits enthalten ist.
    if chosen is None:
        current_match = full[full["auftrag"].astype(str) == str(selected_order_id)]
        if not current_match.empty:
            current = current_match.iloc[0]
            if str(current.get("treffer", "") or "").strip() == selected_address:
                chosen = {
                    "display_name": selected_address,
                    "lat": current.get("lat"),
                    "lon": current.get("lon"),
                    "confidence": current.get("confidence", 0),
                    "provider": current.get("provider", ""),
                }

    if chosen is None:
        raise gr.Error("Der gewählte Adressvorschlag konnte intern nicht zugeordnet werden.")

    if chosen.get("lat") is None or chosen.get("lon") is None:
        raise gr.Error("Der gewählte Vorschlag besitzt keine gültigen Koordinaten.")

    mask = full["auftrag"].astype(str) == str(selected_order_id)
    if not mask.any():
        raise gr.Error("Auftrag wurde nicht gefunden.")

    full.loc[mask, "treffer"] = chosen.get("display_name", "")
    full.loc[mask, "confidence"] = float(chosen.get("confidence", 0) or 0)
    full.loc[mask, "lat"] = chosen.get("lat")
    full.loc[mask, "lon"] = chosen.get("lon")
    full.loc[mask, "provider"] = chosen.get("provider", "")
    full.loc[mask, "status"] = "MANUELL BESTÄTIGT"
    full.loc[mask, "fehler"] = ""

    return (
        _address_visible_df(full),
        full.to_json(orient="records", force_ascii=False),
        _address_status_text(full),
        (
            f"✅ **Adresse bestätigt**  \n"
            f"Auftrag {selected_order_id}: {chosen.get('display_name', '')}"
        ),
    )

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


def save_addresses(address_state_json, orders_json):
    orders = _read_json_records(orders_json)
    adr = _read_json_records(address_state_json)
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
            gr.Markdown(
                '<div class="step-title">CSV importieren und Adressen prüfen</div>'
            )

            upload = gr.File(
                label="Kunden-CSV auswählen",
                file_types=[".csv"],
                type="filepath",
            )
            import_btn = gr.Button(
                "CSV importieren",
                variant="primary",
            )

            order_summary = gr.Markdown()
            orders_table = gr.Dataframe(
                interactive=False,
                label="Importierte Aufträge",
                wrap=True,
            )

            geocode_btn = gr.Button(
                "CSV importieren und Adressen prüfen",
                variant="primary",
            )

            gr.Markdown("### Prüfstatus")
            address_status = gr.Markdown("Noch keine Adressen geprüft.")
            address_note = gr.Markdown()

            # Vollständige Geocoding-Daten bleiben unsichtbar im State.
            address_state = gr.State("[]")

            address_table = gr.Dataframe(
                headers=[
                    "auftrag",
                    "kunde",
                    "adresse",
                    "status",
                    "treffer",
                    "sicherheit",
                ],
                interactive=False,
                label="Adressprüfung",
                wrap=True,
            )

            gr.Markdown("---")

            selected_order_state = gr.State("")
            selected_hits_state = gr.State("[]")

            selected_address_info = gr.Markdown(
                "### Ausgewählte Zeile\nNoch keine Zeile ausgewählt."
            )

            address_choice = gr.Radio(
                choices=[],
                value=None,
                label="Welche Adresse ist richtig?",
                interactive=False,
            )

            address_choice_note = gr.Markdown()

            confirm_suggestion_btn = gr.Button(
                "Ausgewählte Adresse bestätigen",
                variant="primary",
            )
            suggestion_status = gr.Markdown()

            save_address_btn = gr.Button(
                "Adressprüfung abschließen",
                variant="secondary",
            )
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
        [
            address_table,
            address_state,
            candidates_state,
            address_status,
            address_note,
        ],
    )
    address_table.select(
        open_address_suggestions,
        [address_table, address_state, candidates_state],
        [
            selected_order_state,
            selected_hits_state,
            selected_address_info,
            address_choice,
            address_choice_note,
        ],
    )

    confirm_suggestion_btn.click(
        confirm_address_suggestion,
        [
            address_state,
            selected_order_state,
            selected_hits_state,
            address_choice,
        ],
        [
            address_table,
            address_state,
            address_status,
            suggestion_status,
        ],
    )

    save_address_btn.click(
        save_addresses,
        [address_state, orders_state],
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
