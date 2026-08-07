from __future__ import annotations

import json
import os
from io import StringIO

import gradio as gr
import pandas as pd
import spaces
from dotenv import load_dotenv

from src.capacity import normalize_orders, summarize_orders
from src.clustering import cluster_orders
from src.geocoding import Geocoder
from src.models import Vehicle
from src.routing import OSRMRouter
from src.traffic import AutobahnProvider, combine
from src.maps import build_map
from src.forecast import forecast_summary


@spaces.GPU(duration=1)
def zerogpu_startup_check():
    """ZeroGPU-Kompatibilitätsfunktion; die Logistikberechnung läuft auf CPU."""
    return True


load_dotenv()
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))

COLORS = [
    "#2563eb", "#7c3aed", "#0891b2", "#dc2626", "#ea580c",
    "#16a34a", "#9333ea", "#0f766e", "#be123c", "#a16207",
    "#1d4ed8", "#15803d", "#b91c1c", "#6d28d9", "#0369a1",
]

SMALL = {"class": "14 t", "pallet_capacity": 18, "payload_kg": 6000}
LARGE = {"class": "40 t", "pallet_capacity": 33, "payload_kg": 24000}


def _read_json(payload: str) -> pd.DataFrame:
    if not payload or str(payload).strip() in {"", "[]", "null"}:
        return pd.DataFrame()
    return pd.read_json(StringIO(payload), orient="records")


def _normalize_column_name(name):
    v = str(name).strip().lower().replace(" ", "_").replace("-", "_")
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
    return aliases.get(v, v)


def make_fleet(n_small=3, n_large=3):
    rows = []
    color_idx = 0
    for i in range(1, int(n_small) + 1):
        rows.append({
            "vehicle_id": f"LKW-K{i:02d}",
            **SMALL,
            "color": COLORS[color_idx % len(COLORS)],
            "available": True,
        })
        color_idx += 1
    for i in range(1, int(n_large) + 1):
        rows.append({
            "vehicle_id": f"LKW-G{i:02d}",
            **LARGE,
            "color": COLORS[color_idx % len(COLORS)],
            "available": True,
        })
        color_idx += 1
    return pd.DataFrame(rows)


def update_fleet(n_small, n_large):
    df = make_fleet(max(0, int(n_small or 0)), max(0, int(n_large or 0)))
    if df.empty:
        return df, "⚠️ Keine Fahrzeuge konfiguriert."
    return (
        df,
        f"**{len(df)} Fahrzeuge · {int(df.pallet_capacity.sum())} Palettenplätze · "
        f"{int(df.payload_kg.sum()):,} kg Nutzlast**",
    )


def read_csv(file):
    if not file:
        raise gr.Error("Bitte CSV auswählen.")

    try:
        try:
            df = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(file, sep=None, engine="python", encoding="cp1252")

        if df.empty:
            raise gr.Error("CSV ist leer.")

        df.columns = [_normalize_column_name(c) for c in df.columns]
        df = normalize_orders(df)
        s = summarize_orders(df)

        return (
            df,
            f"✅ **{s['auftraege']} Aufträge · {s['paletten']} Paletten · "
            f"{s['gewicht_kg']:,} kg**",
            df.to_json(orient="records", force_ascii=False),
        )
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"CSV konnte nicht importiert werden: {type(exc).__name__}: {exc}")


def geocode_all(orders_json):
    orders = _read_json(orders_json)
    if orders.empty:
        raise gr.Error("Bitte zuerst CSV importieren.")

    geocoder = Geocoder(TIMEOUT)
    full_rows = []
    candidates = {}

    for _, row in orders.iterrows():
        hits = geocoder.search(row["adresse"], 5)
        oid = str(row["auftrag"])
        candidates[oid] = hits
        best = hits[0] if hits else None

        if best:
            confidence = float(best.get("confidence", 0))
            status = "OK" if confidence >= 0.72 else "MANUELL PRÜFEN"
        else:
            confidence = 0.0
            status = "MANUELL PRÜFEN"

        full_rows.append({
            "auftrag": oid,
            "kunde": str(row["kunde"]),
            "eingabe": str(row["adresse"]),
            "treffer": best.get("display_name", "") if best else "",
            "confidence": confidence,
            "status": status,
            "lat": best.get("lat") if best else None,
            "lon": best.get("lon") if best else None,
            "provider": best.get("provider", "") if best else "",
        })

    full = pd.DataFrame(full_rows)
    visible = address_visible(full)
    status = address_status(full)
    return (
        visible,
        full.to_json(orient="records", force_ascii=False),
        json.dumps(candidates, ensure_ascii=False),
        status,
    )


def address_visible(full):
    if full.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "auftrag": full["auftrag"].astype(str),
        "kunde": full["kunde"].astype(str),
        "adresse": full["eingabe"].astype(str),
        "status": full["status"].astype(str),
        "treffer": full["treffer"].fillna("").astype(str),
        "sicherheit": (
            pd.to_numeric(full["confidence"], errors="coerce")
            .fillna(0).mul(100).round().astype(int).astype(str) + " %"
        ),
    })


def address_status(full):
    if full.empty:
        return "Noch keine Adressen geprüft."
    open_count = int((full["status"] == "MANUELL PRÜFEN").sum())
    confirmed = len(full) - open_count
    return (
        f"**{len(full)} Adressen geprüft · ✅ {confirmed} automatisch/bestätigt · "
        f"⚠️ {open_count} offen**"
    )


def _next_open(full):
    p = full[full["status"].astype(str) == "MANUELL PRÜFEN"]
    return None if p.empty else p.iloc[0]


def prepare_review(address_state_json, candidates_json):
    full = _read_json(address_state_json)
    row = _next_open(full)
    if row is None:
        return (
            "",
            "[]",
            "✅ Alle Adressen sind eindeutig.",
            gr.update(choices=[], value=None, interactive=False),
        )

    oid = str(row["auftrag"])
    candidates = json.loads(candidates_json or "{}")
    hits = candidates.get(oid, [])
    choices = [h.get("display_name", "") for h in hits if h.get("display_name")]
    current = str(row.get("treffer", "") or "")
    if current and current not in choices:
        choices.insert(0, current)

    return (
        oid,
        json.dumps(hits, ensure_ascii=False),
        f"### Jetzt prüfen\n**{row['kunde']}**  \n{row['eingabe']}",
        gr.update(
            choices=choices,
            value=choices[0] if choices else None,
            interactive=bool(choices),
        ),
    )


def confirm_review(address_state_json, candidates_json, oid, hits_json, selected):
    full = _read_json(address_state_json)
    if not oid:
        raise gr.Error("Keine offene Adresse ausgewählt.")
    if not selected:
        raise gr.Error("Bitte Adresse auswählen.")

    hits = json.loads(hits_json or "[]")
    chosen = next((h for h in hits if h.get("display_name") == selected), None)

    if chosen is None:
        current = full[full["auftrag"].astype(str) == str(oid)].iloc[0]
        if str(current.get("treffer", "")) == str(selected):
            chosen = {
                "display_name": selected,
                "lat": current["lat"],
                "lon": current["lon"],
                "confidence": current["confidence"],
                "provider": current["provider"],
            }

    if chosen is None or chosen.get("lat") is None or chosen.get("lon") is None:
        raise gr.Error("Vorschlag konnte nicht übernommen werden.")

    mask = full["auftrag"].astype(str) == str(oid)
    full.loc[mask, "treffer"] = chosen["display_name"]
    full.loc[mask, "lat"] = chosen["lat"]
    full.loc[mask, "lon"] = chosen["lon"]
    full.loc[mask, "confidence"] = chosen.get("confidence", 0)
    full.loc[mask, "provider"] = chosen.get("provider", "")
    full.loc[mask, "status"] = "MANUELL BESTÄTIGT"

    next_row = _next_open(full)
    if next_row is None:
        next_oid, next_hits, info, choice = (
            "",
            "[]",
            "✅ **Adressprüfung abgeschlossen.**",
            gr.update(choices=[], value=None, interactive=False),
        )
    else:
        next_oid = str(next_row["auftrag"])
        all_candidates = json.loads(candidates_json or "{}")
        hits2 = all_candidates.get(next_oid, [])
        choices = [h.get("display_name", "") for h in hits2 if h.get("display_name")]
        current2 = str(next_row.get("treffer", "") or "")
        if current2 and current2 not in choices:
            choices.insert(0, current2)
        next_hits = json.dumps(hits2, ensure_ascii=False)
        info = f"### Jetzt prüfen\n**{next_row['kunde']}**  \n{next_row['eingabe']}"
        choice = gr.update(
            choices=choices,
            value=choices[0] if choices else None,
            interactive=bool(choices),
        )

    return (
        address_visible(full),
        full.to_json(orient="records", force_ascii=False),
        address_status(full),
        f"✅ Bestätigt: **{selected}**",
        next_oid,
        next_hits,
        info,
        choice,
    )


def save_addresses(address_state_json, orders_json):
    full = _read_json(address_state_json)
    orders = _read_json(orders_json)

    if full.empty:
        raise gr.Error("Keine Adressprüfung vorhanden.")
    if (full["status"] == "MANUELL PRÜFEN").any():
        raise gr.Error("Es sind noch Adressen offen.")

    merged = orders.merge(
        full[["auftrag", "treffer", "lat", "lon"]],
        on="auftrag",
        how="left",
    )
    if merged[["lat", "lon"]].isna().any().any():
        raise gr.Error("Mindestens eine Adresse hat keine Koordinaten.")

    return (
        merged.to_json(orient="records", force_ascii=False),
        "✅ Adressen abgeschlossen. Weiter zur Flotte und Clusterbildung.",
    )


def search_depot(address):
    address = str(address or "").strip()
    if not address:
        raise gr.Error("Depotadresse eingeben.")

    hits = Geocoder(TIMEOUT).search(address, 5)
    if not hits:
        return (
            "[]",
            gr.update(choices=[], value=None, interactive=False),
            "⚠️ Kein Depot-Treffer.",
        )

    choices = [h["display_name"] for h in hits]
    return (
        json.dumps(hits, ensure_ascii=False),
        gr.update(choices=choices, value=choices[0], interactive=True),
        "Depotvorschlag gefunden – bitte bestätigen.",
    )


def confirm_depot(input_address, hits_json, selected):
    hits = json.loads(hits_json or "[]")
    chosen = next((h for h in hits if h.get("display_name") == selected), None)
    if not chosen:
        raise gr.Error("Depotvorschlag auswählen.")

    state = {
        "address_input": input_address,
        "display_name": chosen["display_name"],
        "lat": chosen["lat"],
        "lon": chosen["lon"],
        "confidence": chosen.get("confidence", 0),
        "provider": chosen.get("provider", ""),
    }
    return json.dumps(state, ensure_ascii=False), f"✅ **Depot:** {chosen['display_name']}"


def create_clusters(geo_json, vehicle_table, depot_json):
    orders = _read_json(geo_json)
    vehicles = pd.DataFrame(vehicle_table)

    if orders.empty:
        raise gr.Error("Adressprüfung zuerst abschließen.")
    if vehicles.empty:
        raise gr.Error("Keine Fahrzeuge.")
    if not depot_json:
        raise gr.Error("Depot zuerst bestätigen.")

    depot = json.loads(depot_json)
    depot_coord = (float(depot["lat"]), float(depot["lon"]))

    assignments, summary, warnings = cluster_orders(
        orders, vehicles, depot_coord
    )

    msg = (
        "✅ Geografische Cluster erstellt. "
        "Jeder LKW erhält ein möglichst zusammenhängendes Liefergebiet."
    )
    if warnings:
        msg += "\n\n⚠️ " + " | ".join(warnings)

    visible = assignments[
        [
            "cluster_id", "vehicle_id", "auftrag", "kunde",
            "adresse", "paletten", "gesamtgewicht_kg"
        ]
    ].copy()

    return (
        visible,
        summary,
        assignments.to_json(orient="records", force_ascii=False),
        msg,
    )


def optimize_routes(cluster_json, vehicle_table, depot_json):
    clustered = _read_json(cluster_json)
    vehicles = pd.DataFrame(vehicle_table)

    if clustered.empty:
        raise gr.Error("Zuerst Cluster bilden.")
    depot = json.loads(depot_json)
    depot_coord = (float(depot["lat"]), float(depot["lon"]))

    router = OSRMRouter(
        os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org"),
        TIMEOUT,
    )

    route_rows = []
    routes = []
    debug = []

    active = clustered[clustered["vehicle_id"] != "NICHT ZUGEWIESEN"]

    for vid, group in active.groupby("vehicle_id"):
        vehicle = vehicles[vehicles["vehicle_id"].astype(str) == str(vid)]
        if vehicle.empty:
            continue
        v = vehicle.iloc[0]

        stops = [r.to_dict() for _, r in group.iterrows()]
        try:
            result = router.optimize_roundtrip(depot_coord, stops)
        except Exception as exc:
            debug.append({"vehicle_id": vid, "error": str(exc)})
            continue

        ordered = result["ordered_stops"]
        ordered_records = []
        for seq, stop in enumerate(ordered, 1):
            stop = dict(stop)
            stop["stopp_nr"] = seq
            ordered_records.append(stop)
            route_rows.append({
                "vehicle_id": vid,
                "cluster_id": stop.get("cluster_id", ""),
                "stopp_nr": seq,
                "auftrag": stop.get("auftrag", ""),
                "kunde": stop.get("kunde", ""),
                "adresse": stop.get("adresse", ""),
                "paletten": int(stop.get("paletten", 0)),
                "gewicht_kg": int(stop.get("gesamtgewicht_kg", 0)),
            })

        routes.append({
            "vehicle_id": vid,
            "cluster_id": str(group.iloc[0]["cluster_id"]),
            "color": v.get("color", "#2563eb"),
            "geometry": result["geometry"],
            "stops": [
                {
                    "auftrag": s.get("auftrag", ""),
                    "kunde": s.get("kunde", ""),
                    "lat": float(s["lat"]),
                    "lon": float(s["lon"]),
                    "paletten": int(s["paletten"]),
                    "gesamtgewicht_kg": int(s["gesamtgewicht_kg"]),
                }
                for s in ordered_records
            ],
            "distance_m": float(result["distance_m"]),
            "duration_s": float(result["duration_s"]),
            "optimizer": result.get("optimizer", "osrm-trip"),
            "service_min": float(group["service_min"].sum()),
        })

        debug.append({
            "vehicle_id": vid,
            "optimizer": result.get("optimizer"),
            "distance_m": result["distance_m"],
            "duration_s": result["duration_s"],
            "optimized_stop_count": len(ordered),
        })

    if not routes:
        raise gr.Error("Keine Route konnte optimiert werden.")

    choices = ["Alle Touren"] + [r["vehicle_id"] for r in routes]
    return (
        pd.DataFrame(route_rows),
        gr.update(choices=choices, value="Alle Touren", interactive=True),
        build_map(routes, depot),
        json.dumps(routes, ensure_ascii=False),
        json.dumps(debug, ensure_ascii=False, indent=2),
        "✅ Routen innerhalb der regionalen Cluster optimiert.",
    )


def render_route(routes_json, selected, depot_json):
    routes = json.loads(routes_json or "[]")
    depot = json.loads(depot_json or "{}")
    if selected and selected != "Alle Touren":
        routes = [r for r in routes if str(r["vehicle_id"]) == str(selected)]
    return build_map(routes, depot)


def calculate_forecast(routes_json):
    routes = json.loads(routes_json or "[]")
    if not routes:
        raise gr.Error("Zuerst Routen optimieren.")

    provider = AutobahnProvider(timeout=TIMEOUT)
    rows = []
    debug = []

    for r in routes:
        traffic_raw = provider.analyze_route(r["geometry"], r["duration_s"])
        traffic = combine([traffic_raw], {"Autobahn API": 1.0})

        summary = forecast_summary(
            r["distance_m"],
            r["duration_s"],
            traffic["delay_s"],
            r["service_min"],
        )
        summary.update({
            "vehicle_id": r["vehicle_id"],
            "cluster_id": r["cluster_id"],
            "stopps": len(r["stops"]),
            "paletten": sum(int(s["paletten"]) for s in r["stops"]),
            "gewicht_kg": sum(int(s["gesamtgewicht_kg"]) for s in r["stops"]),
            "traffic_score": round(traffic["score"], 1),
            "datenvertrauen_pct": round(traffic["confidence"] * 100),
        })
        rows.append(summary)
        debug.append({
            "vehicle_id": r["vehicle_id"],
            "traffic": traffic,
            "hinweis": (
                "Verkehr wird bewusst erst NACH der Routenoptimierung geprüft. "
                "Die Autobahn-API-Routenmatchlogik ist der nächste Ausbauschritt."
            ),
        })

    df = pd.DataFrame(rows)
    lines = ["## Forecast nach Routenoptimierung", ""]
    for _, r in df.iterrows():
        lines += [
            f"### 🚚 {r['vehicle_id']} · {r['cluster_id']}",
            f"- Stopps: **{r['stopps']}**",
            f"- Distanz: **{r['distanz_km']} km**",
            f"- Basis-Fahrzeit: **{r['basis_fahrzeit_min']} min**",
            f"- Störungs-/Live-Zuschlag: **{r['live_zuschlag_min']} min**",
            f"- Servicezeit: **{r['servicezeit_min']} min**",
            f"- Gesamtzeit: **{r['gesamtzeit_min']} min**",
            "",
        ]

    return df, "\n".join(lines), json.dumps(debug, ensure_ascii=False, indent=2, default=str)


CSS = """
html, body {
  background: #0f1117 !important;
}

.gradio-container,
.gradio-container > .main {
  background: #0f1117 !important;
  color: #f3f4f6 !important;
}

.gradio-container {
  min-height: 100vh;
}

.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .prose h4,
.gradio-container label,
.gradio-container span {
  color: #f3f4f6;
}

.step-title{font-size:1.35rem;font-weight:750;margin-bottom:.35rem}
.process{padding:.8rem 1rem;border:1px solid #374151;border-radius:12px;margin:.3rem 0}

#tour-map iframe{
  width:100%!important;
  min-height:560px!important;
  height:62vh!important;
  border-radius:12px
}
#tour-map{min-height:560px}

@media(max-width:768px){
  .gradio-container {
    padding-left: 0 !important;
    padding-right: 0 !important;
  }
  #tour-map iframe{min-height:500px!important;height:58vh!important}
}
"""

with gr.Blocks(title="Logistik Forecast 4.9.3", css=CSS) as demo:
    gr.Markdown(
        "# Logistik Forecast 4.9.3\n"
        "**Neue feste Prozessstruktur:** Adressen → Flotte → regionale Cluster → "
        "Routenoptimierung → Verkehr/Forecast"
    )

    orders_state = gr.State("[]")
    address_state = gr.State("[]")
    candidates_state = gr.State("{}")
    geo_state = gr.State("[]")
    depot_hits_state = gr.State("[]")
    depot_state = gr.State("")
    cluster_state = gr.State("[]")
    routes_state = gr.State("[]")
    selected_order_state = gr.State("")
    selected_hits_state = gr.State("[]")

    with gr.Tabs():
        with gr.Tab("1 · Aufträge & Adressen"):
            gr.Markdown('<div class="step-title">1. CSV importieren und Adressen einmal sauber prüfen</div>')
            upload = gr.File(label="CSV-Datei", file_types=[".csv"], type="filepath")
            import_btn = gr.Button("CSV importieren", variant="primary")
            order_summary = gr.Markdown()
            orders_table = gr.Dataframe(label="Aufträge", interactive=False, wrap=True)

            geocode_btn = gr.Button("Adressen automatisch prüfen")
            address_status_md = gr.Markdown()
            address_table = gr.Dataframe(
                headers=["auftrag","kunde","adresse","status","treffer","sicherheit"],
                label="Adressprüfung",
                interactive=False,
                wrap=True,
            )

            gr.Markdown("### Nur unsichere Adressen")
            review_info = gr.Markdown("Noch keine Prüfung gestartet.")
            address_choice = gr.Radio(
                choices=[], label="Welche Adresse ist richtig?", interactive=False
            )
            confirm_review_btn = gr.Button("Adresse bestätigen", variant="primary")
            confirm_status = gr.Markdown()
            finish_addresses_btn = gr.Button("Adressprüfung abschließen")
            finish_status = gr.Markdown()

        with gr.Tab("2 · Depot, Flotte & Regionen"):
            gr.Markdown('<div class="step-title">2. Depot und Fahrzeugkapazität festlegen</div>')

            depot_input = gr.Textbox(
                label="Depotadresse",
                placeholder="z. B. Mercedesstraße 1, 70372 Stuttgart",
            )
            depot_search_btn = gr.Button("Depot automatisch prüfen")
            depot_choice = gr.Radio(
                choices=[], label="Gefundene Depotadresse", interactive=False
            )
            depot_search_status = gr.Markdown()
            depot_confirm_btn = gr.Button("Depot bestätigen", variant="primary")
            depot_status = gr.Markdown()

            gr.Markdown("### Flotte")
            with gr.Row():
                small_count = gr.Number(label="14-t-LKW", value=3, minimum=0, precision=0)
                large_count = gr.Number(label="40-t-LKW", value=3, minimum=0, precision=0)
                fleet_btn = gr.Button("Flotte aktualisieren")
            fleet_status = gr.Markdown()
            vehicle_table = gr.Dataframe(
                value=make_fleet(3, 3),
                label="Verfügbare Fahrzeuge",
                interactive=True,
                wrap=True,
            )

            gr.Markdown(
                "### 3. Geografische Cluster bilden\n"
                "Jetzt werden nahe Stopps zu zusammenhängenden Liefergebieten gebündelt. "
                "Kapazität und Nutzlast bleiben harte Grenzen."
            )
            cluster_btn = gr.Button("Regionen bilden & LKW zuweisen", variant="primary")
            cluster_status = gr.Markdown()
            cluster_summary = gr.Dataframe(label="Regionen / LKW-Auslastung", wrap=True)
            cluster_assignments = gr.Dataframe(label="Aufträge nach Region", wrap=True)

        with gr.Tab("3 · Routenoptimierung"):
            gr.Markdown(
                '<div class="step-title">4. Stopp-Reihenfolge je LKW optimieren</div>'
            )
            gr.Markdown(
                "Erst jetzt wird innerhalb jedes regionalen Clusters die schnellere "
                "Stopp-Reihenfolge gesucht. Verkehr beeinflusst diesen Schritt noch nicht."
            )
            optimize_btn = gr.Button("LKW-Routen optimieren", variant="primary")
            optimize_status = gr.Markdown()
            optimized_stops = gr.Dataframe(label="Optimierte Stopp-Reihenfolge", wrap=True)

            route_selector = gr.Dropdown(
                choices=[], label="Tour anzeigen", interactive=False
            )
            map_html = gr.HTML(elem_id="tour-map")
            with gr.Accordion("Routing-Debug", open=False):
                routing_debug = gr.Code(language="json")

        with gr.Tab("4 · Verkehr & Forecast"):
            gr.Markdown(
                '<div class="step-title">5. Erst die fertige Route auf Störungen prüfen</div>'
            )
            gr.Markdown(
                "Hier werden die bereits optimierten Touren auf Verkehr, Baustellen, "
                "Ferien und weitere Einflussfaktoren geprüft. Dadurch gibt es kein "
                "Ping-Pong mehr zwischen Verteilung und Forecast."
            )
            forecast_btn = gr.Button("Verkehr prüfen & Forecast berechnen", variant="primary")
            forecast_md = gr.Markdown()
            forecast_table = gr.Dataframe(label="Forecast je LKW", wrap=True)
            with gr.Accordion("Traffic-Debug", open=False):
                traffic_debug = gr.Code(language="json")

    import_btn.click(
        read_csv, upload, [orders_table, order_summary, orders_state]
    )

    geocode_event = geocode_btn.click(
        geocode_all,
        orders_state,
        [address_table, address_state, candidates_state, address_status_md],
    )
    geocode_event.then(
        prepare_review,
        [address_state, candidates_state],
        [selected_order_state, selected_hits_state, review_info, address_choice],
    )

    confirm_review_btn.click(
        confirm_review,
        [address_state, candidates_state, selected_order_state, selected_hits_state, address_choice],
        [
            address_table, address_state, address_status_md, confirm_status,
            selected_order_state, selected_hits_state, review_info, address_choice,
        ],
    )

    finish_addresses_btn.click(
        save_addresses,
        [address_state, orders_state],
        [geo_state, finish_status],
    )

    depot_search_btn.click(
        search_depot,
        depot_input,
        [depot_hits_state, depot_choice, depot_search_status],
    )
    depot_confirm_btn.click(
        confirm_depot,
        [depot_input, depot_hits_state, depot_choice],
        [depot_state, depot_status],
    )

    fleet_btn.click(
        update_fleet,
        [small_count, large_count],
        [vehicle_table, fleet_status],
    )

    cluster_btn.click(
        create_clusters,
        [geo_state, vehicle_table, depot_state],
        [cluster_assignments, cluster_summary, cluster_state, cluster_status],
    )

    optimize_btn.click(
        optimize_routes,
        [cluster_state, vehicle_table, depot_state],
        [
            optimized_stops, route_selector, map_html, routes_state,
            routing_debug, optimize_status,
        ],
    )

    route_selector.change(
        render_route,
        [routes_state, route_selector, depot_state],
        map_html,
    )

    forecast_btn.click(
        calculate_forecast,
        routes_state,
        [forecast_table, forecast_md, traffic_debug],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )
