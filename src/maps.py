from __future__ import annotations
import folium


def build_map(routes: list[dict], depot: dict | None = None, center=(51.0, 10.0)) -> str:
    m = folium.Map(location=center, zoom_start=6, control_scale=True)
    bounds = []

    if depot and depot.get("lat") is not None and depot.get("lon") is not None:
        d = [float(depot["lat"]), float(depot["lon"])]
        folium.Marker(
            d,
            tooltip="Depot",
            popup=f"<b>Depot</b><br>{depot.get('display_name', '')}",
            icon=folium.Icon(color="black", icon="home"),
        ).add_to(m)
        bounds.append(tuple(d))

    for route in routes:
        geom = route.get("geometry", [])
        color = route.get("color", "#2563eb")
        vid = route.get("vehicle_id", "LKW")
        if not geom:
            continue

        bounds += geom

        # Dunkle Kontur + Fahrzeugfarbe.
        folium.PolyLine(
            geom, color="#111827", weight=9, opacity=0.82, tooltip=vid
        ).add_to(m)
        folium.PolyLine(
            geom, color=color, weight=5, opacity=1, tooltip=vid
        ).add_to(m)

        for sequence, stop in enumerate(route.get("stops", []), start=1):
            popup = (
                f"<b>{sequence}. {stop['kunde']}</b><br>"
                f"Auftrag: {stop.get('auftrag', '')}<br>"
                f"{stop['paletten']} Paletten<br>"
                f"{stop['gesamtgewicht_kg']} kg"
            )
            folium.Marker(
                [stop["lat"], stop["lon"]],
                tooltip=f"{vid} · Stopp {sequence}: {stop['kunde']} · {stop['paletten']} Pal.",
                popup=popup,
            ).add_to(m)

    if bounds:
        m.fit_bounds(bounds)

    return m.get_root().render()
