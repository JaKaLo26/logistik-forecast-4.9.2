from __future__ import annotations
import folium

def build_map(routes: list[dict], center=(51.0,10.0)) -> str:
    m=folium.Map(location=center, zoom_start=6, control_scale=True)
    bounds=[]
    for route in routes:
        geom=route.get('geometry',[]); color=route.get('color','#2563eb'); vid=route.get('vehicle_id','LKW')
        if not geom: continue
        bounds += geom
        # dunkle Kontur + farbige Innenlinie
        folium.PolyLine(geom, color='#111827', weight=9, opacity=.8, tooltip=vid).add_to(m)
        folium.PolyLine(geom, color=color, weight=5, opacity=1, tooltip=vid).add_to(m)
        for stop in route.get('stops',[]):
            folium.Marker([stop['lat'],stop['lon']], tooltip=f"{vid}: {stop['kunde']} – {stop['paletten']} Pal.",
                popup=f"<b>{stop['kunde']}</b><br>{stop['paletten']} Paletten<br>{stop['gesamtgewicht_kg']} kg").add_to(m)
    if bounds: m.fit_bounds(bounds)
    return m.get_root().render()
