from __future__ import annotations

import math
from dataclasses import dataclass
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def _window_midpoint(row) -> float | None:
    """Zeitfenster-Mittelpunkt in Minuten; None, falls nicht gepflegt."""
    start = row.get("zeitfenster_von")
    end = row.get("zeitfenster_bis")
    if pd.isna(start) or pd.isna(end) or not str(start).strip() or not str(end).strip():
        return None

    def parse(value):
        parts = str(value).strip().split(":")
        if len(parts) < 2:
            return None
        return int(parts[0]) * 60 + int(parts[1])

    try:
        a, b = parse(start), parse(end)
        if a is None or b is None:
            return None
        return (a + b) / 2
    except Exception:
        return None


def _vehicle_score(v) -> float:
    # Beide Kapazitätsdimensionen einbeziehen.
    return float(v["pallet_capacity"]) + float(v["payload_kg"]) / 1000.0


def _cluster_centroid(rows: list[dict]) -> tuple[float, float]:
    return (
        sum(float(r["lat"]) for r in rows) / len(rows),
        sum(float(r["lon"]) for r in rows) / len(rows),
    )


def _estimated_cluster_km(rows: list[dict], depot: tuple[float, float]) -> float:
    """Schnelle geometrische Tourabschätzung: nearest-neighbour + Rückweg."""
    if not rows:
        return 0.0
    unvisited = rows[:]
    current = depot
    total = 0.0
    while unvisited:
        nxt = min(
            unvisited,
            key=lambda r: haversine_km(current, (float(r["lat"]), float(r["lon"]))),
        )
        p = (float(nxt["lat"]), float(nxt["lon"]))
        total += haversine_km(current, p)
        current = p
        unvisited.remove(nxt)
    total += haversine_km(current, depot)
    return total


def cluster_orders(
    orders: pd.DataFrame,
    vehicles: pd.DataFrame,
    depot: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Kapazitätsbewusste geografische Clusterbildung.

    Strategie:
    - Nur verfügbare Fahrzeuge.
    - Fernste noch offene Adresse wird Seed eines neuen Gebiets.
    - Danach werden die räumlich nächstgelegenen Stopps ergänzt,
      solange Paletten UND Nutzlast passen.
    - Zeitfenster-Mittelpunkte wirken als leichte Zusatzstrafe,
      damit völlig gegensätzliche Zeitfenster seltener gemischt werden.
    - Ziel: kompakte, wenig überlappende Liefergebiete.
    """
    if orders.empty:
        return orders.copy(), pd.DataFrame(), ["Keine Aufträge vorhanden."]

    vehicles = vehicles.copy()
    if "available" in vehicles.columns:
        vehicles = vehicles[vehicles["available"].astype(bool)].copy()
    if vehicles.empty:
        return pd.DataFrame(), pd.DataFrame(), ["Keine verfügbaren Fahrzeuge."]

    for c in ["pallet_capacity", "payload_kg"]:
        vehicles[c] = pd.to_numeric(vehicles[c], errors="raise")

    rows = [r.to_dict() for _, r in orders.iterrows()]
    for r in rows:
        r["_window_mid"] = _window_midpoint(r)

    # Große Fahrzeuge zuerst, damit entfernte Gebiete nicht künstlich zersplittern.
    vehicle_records = (
        vehicles.sort_values(
            by=["pallet_capacity", "payload_kg"],
            ascending=False,
        )
        .to_dict("records")
    )

    unassigned = rows[:]
    assignments: list[dict] = []
    cluster_summaries: list[dict] = []
    warnings: list[str] = []
    cluster_no = 1

    for vehicle in vehicle_records:
        if not unassigned:
            break

        cap_p = int(vehicle["pallet_capacity"])
        cap_w = float(vehicle["payload_kg"])

        feasible_seeds = [
            r for r in unassigned
            if int(r["paletten"]) <= cap_p and float(r["gesamtgewicht_kg"]) <= cap_w
        ]
        if not feasible_seeds:
            continue

        # Farthest-first: Außenregionen werden sauber voneinander getrennt.
        seed = max(
            feasible_seeds,
            key=lambda r: haversine_km(
                depot, (float(r["lat"]), float(r["lon"]))
            ),
        )

        cluster = [seed]
        used_p = int(seed["paletten"])
        used_w = float(seed["gesamtgewicht_kg"])
        unassigned.remove(seed)

        while unassigned:
            centroid = _cluster_centroid(cluster)
            mids = [r["_window_mid"] for r in cluster if r["_window_mid"] is not None]
            mean_mid = sum(mids) / len(mids) if mids else None

            candidates = []
            for r in unassigned:
                new_p = used_p + int(r["paletten"])
                new_w = used_w + float(r["gesamtgewicht_kg"])
                if new_p > cap_p or new_w > cap_w:
                    continue

                geo_km = haversine_km(
                    centroid, (float(r["lat"]), float(r["lon"]))
                )
                tw_penalty = 0.0
                if mean_mid is not None and r["_window_mid"] is not None:
                    # 60 Minuten Zeitfenster-Abstand ≈ 3 km Zusatzkosten.
                    tw_penalty = abs(r["_window_mid"] - mean_mid) * 0.05

                # Leichter Depot-Radialitätsbonus verhindert kreuzende Gebiete.
                radial = abs(
                    haversine_km(depot, centroid)
                    - haversine_km(
                        depot, (float(r["lat"]), float(r["lon"]))
                    )
                ) * 0.10

                candidates.append((geo_km + tw_penalty + radial, r))

            if not candidates:
                break

            _, chosen = min(candidates, key=lambda x: x[0])
            cluster.append(chosen)
            used_p += int(chosen["paletten"])
            used_w += float(chosen["gesamtgewicht_kg"])
            unassigned.remove(chosen)

        cluster_id = f"REGION-{cluster_no:02d}"
        for r in cluster:
            clean = {k: v for k, v in r.items() if not k.startswith("_")}
            clean["cluster_id"] = cluster_id
            clean["vehicle_id"] = vehicle["vehicle_id"]
            assignments.append(clean)

        cluster_summaries.append({
            "cluster_id": cluster_id,
            "vehicle_id": vehicle["vehicle_id"],
            "stopps": len(cluster),
            "paletten": used_p,
            "paletten_kapazitaet": cap_p,
            "paletten_auslastung_pct": round(used_p / cap_p * 100, 1) if cap_p else 0,
            "gewicht_kg": round(used_w),
            "nutzlast_kg": round(cap_w),
            "gewicht_auslastung_pct": round(used_w / cap_w * 100, 1) if cap_w else 0,
            "geschaetzte_geometrische_tour_km": round(
                _estimated_cluster_km(cluster, depot), 1
            ),
        })
        cluster_no += 1

    # Fallback: nicht zuweisbare Aufträge transparent ausgeben.
    for r in unassigned:
        clean = {k: v for k, v in r.items() if not k.startswith("_")}
        clean["cluster_id"] = "NICHT ZUGEWIESEN"
        clean["vehicle_id"] = "NICHT ZUGEWIESEN"
        assignments.append(clean)
        warnings.append(
            f"Auftrag {r['auftrag']} ({int(r['paletten'])} Pal., "
            f"{int(r['gesamtgewicht_kg'])} kg) konnte keinem regionalen Cluster "
            "innerhalb der verfügbaren Fahrzeugkapazität zugewiesen werden."
        )

    return (
        pd.DataFrame(assignments),
        pd.DataFrame(cluster_summaries),
        warnings,
    )
