from __future__ import annotations
import math
import pandas as pd

PALLET_TARE_KG = 25

REQUIRED = [
    "auftrag", "kunde", "strasse", "plz", "ort",
    "paletten", "warengewicht_kg", "service_min",
]


def normalize_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Fehlende CSV-Spalten: " + ", ".join(missing))

    for c in ["paletten", "warengewicht_kg", "service_min"]:
        df[c] = pd.to_numeric(df[c], errors="raise")

    df["paletten"] = df["paletten"].astype(int)
    df["palettengewicht_kg"] = df["paletten"] * PALLET_TARE_KG
    df["gesamtgewicht_kg"] = df["warengewicht_kg"] + df["palettengewicht_kg"]
    df["adresse"] = (
        df["strasse"].astype(str)
        + ", "
        + df["plz"].astype(str)
        + " "
        + df["ort"].astype(str)
    )
    return df


def summarize_orders(df: pd.DataFrame) -> dict:
    pallets = int(df["paletten"].sum())
    weight = int(df["gesamtgewicht_kg"].sum())
    return {
        "auftraege": len(df),
        "paletten": pallets,
        "gewicht_kg": weight,
        "durchschnitt_kg_pro_palette": math.floor(weight / pallets) if pallets else 0,
    }
