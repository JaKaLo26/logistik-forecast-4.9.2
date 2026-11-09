"""Load order and delivery progress for an already planned route."""


def loading_plan(stops):
    """Load through the rear door, last delivery first."""
    return [
        {
            "Ladereihenfolge": index,
            "Stopp": len(stops) - index + 1,
            "Auftrag": stop["auftrag"],
            "Kunde": stop.get("kunde", ""),
            "Paletten": int(stop["paletten"]),
        }
        for index, stop in enumerate(reversed(stops), 1)
    ]


def delivery_progress(stops, completed, segments):
    remaining = stops[completed:]
    result = {
        "completed": completed,
        "total": len(stops),
        "remaining_pallets": sum(int(stop["paletten"]) for stop in remaining),
    }
    if not remaining:
        return result

    stop = remaining[0]
    leg = next((s for s in segments if s.get("to_stop_id") == str(stop["auftrag"])), {})
    result.update(next_customer=stop.get("kunde") or stop["auftrag"],
                  next_pallets=int(stop["paletten"]))
    duration = leg.get("tomtom_live_s") if leg.get("tomtom_success") else None
    result["live_duration_s"] = duration
    result["forecast_duration_s"] = leg.get("model_forecast_s")
    return result


def progress_markdown(progress):
    result = (f"**Stopps:** {progress['completed']}/{progress['total']} · "
              f"**Restpaletten:** {progress['remaining_pallets']}")
    if "next_customer" not in progress:
        return result + " · Alle Lieferstopps abgeschlossen."
    result += (f"\n\n**Nächster Kunde:** {progress['next_customer']} · "
               f"**Paletten am nächsten Stopp:** {progress['next_pallets']}")
    live = progress["live_duration_s"]
    forecast = progress["forecast_duration_s"]
    if live is not None:
        result += f" · **Aktuelle Fahrzeit:** {live / 60:.1f} min"
    else:
        result += " · **Live-Fahrzeit:** nicht verfügbar"
        if forecast is not None:
            result += f" · **Forecast:** {forecast / 60:.1f} min"
    return result
