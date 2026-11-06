from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

import pytest
import src.forecast as forecast
from src.routing import OSRMRouter
from src.tomtom_routing import TomTomRoutingProvider


def response(payload):
    return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: payload)


def test_screenshot_values_and_invalid_model(monkeypatch):
    monkeypatch.setattr(forecast, 'predict_if_configured', lambda *args: 40 * 60)
    result = forecast.calculate_model_forecast_details(262*60, 341*60, 341*60, 79*60, 1)
    assert result['seconds'] == round(333.1 * 60)
    assert result['warning']
    assert result['source'] == 'weighted_reference'
    # Delay is already represented by live; it must not be added twice.
    assert result['seconds'] == forecast.calculate_model_forecast_s(262*60, 341*60, 341*60, 0, 1)


def test_missing_or_invalid_legs_not_averaged():
    with pytest.raises(ValueError, match='Teilstrecken'):
        forecast.extract_osrm_leg_baselines({'duration_s': 18000, 'legs': []}, 6)
    for duration in [None, float('inf'), -1]:
        with pytest.raises(ValueError):
            forecast.extract_osrm_leg_baselines({'legs': [{'duration': duration}]}, 1)
    assert forecast.extract_osrm_leg_baselines({'legs': [{'duration': 15720}, {'duration': 60}]}, 2) == [15720, 60]


def test_provider_does_not_invent_missing_live_or_historic(monkeypatch):
    provider = TomTomRoutingProvider(api_key='test')
    monkeypatch.setattr(provider.session, 'get', lambda *a, **k: response({'routes': [{'summary': {'travelTimeInSeconds': 20460}}]}))
    result = provider.calculate_segment((48, 9), (49, 10))
    assert result.success
    assert result.tomtom_best_s == 20460
    assert result.tomtom_live_s == result.tomtom_historic_s == 0


def test_partial_tour_live_comparison_is_unavailable():
    parts = [dict(tomtom_success=True, tomtom_live_s=20460),
             dict(tomtom_success=False, tomtom_live_s=60)]
    assert forecast._provider_minutes(parts[0], 'tomtom_live_s') == 341
    assert forecast._provider_minutes({'segments': parts}, 'tomtom_live_s') is None
    parts[1] = dict(tomtom_success=True, tomtom_live_s=60)
    assert forecast._provider_minutes({'segments': parts}, 'tomtom_live_s') == 342


@pytest.mark.parametrize('available', [True, False])
def test_segment_to_tour_eta_and_display_share_seconds(monkeypatch, available):
    calls = []
    class Provider:
        def __init__(self, **kwargs):
            pass
        def calculate_segment(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(success=available, tomtom_no_traffic_s=15720,
                tomtom_historic_s=20460, tomtom_live_s=20460, traffic_delay_s=4740,
                distance_m=300000, error=None if available else 'offline')
    monkeypatch.setattr(forecast, 'TomTomRoutingProvider', Provider)
    monkeypatch.setattr(forecast, 'predict_if_configured', lambda *args: None)
    start = datetime(2026, 11, 6, 8, tzinfo=timezone.utc)
    depot = dict(lat=48, lon=9, auftrag='DEPOT')
    stop = dict(lat=49, lon=10, auftrag='A', kunde='A', service_min=12)
    route = dict(vehicle_id='V1', stops=[stop], legs=[{'duration':15720}, {'duration':600}], duration_s=16320)
    tour = forecast.calculate_tour_forecast(route, depot, start, tour_id='test')
    parts = tour['segments']
    assert len(calls) == len(parts) == 2
    assert calls[0]['origin'] == (48.0, 9.0)
    assert calls[0]['destination'] == calls[1]['origin'] == (49.0, 10.0)
    first = parts[0]
    duration = round(333.1*60) if available else 15720
    assert first['model_forecast_s'] == duration
    assert datetime.fromisoformat(first['arrival_time_forecast']) == start + timedelta(seconds=duration)
    assert datetime.fromisoformat(parts[1]['departure_time']) == start + timedelta(seconds=duration+720)
    assert tour['model_forecast_s'] == sum(p['model_forecast_s'] for p in parts)
    rows = forecast.segment_rows(tour)
    assert rows[0]['forecast_min'] == round(duration/60)
    summary = forecast.tour_summary(tour)
    assert summary['forecast_fahrzeit_min'] == round(tour['model_forecast_s']/60)
    if not available:
        assert first['tomtom_live_s'] == first['tomtom_historic_s'] == 0
        assert first['forecast_source'] == 'osrm_fallback'
        assert rows[0]['tomtom_live_min'] is None
        assert summary['tomtom_live_min'] is None


def test_trip_waypoints_reordered_without_reordering_legs(monkeypatch):
    router = OSRMRouter('https://example.invalid')
    payload = {'code':'Ok','trips':[{'distance':10,'duration':15780,
        'geometry':{'coordinates':[[9,48],[10,49],[11,50],[9,48]]},
        'legs':[{'duration':15720},{'duration':30},{'duration':30}]}],
        'waypoints':[{'waypoint_index':0,'trips_index':0},
                     {'waypoint_index':2,'trips_index':0}, {'waypoint_index':1,'trips_index':0}]}
    monkeypatch.setattr(router.session, 'get', lambda *a, **k: response(payload))
    a, b = dict(lat=50,lon=11,auftrag='A'),dict(lat=49,lon=10,auftrag='B')
    result = router.optimize_roundtrip((48,9), [a,b])
    assert result['ordered_stops'] == [b,a]
    assert result['legs'][0]['duration'] == 15720
    payload['waypoints'][2]['waypoint_index'] = 2
    with pytest.raises(ValueError, match='zuordnen'):
        router.optimize_roundtrip((48,9), [a,b])
