from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from src.builtin_model import load_builtin_model, predict_road_matrix
from src.ml_training import predict_details, predict_seconds
from src.forecast import calculate_tour_forecast

DATE='2026-11-06T10:00:00+01:00'


def test_real_artifact_long_route_no_longer_saturates_at_40_minutes():
    bundle=load_builtin_model()
    decision=predict_details(bundle,262*60,300000,DATE)
    assert decision['raw_model_s'] < 60*60
    assert decision['seconds'] == 262*60
    assert decision['warning']
    result=predict_road_matrix([[0,262],[262,0]],[[0,300000],[300000,0]],DATE,bundle)
    assert result == [[0,262],[262,0]]


def test_in_domain_model_is_preserved():
    bundle=load_builtin_model()
    decision=predict_details(bundle,600,7500,DATE)
    assert decision['warning'] is None
    assert decision['seconds'] == decision['raw_model_s']
    assert predict_seconds(bundle,600,7500,DATE) == decision['seconds']


def test_long_segment_live_fallback_propagates_to_eta_and_total():
    bundle=load_builtin_model()
    route={'vehicle_id':'V','stops':[{'auftrag':'A','lat':49,'lon':10,'service_min':12}],
           'legs':[{'duration':262*60,'distance':300000}]*2}
    provider=SimpleNamespace(success=True,tomtom_no_traffic_s=262*60,tomtom_historic_s=341*60,
        tomtom_live_s=341*60,traffic_delay_s=79*60,distance_m=300000,error=None)
    with patch('src.forecast.TomTomRoutingProvider') as factory:
        factory.return_value.calculate_segment.return_value=provider
        result=calculate_tour_forecast(route,{'lat':48,'lon':9},DATE,ml_bundle=bundle)
    assert result['model_forecast_s']==2*341*60
    segment=result['segments'][0]
    assert segment['model_forecast_s']==341*60
    assert segment['forecast_source']=='TomTom-Live-Ersatzwert'
    assert segment['forecast_warning']
    from datetime import datetime
    a=datetime.fromisoformat(segment['departure_time'])
    b=datetime.fromisoformat(segment['arrival_time_forecast'])
    assert (b-a).total_seconds()==(341+45)*60
    assert segment["planned_break_s"] == 45*60


def test_long_segment_without_live_uses_osrm_not_fake_tomtom():
    bundle=load_builtin_model()
    route={'vehicle_id':'V','stops':[{'auftrag':'A','lat':49,'lon':10,'service_min':12}],
           'legs':[{'duration':262*60,'distance':300000}]*2}
    with patch('src.forecast.TomTomRoutingProvider') as factory:
        factory.return_value.calculate_segment.return_value=SimpleNamespace(success=False,error='offline')
        result=calculate_tour_forecast(route,{'lat':48,'lon':9},DATE,ml_bundle=bundle)
    first=result['segments'][0]
    assert first['model_forecast_s']==262*60
    assert first['tomtom_live_s']==first['tomtom_historic_s']==0
    assert first['forecast_source']=='OSRM-Ersatzwert'


def test_card_discloses_fallback_and_never_calls_live_actual():
    import ast
    from pathlib import Path
    from html import escape
    module=ast.parse(Path('app.py').read_text())
    node=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='render_forecast_card')
    scope={'escape':escape}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'app.py','exec'),scope)
    html=scope['render_forecast_card']('V','Test',[dict(to_stop_id='A',model_forecast_s=341*60,
        osrm_baseline_s=262*60,tomtom_success=True,tomtom_live_s=341*60,tomtom_historic_s=0,
        forecast_source='TomTom-Live-Ersatzwert',forecast_warning='Außerhalb des Trainingsbereichs')])
    assert '341 min' in html and 'TomTom-Live-Ersatzwert' in html
    assert 'Ist-Zeit' not in html
    assert 'nicht verfügbar' in html
