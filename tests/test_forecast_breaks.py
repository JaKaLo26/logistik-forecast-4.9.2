from datetime import datetime
from src.forecast import _apply_segment_breaks, _forecast_break_plan

START = datetime.fromisoformat('2026-11-07T07:00:00+01:00')

def segment(minutes, service=0):
    return dict(model_forecast_s=minutes*60, actual_travel_s=None,
                planned_service_s=service*60, actual_service_s=None)

def test_exact_limit_at_tour_end_needs_no_extra_break():
    s=segment(270)
    end=_apply_segment_breaks(s, [], START, {})
    assert s['planned_break_s']==0
    assert (end-START).total_seconds()==270*60

def test_long_leg_includes_break_in_eta_not_driving_target():
    s=segment(300)
    end=_apply_segment_breaks(s, [], START, {})
    assert s['model_forecast_s']==300*60
    assert s['planned_break_s']==45*60
    assert (end-START).total_seconds()==345*60
    assert s['break_events'][0]['starts_at']=='2026-11-07T11:30:00+01:00'

def test_return_to_depot_and_service_does_not_reset_driving():
    a=segment(260,30)
    end=_apply_segment_breaks(a, [], START, {})
    b=segment(20)
    finish=_apply_segment_breaks(b, [a], end, {})
    assert b['planned_break_s']==2700
    assert (finish-START).total_seconds()==355*60

def test_initial_driver_history_and_budget():
    s=segment(20)
    profile={'drive_since_break_min':260,'remaining_duty_min':60}
    end=_apply_segment_breaks(s, [], START, profile)
    assert (end-START).total_seconds()==65*60
    assert not _forecast_break_plan([s], profile)['feasible']

def test_service_at_driving_limit_has_separate_break():
    s=segment(270,20)
    end=_apply_segment_breaks(s, [], START, {})
    assert s['break_during_drive_s']==0
    assert s['break_during_service_s']==2700
    assert datetime.fromisoformat(s['arrival_time_forecast']).hour==11
    assert (end-START).total_seconds()==335*60
