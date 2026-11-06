from unittest.mock import Mock
import pytest
from src import tomtom_routing as routing

@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(routing.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(routing.time, 'sleep', lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(routing, '_ROUTING_NEXT_REQUEST', 0.0)
    return now

def response(status=200, headers=None):
    return Mock(status_code=status, headers=headers or {})

def test_separate_sessions_share_spacing(clock):
    starts = []
    for _ in range(7):
        session = Mock()
        session.get.side_effect = lambda *a, **k: (starts.append(clock[0]), response())[1]
        routing._routing_get(session, 'url', {}, 20)
    assert starts == [100 + i * .25 for i in range(7)]

def test_429_retry_after(clock):
    session = Mock()
    session.get.side_effect = [response(429, {'Retry-After': '3'}), response()]
    assert routing._routing_get(session, 'url', {}, 20).status_code == 200
    assert clock[0] == 103
    assert session.get.call_count == 2

def test_retries_bounded_and_cooldown_shared(clock):
    session = Mock()
    session.get.return_value = response(429)
    assert routing._routing_get(session, 'url', {}, 20).status_code == 429
    assert session.get.call_count == 3
    other = Mock()
    other.get.return_value = response()
    routing._routing_get(other, 'url', {}, 20)
    assert clock[0] == 107

def test_long_retry_after_not_shortened(clock):
    session = Mock()
    session.get.return_value = response(429, {'Retry-After': '120'})
    routing._routing_get(session, 'url', {}, 20)
    with pytest.raises(RuntimeError, match='Wartezeit'):
        routing._routing_get(session, 'url', {}, 20)
    assert session.get.call_count == 1

def test_auth_error_not_retried(clock):
    session = Mock()
    session.get.return_value = response(401)
    assert routing._routing_get(session, 'url', {}, 20).status_code == 401
    assert session.get.call_count == 1
