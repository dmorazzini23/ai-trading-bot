from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ai_trading.execution import live_trading as lt


def test_capacity_observation_reuses_read_and_preserves_pending_pressure_guard(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(lt, "monotonic_time", lambda:clock[0])
    monkeypatch.setattr(lt, "_config_int", lambda _key, default=None:default)
    monkeypatch.setattr(lt, "_config_int_alias", lambda _keys, default=None:default)
    monkeypatch.setattr(lt, "_config_float", lambda _key, default=None:default)
    monkeypatch.setattr(lt, "_resolve_bool_env", lambda _key:True)
    orders = [SimpleNamespace(status="pending_new", symbol="AMZN", side="buy", qty="1", limit_price="100", created_at=datetime.now(UTC)-timedelta(minutes=2)) for _ in range(8)]
    broker = SimpleNamespace(list_orders=lambda:orders)
    calls = []
    def list_orders(client, **kwargs):
        calls.append((client, kwargs))
        return orders
    monkeypatch.setattr(lt, "list_alpaca_orders", list_orders)
    account = SimpleNamespace(buying_power="100000", cash="100000", non_marginable_buying_power="100000")
    cap = lt.preflight_capacity("AMZN", "buy", 100, 1, broker, account=account)
    assert cap.can_submit and len(calls) == 1
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine.trading_client = broker
    monkeypatch.setattr(engine, "_capacity_broker", lambda client:client)
    allowed, context = engine._runtime_pending_new_pressure_allows_openings(
        pending_orders=lt._capacity_open_orders(cap, broker), pending_orders_observed_at=cap.open_orders_observed_at,
    )
    assert allowed is False and context["pending_new_count"] == 8
    assert len(calls) == 1
    # A stale observation requires a fresh read and still enforces the guard.
    clock[0] += 2
    allowed, _ = engine._runtime_pending_new_pressure_allows_openings(
        pending_orders=cap.open_orders, pending_orders_observed_at=cap.open_orders_observed_at,
    )
    assert allowed is False and len(calls) == 2


@pytest.mark.parametrize("age", [-1.0, 1.0, 120.0, float("nan")])
def test_capacity_observation_does_not_extend_freshness(monkeypatch, age):
    broker = object()
    monkeypatch.setattr(lt, "monotonic_time", lambda:100.0+age)
    cap = lt.CapacityCheck(True, 1, open_orders=(), open_orders_observed_at=100.0, open_orders_broker=broker)
    assert lt._capacity_open_orders(cap, broker) is None


def test_capacity_observation_is_owned_by_one_broker(monkeypatch):
    monkeypatch.setattr(lt, "monotonic_time", lambda:100.0)
    cap = lt.CapacityCheck(True, 1, open_orders=(), open_orders_observed_at=100.0, open_orders_broker=object())
    assert lt._capacity_open_orders(cap, object()) is None
