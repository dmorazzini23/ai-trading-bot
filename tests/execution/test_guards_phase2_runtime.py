from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from ai_trading.execution import guards


def test_timestamp_coercion_and_quote_staleness_paths() -> None:
    now = datetime(2026, 4, 25, 16, 0, tzinfo=UTC)

    assert guards._coerce_timestamp(None) is None
    assert guards._coerce_timestamp(0) == datetime(1970, 1, 1, tzinfo=UTC)
    assert guards._coerce_timestamp(datetime(2026, 4, 25, 16, 0)) == now
    assert guards._coerce_timestamp(datetime(2026, 4, 25, 12, 0, tzinfo=timezone(timedelta(hours=-4)))) == now
    assert guards._coerce_timestamp("2026-04-25T16:00:00Z") is None

    assert guards._is_stale({"timestamp": now - timedelta(seconds=5)}, now, 10) == (
        False,
        None,
    )
    assert guards._is_stale({"timestamp": now + timedelta(seconds=6)}, now, 10) == (
        True,
        "future_quote_timestamp",
    )
    assert guards._is_stale({"ts": now - timedelta(seconds=11)}, now, 10) == (
        True,
        "stale_quote",
    )
    assert guards._is_stale({"bid": 1.0}, now, 10) == (True, "quote_timestamp_missing")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        (Decimal("1"), True),
        (" yes ", True),
        ("FALSE", False),
        ("surprise", False),
        (None, False),
    ],
)
def test_safe_bool_normalizes_runtime_payloads(value: Any, expected: bool) -> None:
    assert guards._safe_bool(value) is expected


def test_config_fallbacks_and_explicit_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        guards,
        "get_trading_config",
        lambda: SimpleNamespace(execution_require_bid_ask=False, execution_max_staleness_sec="7"),
    )

    assert guards._require_bid_ask() is False
    assert guards._max_age_seconds() == 7

    monkeypatch.setattr(guards, "get_trading_config", lambda: (_ for _ in ()).throw(RuntimeError("cfg")))
    assert guards._require_bid_ask() is True
    assert guards._max_age_seconds() == 60

    monkeypatch.setattr(
        guards,
        "get_trading_config",
        lambda: SimpleNamespace(execution_require_bid_ask=True, execution_max_staleness_sec="bad"),
    )
    assert guards._max_age_seconds() == 60


def test_can_execute_quote_gate_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 4, 25, 16, 0, tzinfo=UTC)
    monkeypatch.setattr(guards, "_now", lambda: now)
    monkeypatch.setattr(guards, "_require_bid_ask", lambda: True)

    assert guards.can_execute(None, now=now) == (False, "no_quote")
    assert guards.can_execute({"timestamp": now}, now=now) == (False, "missing_bid_ask")
    assert guards.can_execute({"bid": 2.0, "ask": 1.0, "timestamp": now}, now=now) == (
        False,
        "negative_spread",
    )
    assert guards.can_execute({"bp": "1", "ap": "2", "t": now - timedelta(seconds=70)}, now=now) == (
        False,
        "stale_quote",
    )
    assert guards.can_execute({"bp": "1", "ap": "2", "t": now + timedelta(seconds=6)}, now=now) == (
        False,
        "future_quote_timestamp",
    )
    assert guards.can_execute({"bid_price": "1", "ask_price": "2", "time": now}, now=now) == (
        True,
        None,
    )

    monkeypatch.setattr(guards, "_require_bid_ask", lambda: False)
    assert guards.can_execute({"timestamp": now}, now=now, max_age_sec=1) == (True, None)


def test_quote_fresh_enough_and_cycle_state(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 4, 25, 16, 0, tzinfo=UTC)
    monkeypatch.setattr(guards, "_utcnow", lambda: now)

    assert guards.quote_fresh_enough(None, 10) is False
    assert guards.quote_fresh_enough(now + timedelta(seconds=6), 10) is False
    assert guards.quote_fresh_enough(now.replace(tzinfo=None) - timedelta(seconds=5), 10) is True
    assert guards.quote_fresh_enough(now - timedelta(seconds=11), 10) is False

    guards.begin_cycle(universe_size=3, degraded=True)
    guards.mark_symbol_stale()
    guards.mark_symbol_stale()
    guards.end_cycle(stale_threshold_ratio=0.1)

    assert guards.STATE.universe_size == 3
    assert guards.STATE.stale_symbols == 2
    assert guards.shadow_active() is False
