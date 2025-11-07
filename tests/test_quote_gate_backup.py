import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ai_trading.execution import live_trading
from tests.execution.test_live_trading_degraded_feed import DummyLiveEngine


def _quote_payload():
    now = datetime.now(UTC)
    return {
        "bid": 100.0,
        "ask": 100.25,
        "ts": now,
        "timestamp": now,
        "synthetic": False,
    }


@pytest.fixture(autouse=True)
def _patch_guards(monkeypatch):
    monkeypatch.setattr(live_trading, "_safe_mode_guard", lambda *a, **k: False)
    monkeypatch.setattr(live_trading, "_require_bid_ask_quotes", lambda: False)
    monkeypatch.setattr(live_trading, "quote_fresh_enough", lambda *a, **k: True)
    monkeypatch.setattr(live_trading, "guard_shadow_active", lambda: False)
    monkeypatch.setattr(live_trading, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(
        live_trading,
        "_call_preflight_capacity",
        lambda *a, **k: live_trading.CapacityCheck(True, int(a[3]), None),
    )
    monkeypatch.setattr(
        live_trading.provider_monitor,
        "is_disabled",
        lambda *_a, **_k: False,
    )


def test_backup_quote_gate_allows_entries(monkeypatch, caplog):
    engine = DummyLiveEngine()
    config = SimpleNamespace(
        min_quote_freshness_ms=1500,
        degraded_feed_mode="block",
        degraded_feed_limit_widen_bps=0,
        execution_require_realtime_nbbo=False,
    )
    monkeypatch.setattr(live_trading, "get_trading_config", lambda: config)
    monkeypatch.setattr(engine, "_broker_lock_suppressed", lambda **_: False)
    monkeypatch.setattr(live_trading, "_require_bid_ask_quotes", lambda: True)
    annotations = {
        "price_source": "yahoo",
        "gap_ratio": 0.01,
        "gap_limit": 0.05,
        "fallback_quote_age": 0.25,
        "fallback_quote_limit": 2.0,
    }

    caplog.set_level(logging.INFO, logger="ai_trading.execution.live_trading")
    result = engine.execute_order(
        "AAPL",
        "buy",
        5,
        order_type="limit",
        limit_price=100.0,
        quote=_quote_payload(),
        annotations=annotations,
    )

    assert result is not None
    assert engine.last_submitted is not None
    assert any(rec.msg == "QUOTE_GATE_BACKUP_OK" for rec in caplog.records)
