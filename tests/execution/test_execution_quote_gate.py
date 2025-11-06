import logging
from datetime import UTC, datetime

import pytest

from ai_trading.execution import live_trading
from tests.execution.test_live_trading_degraded_feed import DummyLiveEngine


@pytest.fixture(autouse=True)
def _patch_quote_gate_guards(monkeypatch) -> None:
    monkeypatch.setattr(live_trading, "_safe_mode_guard", lambda *a, **k: False)
    monkeypatch.setattr(live_trading, "_require_bid_ask_quotes", lambda: False)
    monkeypatch.setattr(live_trading, "quote_fresh_enough", lambda *a, **k: True)
    monkeypatch.setattr(live_trading, "guard_shadow_active", lambda: False)
    monkeypatch.setattr(live_trading, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(
        live_trading.provider_monitor,
        "is_disabled",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        live_trading,
        "_call_preflight_capacity",
        lambda *a, **k: live_trading.CapacityCheck(True, int(a[3]), None),
    )


def _quote_payload() -> dict:
    return {
        "bid": 100.0,
        "ask": 101.0,
        "ts": datetime.now(UTC),
        "synthetic": True,
    }


def test_entry_block_log_emitted_for_degraded_gate(monkeypatch, caplog) -> None:
    engine = DummyLiveEngine()
    config = type("Cfg", (), {
        "min_quote_freshness_ms": 1500,
        "degraded_feed_mode": "block",
        "degraded_feed_limit_widen_bps": 0,
        "execution_require_realtime_nbbo": False,
    })()
    monkeypatch.setattr(live_trading, "get_trading_config", lambda: config)
    monkeypatch.setattr(engine, "_broker_lock_suppressed", lambda **_: False)

    caplog.set_level(logging.WARNING, logger="ai_trading.execution.live_trading")

    result = engine.execute_order(
        "AAPL",
        "buy",
        5,
        order_type="limit",
        limit_price=100.0,
        quote=_quote_payload(),
        annotations={"price_source": "backup"},
    )

    assert result is None
    assert engine.last_submitted is None

    entry_logs = [
        record
        for record in caplog.records
        if record.msg == "ENTRY_BLOCKED_BY_QUOTE_QUALITY"
    ]
    assert entry_logs, "ENTRY_BLOCKED_BY_QUOTE_QUALITY not emitted"
    latest = entry_logs[-1]
    assert latest.symbol == "AAPL"
    assert latest.gate == "degraded_feed_block"
    assert latest.limit_basis is not None
