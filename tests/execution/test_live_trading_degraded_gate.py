"""NBBO gating tests for degraded data scenarios."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from ai_trading.execution import live_trading
from tests.execution.test_live_trading_degraded_feed import DummyLiveEngine


@pytest.fixture(autouse=True)
def _patch_common_guards(monkeypatch) -> None:
    """Keep execution heuristics deterministic for gating assertions."""

    monkeypatch.setattr(live_trading, "_safe_mode_guard", lambda *a, **k: False)
    monkeypatch.setattr(live_trading, "guard_shadow_active", lambda: False)
    monkeypatch.setattr(live_trading, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(
        live_trading.provider_monitor,
        "is_disabled",
        lambda *_a, **_k: True,
    )


def _synthetic_quote_payload() -> dict:
    return {
        "bid": 100.0,
        "ask": 101.0,
        "ts": datetime.now(UTC),
        "synthetic": True,
    }


def test_synthetic_quotes_blocked_when_nbbo_required(monkeypatch, caplog) -> None:
    """Synthetic quotes must not submit orders when NBBO is required."""

    engine = DummyLiveEngine()
    monkeypatch.setenv("TRADING__DEGRADED_FEED_MODE", "block")
    monkeypatch.setenv("NBBO_REQUIRED_FOR_LIMIT", "true")
    from ai_trading.config.management import reload_trading_config

    reload_trading_config()

    caplog.set_level(logging.INFO)
    monkeypatch.setattr(engine, "_broker_lock_suppressed", lambda **_: False)

    result = engine.execute_order(
        "AAPL",
        "buy",
        5,
        order_type="limit",
        limit_price=100.0,
        annotations={"price_source": "backup", "quote": _synthetic_quote_payload()},
    )

    assert result is None
    assert engine.last_submitted is None

    emitted_messages = {
        record.msg
        for record in caplog.records
        if record.name == "ai_trading.execution.live_trading"
    }
    assert "DEGRADED_FEED_BLOCK_ENTRY" in emitted_messages or "ORDER_SKIPPED_PRICE_GATED" in emitted_messages

    monkeypatch.delenv("TRADING__DEGRADED_FEED_MODE", raising=False)
    monkeypatch.delenv("NBBO_REQUIRED_FOR_LIMIT", raising=False)
    reload_trading_config()
