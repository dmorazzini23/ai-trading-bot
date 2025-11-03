"""Ensure degraded NBBO gating can downgrade to market orders."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ai_trading.execution import live_trading
from tests.execution.test_live_trading_degraded_feed import DummyLiveEngine


@pytest.fixture(autouse=True)
def _patch_execution_guards(monkeypatch) -> None:
    """Stabilize guard behaviour for deterministic assertions."""

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
        lambda *_a, **_k: True,
    )
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)


def _synthetic_quote() -> dict:
    return {
        "bid": 100.0,
        "ask": 100.5,
        "ts": datetime.now(UTC) - timedelta(seconds=5),
        "synthetic": True,
    }


def test_market_downgrade_when_realtime_nbbo_required(monkeypatch, caplog) -> None:
    """Degraded NBBO requirement should downgrade instead of skipping when enabled."""

    engine = DummyLiveEngine()
    config = SimpleNamespace(
        min_quote_freshness_ms=2500,
        degraded_feed_mode="widen",
        degraded_feed_limit_widen_bps=50,
        execution_require_realtime_nbbo=True,
        execution_market_on_degraded=True,
        nbbo_required_for_limit=False,
    )
    monkeypatch.setattr(live_trading, "get_trading_config", lambda: config)
    monkeypatch.setattr(engine, "_broker_lock_suppressed", lambda **_: False)

    def _fake_submit_market(self, symbol, side, quantity, **kwargs):
        payload = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": "market",
        }
        payload.update(kwargs)
        client_oid = payload.get("client_order_id") or "test-market-order"
        payload["client_order_id"] = client_oid
        payload.setdefault("id", "test-market-order")
        self.last_submitted = payload
        return {
            "status": "accepted",
            "type": "market",
            "id": payload["id"],
            "client_order_id": client_oid,
        }

    monkeypatch.setattr(DummyLiveEngine, "submit_market_order", _fake_submit_market, raising=False)

    caplog.set_level(logging.INFO)

    order = engine.execute_order(
        "AAPL",
        "buy",
        5,
        order_type="limit",
        limit_price=100.0,
        quote=_synthetic_quote(),
        annotations={"price_source": "backup"},
    )

    assert order is not None
    assert engine.last_submitted is not None
    assert engine.last_submitted.get("order_type") == "market"
    assert "price" not in engine.last_submitted
    assert "limit_price" not in engine.last_submitted

    messages = [
        record.msg
        for record in caplog.records
        if record.name == "ai_trading.execution.live_trading"
    ]
    assert "ORDER_SKIPPED_PRICE_GATED" in messages
    assert "ORDER_DOWNGRADED_TO_MARKET" in messages
