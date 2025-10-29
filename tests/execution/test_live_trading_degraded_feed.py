"""Tests for degraded-feed limit pricing and logging."""

from collections import defaultdict
from datetime import UTC, datetime
from types import SimpleNamespace

import logging

import pytest

from ai_trading.execution import live_trading


class DummyLiveEngine(live_trading.LiveTradingExecutionEngine):
    """Minimal live engine with predictable broker interactions."""

    def __init__(self) -> None:
        super().__init__(ctx=None)
        self.is_initialized = True
        self.shadow_mode = False
        self.stats = defaultdict(float)
        self._cycle_account = {
            "pattern_day_trader": False,
            "daytrade_limit": 3,
            "daytrade_count": 0,
        }
        self.trading_client = SimpleNamespace(
            get_asset=lambda _symbol: SimpleNamespace(shortable=True)
        )
        self.last_submitted: dict | None = None

    def _refresh_settings(self) -> None:  # pragma: no cover - overridden to no-op
        return None

    def _ensure_initialized(self) -> bool:  # pragma: no cover - deterministic
        return True

    def _pre_execution_order_checks(self, _order: dict | None) -> bool:
        return True

    def _pre_execution_checks(self) -> bool:  # pragma: no cover - deterministic
        return True

    def _execute_with_retry(self, submit_fn, order: dict):  # pragma: no cover - deterministic
        """Capture submitted order payload and return stub response."""

        self.last_submitted = dict(order)
        return {
            "status": "accepted",
            "id": "test-order",
            "symbol": order.get("symbol"),
            "side": order.get("side"),
        }

    def _get_account_snapshot(self):  # pragma: no cover - deterministic
        return dict(self._cycle_account)


@pytest.fixture(autouse=True)
def patch_shared_guards(monkeypatch):
    """Stub shared guard helpers for predictable behaviour."""

    monkeypatch.setattr(live_trading, "_safe_mode_guard", lambda *a, **k: False)
    monkeypatch.setattr(live_trading, "_require_bid_ask_quotes", lambda: False)
    monkeypatch.setattr(live_trading, "quote_fresh_enough", lambda *a, **k: True)
    monkeypatch.setattr(live_trading, "guard_shadow_active", lambda: False)
    monkeypatch.setattr(live_trading, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(
        live_trading, "_call_preflight_capacity",
        lambda *a, **k: live_trading.CapacityCheck(True, int(a[3]), None),
    )
    monkeypatch.setattr(
        live_trading.provider_monitor,
        "is_disabled",
        lambda *_a, **_k: True,
    )


def _quote_payload() -> dict:
    return {
        "bid": 99.0,
        "ask": 101.0,
        "ts": datetime.now(UTC),
        "synthetic": True,
    }


def test_degraded_feed_widen_logs_basis(monkeypatch, caplog) -> None:
    """Degraded feed should widen limit pricing and log basis metadata."""

    engine = DummyLiveEngine()
    config = SimpleNamespace(
        min_quote_freshness_ms=1500,
        degraded_feed_mode="widen",
        degraded_feed_limit_widen_bps=12,
    )
    monkeypatch.setattr(live_trading, "get_trading_config", lambda: config)

    caplog.set_level(logging.INFO)

    monkeypatch.setattr(engine, "_broker_lock_suppressed", lambda **_: False)

    result = engine.execute_order(
        "AAPL",
        "buy",
        10,
        order_type="limit",
        limit_price=100.0,
        quote=_quote_payload(),
        annotations={"price_source": "backup"},
    )

    assert result is not None
    assert engine.last_submitted is not None
    submitted_price = engine.last_submitted.get("price") or engine.last_submitted.get("limit_price")
    assert submitted_price is not None and submitted_price > 100.0

    limit_records = [rec for rec in caplog.records if rec.msg == "LIMIT_BASIS"]
    assert limit_records, "LIMIT_BASIS log not emitted"
    entry = limit_records[-1]
    assert entry.degraded is True
    assert entry.mode == "widen"
    assert entry.widen_bps == 12
    assert entry.limit is not None


def test_degraded_feed_block_prevents_submission(monkeypatch, caplog) -> None:
    """Block mode should prevent new orders and emit diagnostics."""

    engine = DummyLiveEngine()
    config = SimpleNamespace(
        min_quote_freshness_ms=1500,
        degraded_feed_mode="block",
        degraded_feed_limit_widen_bps=8,
    )
    monkeypatch.setattr(live_trading, "get_trading_config", lambda: config)

    caplog.set_level(logging.INFO)

    monkeypatch.setattr(engine, "_broker_lock_suppressed", lambda **_: False)

    result = engine.execute_order(
        "AAPL",
        "buy",
        10,
        order_type="limit",
        limit_price=100.0,
        quote=_quote_payload(),
        annotations={"price_source": "backup"},
    )

    assert result is None
    assert engine.last_submitted is None

    limit_records = [rec for rec in caplog.records if rec.msg == "LIMIT_BASIS"]
    assert limit_records, "LIMIT_BASIS log not emitted"
    entry = limit_records[-1]
    assert entry.mode == "block"
    assert entry.degraded is True

    block_records = [rec for rec in caplog.records if rec.msg == "DEGRADED_FEED_BLOCK_ENTRY"]
    assert block_records, "Block diagnostics not emitted"
