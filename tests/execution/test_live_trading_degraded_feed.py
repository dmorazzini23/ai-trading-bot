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
        execution_require_realtime_nbbo=False,
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
        execution_require_realtime_nbbo=False,
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


def test_nbbo_required_blocks_degraded_quotes(monkeypatch, caplog) -> None:
    """NBBO requirement should block synthetic quotes even when a limit is provided."""

    engine = DummyLiveEngine()
    monkeypatch.setenv("TRADING__DEGRADED_FEED_MODE", "block")
    monkeypatch.setenv("NBBO_REQUIRED_FOR_LIMIT", "true")
    from ai_trading.config.management import reload_trading_config

    reload_trading_config()

    caplog.set_level(logging.INFO)
    monkeypatch.setattr(engine, "_broker_lock_suppressed", lambda **_: False)

    result = engine.execute_order(
        "MSFT",
        "buy",
        5,
        order_type="limit",
        limit_price=300.0,
        quote=_quote_payload(),
        annotations={"price_source": "backup"},
    )

    assert result is None
    assert engine.last_submitted is None

    emitted = {rec.msg for rec in caplog.records}
    assert "DEGRADED_FEED_BLOCK_ENTRY" in emitted or "ORDER_SKIPPED_PRICE_GATED" in emitted

    monkeypatch.delenv("TRADING__DEGRADED_FEED_MODE", raising=False)
    monkeypatch.delenv("NBBO_REQUIRED_FOR_LIMIT", raising=False)
    reload_trading_config()


def test_broker_kwargs_preserve_degraded_hints() -> None:
    """Filtered kwargs must retain annotations and degraded pricing hints."""

    payload = {
        "annotations": {"price_source": "backup"},
        "using_fallback_price": True,
        "price_hint": 101.23,
        "client_order_id": "abc123",
    }

    filtered = live_trading._broker_kwargs_for_route("limit", payload)

    assert filtered is not payload
    assert filtered["annotations"]["price_source"] == "backup"
    assert filtered["using_fallback_price"] is True
    assert filtered["price_hint"] == pytest.approx(101.23)


def test_realtime_nbbo_gate_skips_degraded_openings(monkeypatch, caplog) -> None:
    """Require real-time NBBO should skip degraded openings and emit diagnostic."""

    engine = DummyLiveEngine()
    config = SimpleNamespace(
        min_quote_freshness_ms=1500,
        degraded_feed_mode="widen",
        degraded_feed_limit_widen_bps=0,
        execution_require_realtime_nbbo=True,
        execution_market_on_degraded=False,
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

    gated_record = next(
        (rec for rec in caplog.records if rec.msg == "ORDER_SKIPPED_PRICE_GATED"),
        None,
    )
    assert gated_record is not None
    assert getattr(gated_record, "reason", None) == "realtime_nbbo_required"
    assert getattr(gated_record, "provider", None) == "backup/synthetic"
    assert getattr(gated_record, "degraded", None) is True


def test_market_on_degraded_converts_limit_to_market(monkeypatch, caplog) -> None:
    """Opt-in should convert degraded entries into market submissions."""

    engine = DummyLiveEngine()
    config = SimpleNamespace(
        min_quote_freshness_ms=0,
        degraded_feed_mode="widen",
        degraded_feed_limit_widen_bps=0,
        execution_require_realtime_nbbo=False,
        execution_market_on_degraded=True,
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
    submitted_type = engine.last_submitted.get("type") or engine.last_submitted.get("order_type")
    assert submitted_type == "market"

    downgrade_record = next(
        (rec for rec in caplog.records if rec.msg == "ORDER_DOWNGRADED_TO_MARKET"),
        None,
    )
    assert downgrade_record is not None
    assert getattr(downgrade_record, "provider", None) == "backup/synthetic"
    assert getattr(downgrade_record, "degraded", None) is True
    assert getattr(downgrade_record, "mode", None) == "widen"


def test_execute_order_logs_real_order_id(monkeypatch, caplog) -> None:
    """Execution logs should include the normalized broker order identifier."""

    engine = DummyLiveEngine()
    config = SimpleNamespace(
        min_quote_freshness_ms=1500,
        degraded_feed_mode="widen",
        degraded_feed_limit_widen_bps=0,
        execution_require_realtime_nbbo=False,
        execution_market_on_degraded=False,
    )
    monkeypatch.setattr(live_trading, "get_trading_config", lambda: config)
    monkeypatch.setattr(live_trading.provider_monitor, "is_disabled", lambda *_a, **_k: False)

    caplog.set_level(logging.INFO)
    monkeypatch.setattr(engine, "_broker_lock_suppressed", lambda **_: False)

    fresh_quote = _quote_payload()
    fresh_quote["synthetic"] = False

    result = engine.execute_order(
        "AAPL",
        "buy",
        1,
        order_type="market",
        quote=fresh_quote,
    )

    assert result is not None

    exec_record = next(
        (rec for rec in caplog.records if rec.msg == "EXEC_ENGINE_EXECUTE_ORDER"),
        None,
    )
    assert exec_record is not None
    assert getattr(exec_record, "order_id", None) == "test-order"
