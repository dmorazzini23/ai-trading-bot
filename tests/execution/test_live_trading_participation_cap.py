from __future__ import annotations

import types
from types import SimpleNamespace

from ai_trading.execution import live_trading


def _build_engine(monkeypatch):
    engine = live_trading.ExecutionEngine.__new__(live_trading.ExecutionEngine)
    engine.is_initialized = True
    engine.shadow_mode = False
    engine.execution_mode = "paper"
    engine._explicit_mode = None
    engine._explicit_shadow = None
    engine.stats = {
        "total_execution_time": 0.0,
        "total_orders": 0,
        "successful_orders": 0,
        "failed_orders": 0,
        "retry_count": 0,
        "capacity_skips": 0,
        "skipped_orders": 0,
        "circuit_breaker_trips": 0,
    }
    engine.retry_config = {
        "max_attempts": 1,
        "base_delay": 0.01,
        "max_delay": 0.01,
        "exponential_base": 2.0,
    }
    engine.circuit_breaker = {
        "failure_count": 0,
        "max_failures": 5,
        "reset_time": 300,
        "last_failure": None,
        "is_open": False,
    }
    engine.order_manager = object()
    engine._pending_orders = {}
    engine._order_signal_meta = {}
    engine._cycle_account = None
    engine._cycle_account_fetched = True
    engine.order_ttl_seconds = 0
    engine.marketable_limit_slippage_bps = 10
    engine.max_participation_rate = None
    engine.trading_client = SimpleNamespace()
    engine.settings = None
    engine.config = None
    engine.price_provider_order = ("alpaca_quote",)
    engine.data_feed_intraday = "iex"
    engine.slippage_limit_bps = 0

    monkeypatch.setattr(engine, "_ensure_initialized", lambda: True)
    monkeypatch.setattr(engine, "_pre_execution_checks", lambda: True)
    monkeypatch.setattr(engine, "_pre_execution_order_checks", lambda _order: True)
    monkeypatch.setattr(engine, "_normalized_order_side", lambda _side: "buy")
    monkeypatch.setattr(engine, "_map_core_side", lambda _side: "buy")
    monkeypatch.setattr(engine, "_broker_lock_suppressed", lambda **_kwargs: False)
    monkeypatch.setattr(engine, "_supports_asset_class", lambda: False)
    monkeypatch.setattr(engine, "_fetch_broker_state", lambda: ([], []))
    monkeypatch.setattr(engine, "_fetch_account_state", lambda: ({}, None))
    monkeypatch.setattr(engine, "_update_position_tracker_snapshot", lambda _positions: None)
    monkeypatch.setattr(engine, "_update_broker_snapshot", lambda _orders, _positions: None)

    monkeypatch.setattr(live_trading, "_require_bid_ask_quotes", lambda: False)
    monkeypatch.setattr(live_trading.provider_monitor, "is_disabled", lambda *_a, **_k: False)
    monkeypatch.setattr(live_trading, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(live_trading, "guard_shadow_active", lambda: False)
    monkeypatch.setattr(
        live_trading,
        "get_trading_config",
        lambda: SimpleNamespace(
            nbbo_required_for_limit=False,
            execution_require_realtime_nbbo=False,
            execution_market_on_degraded=False,
            degraded_feed_mode="widen",
            degraded_feed_limit_widen_bps=0,
            min_quote_freshness_ms=1500,
        ),
    )
    monkeypatch.setattr(
        live_trading,
        "_call_preflight_capacity",
        lambda _symbol, _side, _price_hint, quantity, _broker, _account: live_trading.CapacityCheck(
            True, int(quantity), None
        ),
    )

    submitted: dict[str, int] = {}

    def _submit_limit_order(self, _symbol, _side, qty, limit_price=None, **_kwargs):
        submitted["qty"] = int(qty)
        return {
            "id": "order-1",
            "client_order_id": "cid-1",
            "status": "filled",
            "qty": int(qty),
            "filled_qty": int(qty),
            "symbol": "AAPL",
            "limit_price": limit_price,
        }

    monkeypatch.setattr(
        engine,
        "submit_limit_order",
        types.MethodType(_submit_limit_order, engine),
    )
    return engine, submitted


def test_execute_order_scales_qty_when_participation_exceeded(monkeypatch):
    engine, submitted = _build_engine(monkeypatch)

    result = engine.execute_order(
        "AAPL",
        "buy",
        120,
        order_type="limit",
        limit_price=100.0,
        max_participation_rate=0.05,
        rolling_volume=1000,
        participation_mode="scale",
    )

    assert result is not None
    assert submitted["qty"] == 50


def test_execute_order_blocks_when_participation_mode_block(monkeypatch):
    engine, submitted = _build_engine(monkeypatch)

    result = engine.execute_order(
        "AAPL",
        "buy",
        120,
        order_type="limit",
        limit_price=100.0,
        max_participation_rate=0.05,
        rolling_volume=1000,
        participation_mode="block",
    )

    assert result is None
    assert submitted == {}


def test_refresh_settings_sets_participation_rate(monkeypatch):
    engine = live_trading.ExecutionEngine.__new__(live_trading.ExecutionEngine)
    engine._explicit_mode = None
    engine._explicit_shadow = None
    engine.execution_mode = "paper"
    engine.shadow_mode = False
    monkeypatch.setattr(
        live_trading,
        "get_execution_settings",
        lambda: SimpleNamespace(
            mode="paper",
            shadow_mode=False,
            order_timeout_seconds=30,
            slippage_limit_bps=5,
            order_ttl_seconds=12,
            marketable_limit_slippage_bps=8,
            max_participation_rate=0.03,
            price_provider_order=("alpaca_quote",),
            data_feed_intraday="iex",
        ),
    )

    engine._refresh_settings()

    assert engine.max_participation_rate == 0.03
