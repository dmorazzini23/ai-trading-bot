from __future__ import annotations

import logging
from datetime import time
import types

import pytest

import ai_trading.core.bot_engine as bot_engine
from ai_trading.config.management import reload_trading_config


def test_enter_long_blocks_when_primary_provider_disabled(monkeypatch):
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: False,
        raising=False,
    )
    state = types.SimpleNamespace(degraded_providers=set())
    ctx = types.SimpleNamespace()
    blocked = bot_engine._enter_long(
        ctx,
        state,
        "AAPL",
        1000.0,
        object(),
        1.0,
        0.8,
        "test",
    )
    assert blocked is True


def test_enter_long_blocks_when_safe_mode_active(monkeypatch):
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(bot_engine, "is_safe_mode_active", lambda: True)
    monkeypatch.setattr(bot_engine, "safe_mode_reason", lambda: "provider_safe_mode")
    monkeypatch.setattr(bot_engine, "_safe_mode_blocks_trading", lambda: True)
    state = types.SimpleNamespace(degraded_providers=set())
    ctx = types.SimpleNamespace()
    blocked = bot_engine._enter_long(
        ctx,
        state,
        "AAPL",
        1000.0,
        object(),
        1.0,
        0.8,
        "test",
    )
    assert blocked is True


def test_should_skip_order_for_fallback_price(monkeypatch):
    state = types.SimpleNamespace(auth_skipped_symbols=set())
    assert bot_engine._should_skip_order_for_alpaca_unavailable(state, "AAPL", "yahoo")


def test_degraded_mode_allows_trading_on_safe_mode(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")
    from ai_trading.data import fetch as data_fetch_module
    from ai_trading.data import provider_monitor

    monkeypatch.setenv("TRADING__DEGRADED_FEED_MODE", "widen")
    reload_trading_config()

    caplog.set_level(logging.INFO)

    monkeypatch.setenv("PYTEST_RUNNING", "0")
    provider_monitor.activate_data_kill_switch(
        "minute_gap",
        provider="alpaca",
        metadata={"gap_ratio": 0.7},
    )
    monkeypatch.setenv("PYTEST_RUNNING", "1")

    orders: list[tuple[str, int, str, float | None]] = []

    class _DummyAPI:
        def list_positions(self):
            return []

        def get_account(self):
            return types.SimpleNamespace(
                equity=100000.0,
                portfolio_value=100000.0,
                buying_power=100000.0,
            )

    class _Logger:
        def __init__(self):
            self.entries: list[tuple] = []

        def log_entry(self, *args, **kwargs):
            self.entries.append((args, kwargs))

        def log_exit(self, *args, **kwargs):
            return None

    feat_df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0],
            "open": [99.5, 100.5, 101.5],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "volume": [1_000, 1_200, 1_400],
            "macd": [0.1, 0.2, 0.3],
            "atr": [1.0, 1.0, 1.0],
            "vwap": [100.2, 100.6, 101.0],
            "macds": [0.05, 0.05, 0.05],
            "sma_50": [99.0, 99.5, 100.0],
            "sma_200": [95.0, 95.5, 96.0],
        }
    )

    ctx = types.SimpleNamespace(
        signal_manager=types.SimpleNamespace(last_components=[]),
        data_fetcher=types.SimpleNamespace(get_daily_df=lambda *_a, **_k: feat_df.copy()),
        portfolio_weights={},
        api=_DummyAPI(),
        trade_logger=_Logger(),
        take_profit_targets={},
        stop_targets={},
        market_open=time(6, 30),
        market_close=time(13, 0),
        rebalance_buys={},
        config=types.SimpleNamespace(exposure_cap_aggressive=0.9),
    )

    state = bot_engine.BotState()
    state.position_cache = {}
    state.long_positions = set()
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: True,
        raising=False,
    )
    state.short_positions = set()

    monkeypatch.setattr(bot_engine, "pre_trade_checks", lambda *a, **k: True)
    monkeypatch.setattr(bot_engine, "_fetch_feature_data", lambda *a, **k: (feat_df.copy(), feat_df.copy(), None))

    def _fake_eval(ctx_obj, state_obj, _df, _symbol, _model):
        ctx_obj.signal_manager.last_components = [(1, 0.9, "safe_mode_widen")]
        return 1.0, 0.9, "safe_mode_widen"

    monkeypatch.setattr(bot_engine, "_evaluate_trade_signal", _fake_eval)
    monkeypatch.setattr(bot_engine, "_exit_positions_if_needed", lambda *a, **k: False)
    monkeypatch.setattr(bot_engine, "_check_trade_frequency_limits", lambda *a, **k: False)
    monkeypatch.setattr(bot_engine, "get_trade_cooldown_min", lambda: 0)
    monkeypatch.setattr(bot_engine, "get_buy_threshold", lambda: 0.5)
    monkeypatch.setattr(bot_engine, "_current_qty", lambda *a, **k: 0)
    monkeypatch.setattr(bot_engine, "_apply_sector_cap_qty", lambda _ctx, _symbol, qty, _price: qty)
    monkeypatch.setattr(bot_engine, "scaled_atr_stop", lambda *_a, **_k: (95.0, 105.0))
    monkeypatch.setattr(bot_engine, "is_high_vol_regime", lambda: False)
    monkeypatch.setattr(bot_engine, "get_take_profit_factor", lambda: 1.0)
    monkeypatch.setattr(bot_engine, "_record_trade_in_frequency_tracker", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "_PRICE_SOURCE", {}, raising=False)
    monkeypatch.setattr(bot_engine, "_resolve_order_quote", lambda *a, **k: (101.5, "yahoo"))

    def _fake_submit(_ctx, sym, qty, side, price=None, **_exec):
        orders.append((sym, qty, side, price))
        return types.SimpleNamespace(id="order-1")

    monkeypatch.setattr(bot_engine, "submit_order", _fake_submit)

    try:
        result = bot_engine.trade_logic(
            ctx,
            state,
            "AAPL",
            balance=100000.0,
            model=None,
            regime_ok=True,
        )
    finally:
        provider_monitor._SAFE_MODE_ACTIVE = False
        provider_monitor._SAFE_MODE_REASON = None
        provider_monitor._SAFE_MODE_HEALTHY_PASSES = 0
        data_fetch_module.clear_safe_mode_cycle(reason="test_cleanup")
        data_fetch_module._SAFE_MODE_LOGGED.clear()

    assert result is True
    assert orders
    assert any(record.message == "ORDER_USING_FALLBACK_PRICE" for record in caplog.records)
    assert all(record.message != "SAFE_MODE_BLOCK" for record in caplog.records)


def test_pre_trade_gate_allows_backup_when_failsoft(monkeypatch):
    monkeypatch.setenv("TRADING__SAFE_MODE_FAILSOFT", "true")
    reload_trading_config()

    provider_state = {
        "status": "disabled",
        "using_backup": True,
        "reason": "minute_gap",
        "timeframes": {"1m": True},
    }
    monkeypatch.setattr(bot_engine.runtime_state, "observe_data_provider_state", lambda: provider_state)
    monkeypatch.setattr(bot_engine.runtime_state, "observe_quote_status", lambda: {"allowed": True})
    monkeypatch.setattr(bot_engine.provider_monitor, "is_safe_mode_active", lambda: True)
    monkeypatch.setattr(bot_engine.provider_monitor, "safe_mode_reason", lambda: "minute_gap")
    monkeypatch.setattr(bot_engine.provider_monitor, "safe_mode_degraded_only", lambda: True)
    monkeypatch.setattr(bot_engine, "safe_mode_reason", lambda: "minute_gap")

    assert bot_engine._pre_trade_gate() is False


def test_pre_trade_gate_blocks_without_failsoft(monkeypatch):
    monkeypatch.setenv("TRADING__SAFE_MODE_FAILSOFT", "false")
    monkeypatch.setenv("TRADING__DEGRADED_FEED_MODE", "block")
    reload_trading_config()

    provider_state = {
        "status": "disabled",
        "using_backup": True,
        "reason": "minute_gap",
        "timeframes": {"1m": True},
    }
    monkeypatch.setattr(bot_engine.runtime_state, "observe_data_provider_state", lambda: provider_state)
    monkeypatch.setattr(bot_engine.runtime_state, "observe_quote_status", lambda: {"allowed": True})
    monkeypatch.setattr(bot_engine.provider_monitor, "is_safe_mode_active", lambda: True)
    monkeypatch.setattr(bot_engine.provider_monitor, "safe_mode_reason", lambda: "minute_gap")
    monkeypatch.setattr(bot_engine.provider_monitor, "safe_mode_degraded_only", lambda: True)
    monkeypatch.setattr(bot_engine, "safe_mode_reason", lambda: "minute_gap")

    assert bot_engine._pre_trade_gate() is True


def test_last_close_allowed_when_degraded(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")

    monkeypatch.setenv("TRADING__DEGRADED_FEED_MODE", "widen")
    monkeypatch.setenv("EXECUTION_ALLOW_LAST_CLOSE", "false")
    reload_trading_config()

    monkeypatch.setattr(bot_engine, "is_safe_mode_active", lambda: True)
    monkeypatch.setattr(bot_engine, "safe_mode_reason", lambda: "provider_safe_mode")
    monkeypatch.setattr(bot_engine, "_safe_mode_blocks_trading", lambda: False)

    caplog.set_level(logging.WARNING)

    feat_df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0],
            "open": [99.5, 100.5, 101.5],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "volume": [1_000, 1_200, 1_400],
            "macd": [0.1, 0.2, 0.3],
            "atr": [1.0, 1.0, 1.0],
            "vwap": [100.2, 100.6, 101.0],
            "macds": [0.05, 0.05, 0.05],
            "sma_50": [99.0, 99.5, 100.0],
            "sma_200": [95.0, 95.5, 96.0],
        }
    )

    ctx = types.SimpleNamespace(
        signal_manager=types.SimpleNamespace(last_components=[]),
        trade_logger=types.SimpleNamespace(log_entry=lambda *a, **k: None, log_exit=lambda *a, **k: None),
        take_profit_targets={},
        stop_targets={},
        market_open=time(6, 30),
        market_close=time(13, 0),
    )
    ctx.portfolio_weights = {}
    assert hasattr(ctx, "portfolio_weights")
    ctx.api = types.SimpleNamespace(
        list_positions=lambda: [],
        get_account=lambda: types.SimpleNamespace(portfolio_value=100000.0, equity=100000.0),
    )
    ctx.config = types.SimpleNamespace(exposure_cap_aggressive=0.9)
    state = bot_engine.BotState()
    state.position_cache = {}
    state.long_positions = set()

    monkeypatch.setattr(
        bot_engine,
        "_resolve_order_quote",
        lambda *a, **k: (None, "alpaca_last_close"),
    )

    def _fake_normalize(*_a, **_k):
        return None, "alpaca_last_close", {"last_close_only": True}

    monkeypatch.setattr(bot_engine, "_normalize_order_quote_payload", _fake_normalize)
    monkeypatch.setattr(bot_engine, "_should_skip_order_for_alpaca_unavailable", lambda *a, **k: False)
    monkeypatch.setattr(
        bot_engine,
        "_evaluate_data_gating",
        lambda *a, **k: types.SimpleNamespace(block=False, reasons=(), annotations={}, size_cap=None),
    )
    monkeypatch.setattr(bot_engine, "_synthetic_quote_decision", lambda *a, **k: True)
    orders: list[types.SimpleNamespace] = []

    def _record_order(*_a, **_k):
        order = types.SimpleNamespace(id="order-1")
        orders.append(order)
        return order

    monkeypatch.setattr(bot_engine, "submit_order", _record_order)
    monkeypatch.setattr(bot_engine, "_set_price_source", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "_clear_cached_yahoo_fallback", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "_mark_primary_provider_fallback", lambda *a, **k: None)

    assert bot_engine._allow_last_close_execution() is True

    blocked = bot_engine._enter_long(
        ctx,
        state,
        "AAPL",
        balance=100000.0,
        feat_df=feat_df,
        final_score=1.0,
        conf=0.9,
        strat="last_close",
    )

    assert orders
    assert all(record.message != "ORDER_SKIPPED_NONRETRYABLE_DETAIL" for record in caplog.records)


def test_failsoft_low_coverage_guard(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(bot_engine.provider_monitor, "is_safe_mode_active", lambda: True)
    monkeypatch.setattr(bot_engine.provider_monitor, "safe_mode_degraded_only", lambda: True)

    allowed = bot_engine._should_failsoft_allow_low_coverage(
        fallback_used=True,
        relax_ratio=0.5,
        symbol="AAPL",
        coverage_threshold=100,
        actual_bars=20,
        fallback_feed="yahoo",
        fallback_provider="yahoo",
    )

    assert allowed is True
    assert any(record.message == "DEGRADED_COVERAGE_FAILSOFT_ALLOW" for record in caplog.records)
