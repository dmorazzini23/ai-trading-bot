import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from ai_trading.core import bot_engine
from ai_trading.core.bot_engine import BotState


def test_meta_confidence_cap_does_not_bypass_conf_threshold(monkeypatch):
    ctx = SimpleNamespace(
        signal_manager=SimpleNamespace(meta_confidence_capped=False),
        position_map={},
        portfolio_weights={},
        rebalance_buys={},
    )
    state = BotState()
    feat_df = pd.DataFrame({"close": [1.0, 1.1, 1.2]})

    monkeypatch.setattr(bot_engine, "pre_trade_checks", lambda *a, **k: True)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(bot_engine, "_fetch_feature_data", lambda *a, **k: (feat_df, feat_df, None))
    monkeypatch.setattr(bot_engine, "_model_feature_names", lambda *_a, **_k: [])
    monkeypatch.setattr(bot_engine, "_mark_primary_provider_fallback", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "_clear_primary_provider_fallback", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(bot_engine, "_exit_positions_if_needed", lambda *a, **k: False)
    monkeypatch.setattr(bot_engine, "_check_trade_frequency_limits", lambda *a, **k: False)
    monkeypatch.setattr(bot_engine, "_enter_short", lambda *a, **k: pytest.fail("unexpected short path"))
    monkeypatch.setattr(bot_engine, "get_buy_threshold", lambda: 0.6)
    monkeypatch.setattr(bot_engine, "get_conf_threshold", lambda: 0.55)
    monkeypatch.setattr(bot_engine, "_metafallback_confidence_cap", lambda: 0.2)

    def fake_eval(*_args, **_kwargs):
        ctx.signal_manager.meta_confidence_capped = True
        return 1.0, 0.35, "meta"

    monkeypatch.setattr(bot_engine, "_evaluate_trade_signal", fake_eval)

    monkeypatch.setattr(
        bot_engine,
        "_enter_long",
        lambda *_a, **_k: pytest.fail("entry should remain blocked below confidence threshold"),
    )

    result = bot_engine.trade_logic(ctx, state, "TEST", balance=100000.0, model=None, regime_ok=True)

    assert result is True


def test_trade_logic_blocks_entry_on_fallback_minute_data(monkeypatch, caplog):
    ctx = SimpleNamespace(
        signal_manager=SimpleNamespace(meta_confidence_capped=False),
        position_map={},
        portfolio_weights={},
        rebalance_buys={},
    )
    state = BotState()
    state.data_quality["TEST"] = {
        "using_fallback_provider": True,
        "provider_canonical": "yahoo",
        "fallback_contiguous": False,
        "gap_ratio": 0.002,
    }
    feat_df = pd.DataFrame({"close": [1.0, 1.1, 1.2]})

    monkeypatch.setenv("AI_TRADING_BLOCK_ENTRIES_ON_FALLBACK_MINUTE_DATA", "1")
    bot_engine._block_entries_on_fallback_minute_data.cache_clear()

    monkeypatch.setattr(bot_engine, "pre_trade_checks", lambda *a, **k: True)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(bot_engine, "_fetch_feature_data", lambda *a, **k: (feat_df, feat_df, None))
    monkeypatch.setattr(bot_engine, "_model_feature_names", lambda *_a, **_k: [])
    monkeypatch.setattr(bot_engine, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(bot_engine, "_exit_positions_if_needed", lambda *a, **k: False)
    monkeypatch.setattr(bot_engine, "_check_trade_frequency_limits", lambda *a, **k: False)
    monkeypatch.setattr(bot_engine, "get_buy_threshold", lambda: 0.6)
    monkeypatch.setattr(
        bot_engine,
        "_evaluate_trade_signal",
        lambda *_a, **_k: (1.0, 0.9, "fallback"),
    )
    monkeypatch.setattr(
        bot_engine,
        "_enter_long",
        lambda *_a, **_k: pytest.fail("entry should be blocked on fallback minute data"),
    )

    caplog.set_level(logging.WARNING)
    result = bot_engine.trade_logic(
        ctx,
        state,
        "TEST",
        balance=100000.0,
        model=None,
        regime_ok=True,
    )

    assert result is True
    blocked_logs = [
        record
        for record in caplog.records
        if record.message == "ENTRY_BLOCKED_DEGRADED_MINUTE_DATA"
    ]
    assert blocked_logs
    assert getattr(blocked_logs[-1], "side", "") == "buy"


def test_trade_logic_allows_reliable_contiguous_fallback_entry(monkeypatch):
    ctx = SimpleNamespace(
        signal_manager=SimpleNamespace(meta_confidence_capped=False),
        position_map={},
        portfolio_weights={},
        rebalance_buys={},
    )
    state = BotState()
    state.data_quality["TEST"] = {
        "using_fallback_provider": True,
        "provider_canonical": "yahoo",
        "price_reliable": True,
        "fallback_contiguous": True,
        "fallback_repaired": True,
        "gap_ratio": 0.0,
    }
    feat_df = pd.DataFrame({"close": [1.0, 1.1, 1.2]})

    monkeypatch.setenv("AI_TRADING_BLOCK_ENTRIES_ON_FALLBACK_MINUTE_DATA", "1")
    bot_engine._block_entries_on_fallback_minute_data.cache_clear()

    monkeypatch.setattr(bot_engine, "pre_trade_checks", lambda *a, **k: True)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(bot_engine, "_fetch_feature_data", lambda *a, **k: (feat_df, feat_df, None))
    monkeypatch.setattr(bot_engine, "_model_feature_names", lambda *_a, **_k: [])
    monkeypatch.setattr(bot_engine, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(bot_engine, "_exit_positions_if_needed", lambda *a, **k: False)
    monkeypatch.setattr(bot_engine, "_check_trade_frequency_limits", lambda *a, **k: False)
    monkeypatch.setattr(bot_engine, "get_buy_threshold", lambda: 0.6)
    monkeypatch.setattr(
        bot_engine,
        "_evaluate_trade_signal",
        lambda *_a, **_k: (1.0, 0.9, "fallback"),
    )

    entered: dict[str, bool] = {"called": False}

    def _fake_enter_long(*_args, **_kwargs):
        entered["called"] = True
        return True

    monkeypatch.setattr(bot_engine, "_enter_long", _fake_enter_long)

    result = bot_engine.trade_logic(
        ctx,
        state,
        "TEST",
        balance=100000.0,
        model=None,
        regime_ok=True,
    )

    assert result is True
    assert entered["called"] is True


def test_trade_logic_allows_position_management_on_fallback_data(monkeypatch):
    ctx = SimpleNamespace(
        signal_manager=SimpleNamespace(meta_confidence_capped=False),
        position_map={"TEST": SimpleNamespace(qty=5)},
        portfolio_weights={},
        rebalance_buys={},
    )
    state = BotState()
    state.data_quality["TEST"] = {
        "using_fallback_provider": True,
        "provider_canonical": "yahoo",
    }
    feat_df = pd.DataFrame({"close": [1.0, 1.1, 1.2], "atr": [0.2, 0.2, 0.2]})

    monkeypatch.setenv("AI_TRADING_BLOCK_ENTRIES_ON_FALLBACK_MINUTE_DATA", "1")
    bot_engine._block_entries_on_fallback_minute_data.cache_clear()

    monkeypatch.setattr(bot_engine, "pre_trade_checks", lambda *a, **k: True)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(bot_engine, "_fetch_feature_data", lambda *a, **k: (feat_df, feat_df, None))
    monkeypatch.setattr(bot_engine, "_model_feature_names", lambda *_a, **_k: [])
    monkeypatch.setattr(bot_engine, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(bot_engine, "_exit_positions_if_needed", lambda *a, **k: False)
    monkeypatch.setattr(bot_engine, "_check_trade_frequency_limits", lambda *a, **k: False)
    monkeypatch.setattr(bot_engine, "get_buy_threshold", lambda: 0.6)
    monkeypatch.setattr(
        bot_engine,
        "_evaluate_trade_signal",
        lambda *_a, **_k: (1.0, 0.9, "fallback"),
    )
    monkeypatch.setattr(
        bot_engine,
        "_enter_long",
        lambda *_a, **_k: pytest.fail("existing position path should not enter long"),
    )

    managed: dict[str, bool] = {"called": False}

    def _fake_manage(*_args, **_kwargs):
        managed["called"] = True
        return True

    monkeypatch.setattr(bot_engine, "_manage_existing_position", _fake_manage)

    result = bot_engine.trade_logic(
        ctx,
        state,
        "TEST",
        balance=100000.0,
        model=None,
        regime_ok=True,
    )

    assert result is True
    assert managed["called"] is True


def test_entry_expected_edge_gate_blocks_low_edge(monkeypatch):
    monkeypatch.setattr(bot_engine, "get_min_expected_edge_bps", lambda: 10.0)
    monkeypatch.setattr(
        bot_engine,
        "get_fallback_expected_edge_penalty_bps",
        lambda: 10.0,
    )
    monkeypatch.setattr(bot_engine, "_slippage_setting_bps", lambda: 0.0)
    monkeypatch.setattr(bot_engine, "get_entry_cost_buffer_bps", lambda: 0.0)
    monkeypatch.setattr(bot_engine, "get_entry_age_penalty_per_sec_bps", lambda: 0.0)
    monkeypatch.setattr(bot_engine, "get_entry_age_penalty_cap_bps", lambda: 0.0)

    allowed = bot_engine._entry_expected_edge_gate(
        side="buy",
        symbol="TEST",
        final_score=1.0,
        confidence=0.5,
        atr=0.1,
        price=100.0,
        fallback_used=True,
    )

    assert allowed is False


def test_entry_expected_edge_gate_allows_high_edge(monkeypatch):
    monkeypatch.setattr(bot_engine, "get_min_expected_edge_bps", lambda: 8.0)
    monkeypatch.setattr(
        bot_engine,
        "get_fallback_expected_edge_penalty_bps",
        lambda: 5.0,
    )
    monkeypatch.setattr(bot_engine, "_slippage_setting_bps", lambda: 0.0)
    monkeypatch.setattr(bot_engine, "get_entry_cost_buffer_bps", lambda: 0.0)
    monkeypatch.setattr(bot_engine, "get_entry_age_penalty_per_sec_bps", lambda: 0.0)
    monkeypatch.setattr(bot_engine, "get_entry_age_penalty_cap_bps", lambda: 0.0)

    allowed = bot_engine._entry_expected_edge_gate(
        side="sell_short",
        symbol="TEST",
        final_score=-1.0,
        confidence=0.75,
        atr=0.2,
        price=100.0,
        fallback_used=False,
    )

    assert allowed is True


def test_entry_expected_edge_gate_includes_quote_cost_components(monkeypatch):
    monkeypatch.setattr(bot_engine, "get_min_expected_edge_bps", lambda: 6.0)
    monkeypatch.setattr(bot_engine, "get_fallback_expected_edge_penalty_bps", lambda: 0.0)
    monkeypatch.setattr(bot_engine, "_slippage_setting_bps", lambda: 10.0)
    monkeypatch.setattr(bot_engine, "get_entry_cost_buffer_bps", lambda: 2.0)
    monkeypatch.setattr(bot_engine, "get_entry_age_penalty_per_sec_bps", lambda: 0.1)
    monkeypatch.setattr(bot_engine, "get_entry_age_penalty_cap_bps", lambda: 5.0)

    # Baseline expected edge ~= 15 bps; costs push required edge above baseline.
    allowed = bot_engine._entry_expected_edge_gate(
        side="sell_short",
        symbol="TEST",
        final_score=-1.0,
        confidence=0.75,
        atr=0.2,
        price=100.0,
        fallback_used=False,
        quote_details={"bid": 99.0, "ask": 101.0, "age_sec": 20.0},
    )

    assert allowed is False


def test_trade_logic_requires_flip_confirmation_before_entry(monkeypatch, caplog):
    ctx = SimpleNamespace(
        signal_manager=SimpleNamespace(meta_confidence_capped=False),
        position_map={},
        portfolio_weights={},
        rebalance_buys={},
    )
    state = BotState()
    state.last_entry_side["TEST"] = "short"
    feat_df = pd.DataFrame({"close": [1.0, 1.1, 1.2]})

    monkeypatch.setattr(bot_engine, "pre_trade_checks", lambda *a, **k: True)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine, "_fetch_feature_data", lambda *a, **k: (feat_df, feat_df, None)
    )
    monkeypatch.setattr(bot_engine, "_model_feature_names", lambda *_a, **_k: [])
    monkeypatch.setattr(bot_engine, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(bot_engine, "_exit_positions_if_needed", lambda *a, **k: False)
    monkeypatch.setattr(
        bot_engine, "_check_trade_frequency_limits", lambda *_a, **_k: False
    )
    monkeypatch.setattr(bot_engine, "get_buy_threshold", lambda: 0.5)
    monkeypatch.setattr(bot_engine, "get_conf_threshold", lambda: 0.5)
    monkeypatch.setattr(bot_engine, "get_fallback_entry_confidence_bonus", lambda: 0.0)
    monkeypatch.setattr(bot_engine, "get_entry_flip_confirm_signals", lambda: 2)
    monkeypatch.setattr(
        bot_engine,
        "_evaluate_trade_signal",
        lambda *_a, **_k: (1.0, 0.9, "flip_confirm"),
    )
    monkeypatch.setattr(bot_engine, "_entry_data_degraded", lambda *_a, **_k: (False, {}))

    entered: dict[str, int] = {"calls": 0}

    def _fake_enter_long(*_a, **_k):
        entered["calls"] += 1
        return True

    monkeypatch.setattr(bot_engine, "_enter_long", _fake_enter_long)

    caplog.set_level(logging.INFO)
    result_first = bot_engine.trade_logic(
        ctx, state, "TEST", balance=100000.0, model=None, regime_ok=True
    )
    result_second = bot_engine.trade_logic(
        ctx, state, "TEST", balance=100000.0, model=None, regime_ok=True
    )

    assert result_first is True
    assert result_second is True
    assert entered["calls"] == 1
    assert any(record.message == "ENTRY_FLIP_CONFIRM_PENDING" for record in caplog.records)


def test_trade_logic_blocks_entry_when_expectancy_negative(monkeypatch, caplog):
    ctx = SimpleNamespace(
        signal_manager=SimpleNamespace(meta_confidence_capped=False),
        position_map={},
        portfolio_weights={},
        rebalance_buys={},
    )
    state = BotState()
    state.current_regime = "bull"
    bucket_key = bot_engine._expectancy_bucket_key("TEST", "bull", "long")
    state.expectancy_history[bucket_key] = [-0.03, -0.02, -0.01]
    feat_df = pd.DataFrame({"close": [1.0, 1.1, 1.2]})

    monkeypatch.setattr(bot_engine, "pre_trade_checks", lambda *a, **k: True)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine, "_fetch_feature_data", lambda *a, **k: (feat_df, feat_df, None)
    )
    monkeypatch.setattr(bot_engine, "_model_feature_names", lambda *_a, **_k: [])
    monkeypatch.setattr(bot_engine, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(bot_engine, "_exit_positions_if_needed", lambda *a, **k: False)
    monkeypatch.setattr(
        bot_engine, "_check_trade_frequency_limits", lambda *_a, **_k: False
    )
    monkeypatch.setattr(bot_engine, "get_buy_threshold", lambda: 0.5)
    monkeypatch.setattr(bot_engine, "get_conf_threshold", lambda: 0.5)
    monkeypatch.setattr(bot_engine, "get_fallback_entry_confidence_bonus", lambda: 0.0)
    monkeypatch.setattr(bot_engine, "get_expectancy_filter_enabled", lambda: True)
    monkeypatch.setattr(bot_engine, "get_expectancy_min_samples", lambda: 2)
    monkeypatch.setattr(bot_engine, "get_expectancy_min_mean_pct", lambda: 0.0)
    monkeypatch.setattr(
        bot_engine,
        "_evaluate_trade_signal",
        lambda *_a, **_k: (1.0, 0.9, "expectancy"),
    )
    monkeypatch.setattr(bot_engine, "_entry_data_degraded", lambda *_a, **_k: (False, {}))
    monkeypatch.setattr(
        bot_engine,
        "_enter_long",
        lambda *_a, **_k: pytest.fail("entry should be blocked by expectancy filter"),
    )

    caplog.set_level(logging.INFO)
    result = bot_engine.trade_logic(
        ctx, state, "TEST", balance=100000.0, model=None, regime_ok=True
    )

    assert result is True
    assert any(record.message == "ENTRY_BLOCKED_EXPECTANCY" for record in caplog.records)


def test_trade_logic_raises_threshold_with_alpha_decay(monkeypatch, caplog):
    ctx = SimpleNamespace(
        signal_manager=SimpleNamespace(meta_confidence_capped=False),
        position_map={},
        portfolio_weights={},
        rebalance_buys={},
    )
    state = BotState()
    feat_df = pd.DataFrame({"close": [1.0, 1.1, 1.2]})
    now = datetime(2026, 2, 9, tzinfo=UTC)
    state.trade_history = [
        ("TEST", now - timedelta(minutes=10)),
        ("TEST", now - timedelta(minutes=5)),
    ]

    monkeypatch.setattr(bot_engine, "pre_trade_checks", lambda *a, **k: True)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine, "_fetch_feature_data", lambda *a, **k: (feat_df, feat_df, None)
    )
    monkeypatch.setattr(bot_engine, "_model_feature_names", lambda *_a, **_k: [])
    monkeypatch.setattr(bot_engine, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(bot_engine, "_exit_positions_if_needed", lambda *a, **k: False)
    monkeypatch.setattr(
        bot_engine, "_check_trade_frequency_limits", lambda *_a, **_k: False
    )
    monkeypatch.setattr(bot_engine, "get_buy_threshold", lambda: 0.5)
    monkeypatch.setattr(bot_engine, "get_conf_threshold", lambda: 0.5)
    monkeypatch.setattr(bot_engine, "get_fallback_entry_confidence_bonus", lambda: 0.0)
    monkeypatch.setattr(bot_engine, "get_alpha_decay_window_minutes", lambda: 30)
    monkeypatch.setattr(bot_engine, "get_alpha_decay_start_trades", lambda: 1)
    monkeypatch.setattr(bot_engine, "get_alpha_decay_threshold_step", lambda: 0.1)
    monkeypatch.setattr(bot_engine, "get_alpha_decay_max_bump", lambda: 0.2)
    monkeypatch.setattr(bot_engine, "get_alpha_decay_max_trades_window", lambda: 0)
    monkeypatch.setattr(
        bot_engine,
        "_evaluate_trade_signal",
        lambda *_a, **_k: (1.0, 0.55, "alpha_decay"),
    )
    monkeypatch.setattr(
        bot_engine,
        "_enter_long",
        lambda *_a, **_k: pytest.fail("entry should be skipped after alpha-decay threshold bump"),
    )

    caplog.set_level(logging.INFO)
    result = bot_engine.trade_logic(
        ctx,
        state,
        "TEST",
        balance=100000.0,
        model=None,
        regime_ok=True,
        now_provider=lambda: now,
    )

    assert result is True
    assert any(record.message == "ENTRY_THRESHOLD_RAISED_ALPHA_DECAY" for record in caplog.records)


def test_trade_logic_blocks_entry_when_alpha_decay_window_saturated(monkeypatch, caplog):
    ctx = SimpleNamespace(
        signal_manager=SimpleNamespace(meta_confidence_capped=False),
        position_map={},
        portfolio_weights={},
        rebalance_buys={},
    )
    state = BotState()
    feat_df = pd.DataFrame({"close": [1.0, 1.1, 1.2]})
    now = datetime(2026, 2, 9, tzinfo=UTC)
    state.trade_history = [
        ("TEST", now - timedelta(minutes=10)),
        ("TEST", now - timedelta(minutes=8)),
        ("TEST", now - timedelta(minutes=3)),
    ]

    monkeypatch.setattr(bot_engine, "pre_trade_checks", lambda *a, **k: True)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        bot_engine, "_fetch_feature_data", lambda *a, **k: (feat_df, feat_df, None)
    )
    monkeypatch.setattr(bot_engine, "_model_feature_names", lambda *_a, **_k: [])
    monkeypatch.setattr(bot_engine, "is_safe_mode_active", lambda: False)
    monkeypatch.setattr(bot_engine, "_exit_positions_if_needed", lambda *a, **k: False)
    monkeypatch.setattr(
        bot_engine, "_check_trade_frequency_limits", lambda *_a, **_k: False
    )
    monkeypatch.setattr(bot_engine, "get_buy_threshold", lambda: 0.5)
    monkeypatch.setattr(bot_engine, "get_conf_threshold", lambda: 0.5)
    monkeypatch.setattr(bot_engine, "get_fallback_entry_confidence_bonus", lambda: 0.0)
    monkeypatch.setattr(bot_engine, "get_alpha_decay_window_minutes", lambda: 30)
    monkeypatch.setattr(bot_engine, "get_alpha_decay_start_trades", lambda: 1)
    monkeypatch.setattr(bot_engine, "get_alpha_decay_threshold_step", lambda: 0.0)
    monkeypatch.setattr(bot_engine, "get_alpha_decay_max_bump", lambda: 0.0)
    monkeypatch.setattr(bot_engine, "get_alpha_decay_max_trades_window", lambda: 3)
    monkeypatch.setattr(
        bot_engine,
        "_evaluate_trade_signal",
        lambda *_a, **_k: (1.0, 0.9, "alpha_decay"),
    )
    monkeypatch.setattr(
        bot_engine,
        "_enter_long",
        lambda *_a, **_k: pytest.fail("entry should be blocked by alpha-decay saturation"),
    )

    caplog.set_level(logging.INFO)
    result = bot_engine.trade_logic(
        ctx,
        state,
        "TEST",
        balance=100000.0,
        model=None,
        regime_ok=True,
        now_provider=lambda: now,
    )

    assert result is True
    blocked = [record for record in caplog.records if record.message == "ENTRY_BLOCKED_ALPHA_DECAY"]
    assert blocked
