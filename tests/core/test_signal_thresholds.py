import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from ai_trading.core import bot_engine
from ai_trading.core.bot_engine import BotState


def test_meta_confidence_cap_lowers_buy_threshold(monkeypatch):
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

    def fake_eval(*_args, **_kwargs):
        ctx.signal_manager.meta_confidence_capped = True
        return 1.0, 0.35, "meta"

    monkeypatch.setattr(bot_engine, "_evaluate_trade_signal", fake_eval)

    captured: dict[str, object] = {}

    def fake_enter_long(*args):
        captured["called"] = True
        captured["confidence"] = args[6]
        return True

    monkeypatch.setattr(bot_engine, "_enter_long", fake_enter_long)

    result = bot_engine.trade_logic(ctx, state, "TEST", balance=100000.0, model=None, regime_ok=True)

    assert result is True
    assert captured.get("called") is True
    assert captured.get("confidence") == pytest.approx(0.35)


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
