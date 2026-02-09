from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from ai_trading.core import bot_engine
from ai_trading.core.bot_engine import BotState


def test_reversal_exit_blocked_during_min_hold(monkeypatch):
    state = BotState()
    state.position_entry_times["AAPL"] = datetime.now(UTC) - timedelta(seconds=30)
    ctx = SimpleNamespace(
        trade_logger=SimpleNamespace(log_exit=lambda *_a, **_k: None),
        stop_targets={},
        take_profit_targets={},
        rebalance_buys={},
    )
    exit_calls = {"count": 0}

    monkeypatch.setattr(bot_engine, "get_min_position_hold_seconds", lambda: 300)
    monkeypatch.setattr(bot_engine, "get_conf_threshold", lambda: 0.5)
    monkeypatch.setattr(bot_engine, "_should_hold_position", lambda *_a, **_k: False)
    monkeypatch.setattr(bot_engine, "get_latest_close", lambda *_a, **_k: 100.0)
    monkeypatch.setattr(
        bot_engine,
        "send_exit_order",
        lambda *_a, **_k: exit_calls.__setitem__("count", exit_calls["count"] + 1),
    )

    exited = bot_engine._exit_positions_if_needed(
        ctx=ctx,
        state=state,
        symbol="AAPL",
        feat_df=pd.DataFrame({"close": [100.0]}),
        final_score=-1.0,
        conf=0.9,
        current_qty=10,
    )

    assert exited is False
    assert exit_calls["count"] == 0


def test_reversal_exit_allowed_after_min_hold(monkeypatch):
    state = BotState()
    state.position_entry_times["AAPL"] = datetime.now(UTC) - timedelta(seconds=600)
    ctx = SimpleNamespace(
        trade_logger=SimpleNamespace(log_exit=lambda *_a, **_k: None),
        stop_targets={},
        take_profit_targets={},
        rebalance_buys={},
    )
    exit_calls = {"count": 0}

    monkeypatch.setattr(bot_engine, "get_min_position_hold_seconds", lambda: 300)
    monkeypatch.setattr(bot_engine, "get_conf_threshold", lambda: 0.5)
    monkeypatch.setattr(bot_engine, "_should_hold_position", lambda *_a, **_k: False)
    monkeypatch.setattr(bot_engine, "get_latest_close", lambda *_a, **_k: 100.0)
    monkeypatch.setattr(
        bot_engine,
        "send_exit_order",
        lambda *_a, **_k: exit_calls.__setitem__("count", exit_calls["count"] + 1),
    )

    exited = bot_engine._exit_positions_if_needed(
        ctx=ctx,
        state=state,
        symbol="AAPL",
        feat_df=pd.DataFrame({"close": [100.0]}),
        final_score=-1.0,
        conf=0.9,
        current_qty=10,
    )

    assert exited is True
    assert exit_calls["count"] == 1


def test_should_exit_blocks_trailing_stop_during_min_hold(monkeypatch):
    state = BotState()
    state.position_entry_times["AAPL"] = datetime.now(UTC) - timedelta(seconds=10)
    ctx = SimpleNamespace(
        rebalance_buys={},
        stop_targets={},
        take_profit_targets={},
    )

    monkeypatch.setattr(bot_engine, "get_min_position_hold_seconds", lambda: 300)
    monkeypatch.setattr(bot_engine, "_current_qty", lambda *_a, **_k: 5)
    monkeypatch.setattr(bot_engine, "update_trailing_stop", lambda *_a, **_k: "exit_long")

    should_exit, exit_qty, reason = bot_engine.should_exit(
        ctx=ctx,
        state=state,
        symbol="AAPL",
        price=100.0,
        atr=1.0,
    )

    assert should_exit is False
    assert exit_qty == 0
    assert reason == ""


def test_should_exit_allows_stop_loss_during_min_hold(monkeypatch):
    state = BotState()
    state.position_entry_times["AAPL"] = datetime.now(UTC) - timedelta(seconds=10)
    ctx = SimpleNamespace(
        rebalance_buys={},
        stop_targets={"AAPL": 100.0},
        take_profit_targets={},
    )

    monkeypatch.setattr(bot_engine, "get_min_position_hold_seconds", lambda: 300)
    monkeypatch.setattr(bot_engine, "_current_qty", lambda *_a, **_k: 5)

    should_exit, exit_qty, reason = bot_engine.should_exit(
        ctx=ctx,
        state=state,
        symbol="AAPL",
        price=99.5,
        atr=1.0,
    )

    assert should_exit is True
    assert exit_qty == 5
    assert reason == "stop_loss"


def test_should_exit_defers_take_profit_during_min_hold_without_bypass(monkeypatch):
    state = BotState()
    state.position_entry_times["AAPL"] = datetime.now(UTC) - timedelta(seconds=10)
    ctx = SimpleNamespace(
        rebalance_buys={},
        stop_targets={},
        take_profit_targets={"AAPL": 101.0},
    )

    monkeypatch.setattr(bot_engine, "get_min_position_hold_seconds", lambda: 300)
    monkeypatch.setattr(bot_engine, "get_take_profit_min_hold_bypass_pct", lambda: 0.03)
    monkeypatch.setattr(bot_engine, "get_take_profit_exit_fraction", lambda: 1.0)
    monkeypatch.setattr(bot_engine, "_current_qty", lambda *_a, **_k: 10)
    monkeypatch.setattr(bot_engine, "_position_unrealized_pct", lambda *_a, **_k: 0.01)

    should_exit, exit_qty, reason = bot_engine.should_exit(
        ctx=ctx,
        state=state,
        symbol="AAPL",
        price=101.5,
        atr=1.0,
    )

    assert should_exit is False
    assert exit_qty == 0
    assert reason == ""


def test_should_exit_allows_take_profit_during_min_hold_with_bypass(monkeypatch):
    state = BotState()
    state.position_entry_times["AAPL"] = datetime.now(UTC) - timedelta(seconds=10)
    ctx = SimpleNamespace(
        rebalance_buys={},
        stop_targets={},
        take_profit_targets={"AAPL": 101.0},
    )

    monkeypatch.setattr(bot_engine, "get_min_position_hold_seconds", lambda: 300)
    monkeypatch.setattr(bot_engine, "get_take_profit_min_hold_bypass_pct", lambda: 0.03)
    monkeypatch.setattr(bot_engine, "get_take_profit_exit_fraction", lambda: 0.5)
    monkeypatch.setattr(bot_engine, "_current_qty", lambda *_a, **_k: 10)
    monkeypatch.setattr(bot_engine, "_position_unrealized_pct", lambda *_a, **_k: 0.05)

    should_exit, exit_qty, reason = bot_engine.should_exit(
        ctx=ctx,
        state=state,
        symbol="AAPL",
        price=101.5,
        atr=1.0,
    )

    assert should_exit is True
    assert exit_qty == 5
    assert reason == "take_profit"


def test_reversal_exit_allows_loser_cut_during_min_hold(monkeypatch):
    state = BotState()
    state.position_entry_times["AAPL"] = datetime.now(UTC) - timedelta(seconds=30)
    ctx = SimpleNamespace(
        trade_logger=SimpleNamespace(log_exit=lambda *_a, **_k: None),
        stop_targets={},
        take_profit_targets={},
        rebalance_buys={},
    )
    exit_calls = {"count": 0}

    monkeypatch.setattr(bot_engine, "get_min_position_hold_seconds", lambda: 300)
    monkeypatch.setattr(bot_engine, "get_min_hold_loser_cut_pct", lambda: 0.02)
    monkeypatch.setattr(bot_engine, "get_conf_threshold", lambda: 0.5)
    monkeypatch.setattr(bot_engine, "_should_hold_position", lambda *_a, **_k: False)
    monkeypatch.setattr(bot_engine, "get_latest_close", lambda *_a, **_k: 100.0)
    monkeypatch.setattr(bot_engine, "_position_unrealized_pct", lambda *_a, **_k: -0.03)
    monkeypatch.setattr(
        bot_engine,
        "send_exit_order",
        lambda *_a, **_k: exit_calls.__setitem__("count", exit_calls["count"] + 1),
    )

    exited = bot_engine._exit_positions_if_needed(
        ctx=ctx,
        state=state,
        symbol="AAPL",
        feat_df=pd.DataFrame({"close": [100.0]}),
        final_score=-1.0,
        conf=0.9,
        current_qty=10,
    )

    assert exited is True
    assert exit_calls["count"] == 1


def test_short_reversal_holds_when_short_trend_still_valid(monkeypatch):
    state = BotState()
    state.position_entry_times["AAPL"] = datetime.now(UTC) - timedelta(seconds=600)
    ctx = SimpleNamespace(
        trade_logger=SimpleNamespace(log_exit=lambda *_a, **_k: None),
        stop_targets={},
        take_profit_targets={},
        rebalance_buys={},
    )
    exit_calls = {"count": 0}

    monkeypatch.setattr(bot_engine, "get_min_position_hold_seconds", lambda: 300)
    monkeypatch.setattr(bot_engine, "get_conf_threshold", lambda: 0.5)
    monkeypatch.setattr(bot_engine, "_should_hold_short_position", lambda *_a, **_k: True)
    monkeypatch.setattr(bot_engine, "get_latest_close", lambda *_a, **_k: 100.0)
    monkeypatch.setattr(
        bot_engine,
        "send_exit_order",
        lambda *_a, **_k: exit_calls.__setitem__("count", exit_calls["count"] + 1),
    )

    exited = bot_engine._exit_positions_if_needed(
        ctx=ctx,
        state=state,
        symbol="AAPL",
        feat_df=pd.DataFrame({"close": [100.0]}),
        final_score=1.0,
        conf=0.9,
        current_qty=-10,
    )

    assert exited is False
    assert exit_calls["count"] == 0
