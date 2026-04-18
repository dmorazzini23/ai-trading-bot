from __future__ import annotations

import types
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from ai_trading.core import bot_engine, execution_flow
from ai_trading.settings import (
    get_buy_threshold,
    get_conf_threshold,
    get_live_stop_loss_bps,
    get_live_take_profit_bps,
    get_live_trailing_stop_bps,
    get_max_position_hold_seconds,
    get_min_position_hold_seconds,
    get_settings,
)


def _clear_live_default_env(monkeypatch) -> None:
    for key in (
        "AI_TRADING_CONF_THRESHOLD",
        "AI_TRADING_BUY_THRESHOLD",
        "AI_TRADING_MIN_POSITION_HOLD_SECONDS",
        "AI_TRADING_MAX_POSITION_HOLD_SECONDS",
        "AI_TRADING_LIVE_STOP_LOSS_BPS",
        "AI_TRADING_LIVE_TAKE_PROFIT_BPS",
        "AI_TRADING_LIVE_TRAILING_STOP_BPS",
    ):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()


def test_live_defaults_align_with_replay_profile(monkeypatch) -> None:
    _clear_live_default_env(monkeypatch)

    assert get_conf_threshold() == 0.52
    assert get_buy_threshold() == 0.15
    assert get_min_position_hold_seconds() == 600
    assert get_max_position_hold_seconds() == 7200
    assert get_live_stop_loss_bps() == 60.0
    assert get_live_take_profit_bps() == 160.0
    assert get_live_trailing_stop_bps() == 90.0


def test_apply_bps_exit_overlays_caps_targets() -> None:
    long_stop, long_take = bot_engine._apply_bps_exit_overlays(  # noqa: SLF001
        entry_price=100.0,
        side=1,
        stop=90.0,
        take=120.0,
    )
    assert long_stop == 99.4
    assert long_take == 101.6

    short_stop, short_take = bot_engine._apply_bps_exit_overlays(  # noqa: SLF001
        entry_price=100.0,
        side=-1,
        stop=110.0,
        take=80.0,
    )
    assert short_stop == 100.6
    assert short_take == 98.4


def test_should_exit_enforces_max_hold(monkeypatch) -> None:
    monkeypatch.setattr(bot_engine, "_current_qty", lambda *_a, **_k: 5)
    monkeypatch.setattr(bot_engine, "get_max_position_hold_seconds", lambda: 7200)
    monkeypatch.setattr(bot_engine, "_bps_trailing_stop_action", lambda *_a, **_k: "hold")
    monkeypatch.setattr(bot_engine, "update_trailing_stop", lambda *_a, **_k: "hold")

    ctx = types.SimpleNamespace(
        rebalance_buys={},
        stop_targets={},
        take_profit_targets={},
        trailing_extremes={},
        position_map={"AAPL": types.SimpleNamespace(avg_entry_price=100.0)},
    )
    state = bot_engine.BotState()
    state.position_entry_times["AAPL"] = datetime.now(UTC) - timedelta(seconds=7201)

    should_exit, qty, reason = bot_engine.should_exit(
        cast(Any, ctx),
        state,
        "AAPL",
        101.0,
        1.0,
    )

    assert should_exit is True
    assert qty == 5
    assert reason == "max_hold"


def test_bps_trailing_stop_requires_profit(monkeypatch) -> None:
    monkeypatch.setattr(bot_engine, "get_live_trailing_stop_bps", lambda: 90.0)

    ctx = types.SimpleNamespace(
        trailing_extremes={"AAPL": 102.0},
        position_map={"AAPL": types.SimpleNamespace(avg_entry_price=100.0)},
    )

    assert bot_engine._bps_trailing_stop_action(cast(Any, ctx), "AAPL", 101.0, 5) == "exit_long"  # noqa: SLF001
    assert bot_engine._bps_trailing_stop_action(cast(Any, ctx), "AAPL", 99.5, 5) == "hold"  # noqa: SLF001


def test_liquidate_positions_if_needed_flattens_eod(monkeypatch) -> None:
    monkeypatch.setattr(
        execution_flow,
        "_should_trigger_eod_flatten",
        lambda now_et=None: (True, {"reason": "session_close_window"}),
    )

    import ai_trading.core.bot_engine as live_bot_engine

    monkeypatch.setattr(live_bot_engine, "check_halt_flag", lambda _runtime: False)

    calls: list[str] = []

    monkeypatch.setattr(
        execution_flow,
        "exit_all_positions",
        lambda runtime: calls.append("flattened"),
    )

    runtime = types.SimpleNamespace(
        api=types.SimpleNamespace(
            list_positions=lambda: [types.SimpleNamespace(symbol="AAPL", qty="5")]
        )
    )

    execution_flow.liquidate_positions_if_needed(runtime)

    assert calls == ["flattened"]
