from __future__ import annotations

import types

from ai_trading.core import execution_flow
from ai_trading.settings import (
    get_buy_threshold,
    get_conf_threshold,
    get_settings,
)


def _clear_live_default_env(monkeypatch) -> None:
    for key in (
        "AI_TRADING_CONF_THRESHOLD",
        "AI_TRADING_BUY_THRESHOLD",
    ):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()


def test_live_defaults_align_with_replay_profile(monkeypatch) -> None:
    _clear_live_default_env(monkeypatch)

    assert get_conf_threshold() == 0.52
    assert get_buy_threshold() == 0.15


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

