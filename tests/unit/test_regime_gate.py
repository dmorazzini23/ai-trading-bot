from __future__ import annotations

import types
from typing import Any, cast

import ai_trading.core.bot_engine as eng


def test_check_market_regime_blocks_when_regime_is_configured(monkeypatch) -> None:
    state = eng.BotState()
    runtime = types.SimpleNamespace()

    monkeypatch.setattr(eng, "detect_regime_state", lambda _runtime: "high_volatility")
    monkeypatch.setattr(
        eng,
        "get_env",
        lambda name, default=None, cast=None, **_kwargs: (
            "high_volatility,risk_off"
            if name == "AI_TRADING_BLOCKED_REGIMES"
            else default
        ),
    )

    assert eng.check_market_regime(cast(Any, runtime), state) is False
    assert state.current_regime == "high_volatility"

