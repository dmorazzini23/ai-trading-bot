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


def test_pre_trade_checks_respects_regime_gate(monkeypatch) -> None:
    state = eng.BotState()
    state.current_regime = "risk_off"
    ctx = types.SimpleNamespace()
    diagnostics_calls: list[str] = []

    monkeypatch.setattr(
        eng,
        "_log_health_diagnostics",
        lambda _runtime, reason: diagnostics_calls.append(str(reason)),
    )

    assert (
        eng.pre_trade_checks(cast(Any, ctx), state, "AAPL", 1000.0, regime_ok=False)
        is False
    )
    assert diagnostics_calls == ["regime"]
