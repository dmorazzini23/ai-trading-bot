from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from ai_trading.core.trade_cycle import _resolve_regime_entry_threshold_adjustment


def test_regime_entry_threshold_adjustment_default_off_preserves_threshold() -> None:
    threshold, context = _resolve_regime_entry_threshold_adjustment(
        get_env_func=lambda _key, default=None, cast=None: default,
        state=SimpleNamespace(current_regime="volatile"),
        symbol="AAPL",
        base_threshold=0.55,
        now=datetime(2026, 5, 4, 14, 0, tzinfo=UTC),
    )

    assert threshold == 0.55
    assert context == {}


def test_regime_entry_threshold_adjustment_clamps_up_only() -> None:
    env = {
        "AI_TRADING_REGIME_ENTRY_THRESHOLD_ENABLED": True,
        "AI_TRADING_REGIME_ENTRY_THRESHOLDS": (
            '{"volatile:opening": 0.72, "opening": 0.68, "default": 0.50}'
        ),
    }

    def _get_env(key: str, default=None, cast=None):
        return env.get(key, default)

    threshold, context = _resolve_regime_entry_threshold_adjustment(
        get_env_func=_get_env,
        state=SimpleNamespace(current_regime="volatile"),
        symbol="MSFT",
        base_threshold=0.60,
        now=datetime(2026, 5, 4, 13, 45, tzinfo=UTC),
    )

    assert threshold == 0.72
    assert context["source"] == "regime_session"
    assert context["regime"] == "volatile"
    assert context["session_regime"] == "opening"

    threshold, context = _resolve_regime_entry_threshold_adjustment(
        get_env_func=_get_env,
        state=SimpleNamespace(current_regime="balanced"),
        symbol="MSFT",
        base_threshold=0.60,
        now=datetime(2026, 5, 4, 16, 0, tzinfo=UTC),
    )

    assert threshold == 0.60
    assert context["configured_threshold"] == 0.50
