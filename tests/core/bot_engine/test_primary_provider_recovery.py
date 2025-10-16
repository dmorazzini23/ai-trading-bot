from __future__ import annotations

import types

from ai_trading.core import bot_engine


def test_clear_primary_provider_fallback_resets_preference_globally() -> None:
    state = types.SimpleNamespace(
        prefer_backup_quotes=True,
        primary_fallback_events={("alpaca", "AAPL"), ("alpaca", "MSFT")},
        degraded_providers={"alpaca", "AAPL", "MSFT"},
    )

    bot_engine._clear_primary_provider_fallback(state, "AAPL")

    assert ("alpaca", "AAPL") not in state.primary_fallback_events
    assert ("alpaca", "MSFT") in state.primary_fallback_events
    assert "AAPL" not in state.degraded_providers
    assert getattr(state, "prefer_backup_quotes", False)

    bot_engine._clear_primary_provider_fallback(state, "MSFT")

    assert not state.primary_fallback_events
    assert "alpaca" not in state.degraded_providers
    assert not getattr(state, "prefer_backup_quotes", False)


def test_clear_primary_provider_fallback_handles_missing_state() -> None:
    state = types.SimpleNamespace(prefer_backup_quotes=True)

    bot_engine._clear_primary_provider_fallback(state, "SPY")

    assert not getattr(state, "prefer_backup_quotes", False)
