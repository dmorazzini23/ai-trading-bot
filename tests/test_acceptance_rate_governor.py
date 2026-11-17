from __future__ import annotations

from ai_trading.core import bot_engine


def test_acceptance_rate_governor_activates(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_ACCEPTANCE_RATE_GOVERNOR_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ACCEPTANCE_RATE_GOVERNOR_MIN_RATE", "0.30")
    monkeypatch.setenv("AI_TRADING_ACCEPTANCE_RATE_GOVERNOR_MIN_RECORDS", "10")
    monkeypatch.setenv("AI_TRADING_ACCEPTANCE_RATE_GOVERNOR_TRIGGER_STREAK", "2")
    monkeypatch.setenv("AI_TRADING_ACCEPTANCE_RATE_GOVERNOR_RECOVER_STREAK", "2")

    state = bot_engine.BotState()
    bot_engine._update_acceptance_rate_governor_state(
        state,
        decision_records_total=20,
        accepted_decisions=2,
    )
    assert state.acceptance_rate_governor_active is False
    bot_engine._update_acceptance_rate_governor_state(
        state,
        decision_records_total=20,
        accepted_decisions=1,
    )
    assert state.acceptance_rate_governor_active is True
    assert state.acceptance_rate_governor_streak >= 2


def test_acceptance_rate_governor_recovers_after_strong_acceptance(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_ACCEPTANCE_RATE_GOVERNOR_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ACCEPTANCE_RATE_GOVERNOR_MIN_RATE", "0.30")
    monkeypatch.setenv("AI_TRADING_ACCEPTANCE_RATE_GOVERNOR_MIN_RECORDS", "10")
    monkeypatch.setenv("AI_TRADING_ACCEPTANCE_RATE_GOVERNOR_TRIGGER_STREAK", "1")
    monkeypatch.setenv("AI_TRADING_ACCEPTANCE_RATE_GOVERNOR_RECOVER_STREAK", "2")

    state = bot_engine.BotState()
    bot_engine._update_acceptance_rate_governor_state(
        state,
        decision_records_total=20,
        accepted_decisions=1,
    )
    assert state.acceptance_rate_governor_active is True

    bot_engine._update_acceptance_rate_governor_state(
        state,
        decision_records_total=20,
        accepted_decisions=16,
    )
    assert state.acceptance_rate_governor_active is True
    bot_engine._update_acceptance_rate_governor_state(
        state,
        decision_records_total=20,
        accepted_decisions=18,
    )
    assert state.acceptance_rate_governor_active is False
