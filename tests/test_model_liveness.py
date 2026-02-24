from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trading.monitoring import model_liveness


def _set_default_liveness_env(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_REQUIRE_MARKET_OPEN", "1")
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_ALERT_COOLDOWN_SECONDS", "0")
    monkeypatch.setenv("AI_TRADING_ML_SIGNAL_MAX_AGE_SECONDS", "1")
    monkeypatch.setenv("AI_TRADING_RL_SIGNAL_MAX_AGE_SECONDS", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_CANARY_AUTO_ROLLBACK_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_ON_MODEL_LIVENESS_BREACH", "1")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_COOLDOWN_SECONDS", "300")
    model_liveness._reset_model_liveness_state_for_tests()


def test_model_liveness_breaches_when_signals_are_stale(monkeypatch) -> None:
    _set_default_liveness_env(monkeypatch)

    now = datetime.now(UTC) + timedelta(seconds=2)
    breaches = model_liveness.check_model_liveness(market_open=True, now=now)
    metrics = {entry["metric"] for entry in breaches}

    assert "ml_signal" in metrics
    assert "rl_signal" in metrics


def test_model_liveness_skips_intraday_checks_when_market_closed(monkeypatch) -> None:
    _set_default_liveness_env(monkeypatch)

    now = datetime.now(UTC) + timedelta(seconds=2)
    breaches = model_liveness.check_model_liveness(market_open=False, now=now)

    assert breaches == []


def test_recorded_ml_signal_suppresses_ml_breach(monkeypatch) -> None:
    _set_default_liveness_env(monkeypatch)
    monkeypatch.setenv("AI_TRADING_ML_SIGNAL_MAX_AGE_SECONDS", "10")
    base = datetime.now(UTC)
    model_liveness.note_ml_signal(now=base)

    breaches = model_liveness.check_model_liveness(
        market_open=True,
        now=base + timedelta(seconds=2),
    )
    metrics = {entry["metric"] for entry in breaches}

    assert "ml_signal" not in metrics
    assert "rl_signal" in metrics


def test_after_hours_training_liveness_is_monitored_when_enabled(monkeypatch) -> None:
    _set_default_liveness_env(monkeypatch)
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_MAX_AGE_SECONDS", "60")
    model_liveness._reset_model_liveness_state_for_tests()

    now = datetime.now(UTC) + timedelta(seconds=120)
    breaches = model_liveness.check_model_liveness(market_open=True, now=now)
    metrics = {entry["metric"] for entry in breaches}

    assert "after_hours_training" in metrics


def test_canary_auto_rollback_writes_flag_and_respects_cooldown(monkeypatch, tmp_path) -> None:
    _set_default_liveness_env(monkeypatch)
    rollback_flag = tmp_path / "canary_rollback.flag"
    kill_switch = tmp_path / "kill_switch.flag"
    monkeypatch.setenv("AI_TRADING_CANARY_SYMBOLS", "AAPL")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_FLAG_PATH", str(rollback_flag))
    monkeypatch.setenv("AI_TRADING_KILL_SWITCH_PATH", str(kill_switch))
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_SET_KILL_SWITCH", "1")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_COMMAND", "")
    model_liveness._reset_model_liveness_state_for_tests()

    breach_payload = [
        {
            "metric": "ml_signal",
            "event": "ML_SIGNAL",
            "age_seconds": 120.0,
            "threshold_seconds": 60.0,
            "severity": "critical",
            "reason": "stale",
        }
    ]
    first_now = datetime.now(UTC)
    first = model_liveness.maybe_trigger_canary_auto_rollback(
        breach_payload,
        now=first_now,
    )
    assert first is not None
    assert first["triggered"] is True
    assert Path(rollback_flag).exists()
    assert Path(kill_switch).exists()

    second = model_liveness.maybe_trigger_canary_auto_rollback(
        breach_payload,
        now=first_now + timedelta(seconds=10),
    )
    assert second is not None
    assert second["triggered"] is False
    assert second["status"] == "cooldown"
