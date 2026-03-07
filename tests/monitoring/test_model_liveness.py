from __future__ import annotations

from datetime import UTC, datetime, timedelta
import sys
import types

from ai_trading.monitoring import model_liveness


def _set_liveness_defaults(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_REQUIRE_MARKET_OPEN", "1")
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_ALERT_COOLDOWN_SECONDS", "0")
    monkeypatch.setenv("AI_TRADING_ML_SIGNAL_MAX_AGE_SECONDS", "1")
    monkeypatch.setenv("AI_TRADING_RL_SIGNAL_MAX_AGE_SECONDS", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "0")
    monkeypatch.setenv("USE_RL_AGENT", "0")
    model_liveness._reset_model_liveness_state_for_tests()


def _install_bot_engine_module(monkeypatch, *, use_ml: bool, rl_agent: object | None = None) -> None:
    module = types.ModuleType("ai_trading.core.bot_engine")
    module.USE_ML = use_ml  # type: ignore[attr-defined]
    module.RL_AGENT = rl_agent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_trading.core.bot_engine", module)


def test_liveness_skips_when_signals_not_expected(monkeypatch) -> None:
    _set_liveness_defaults(monkeypatch)
    _install_bot_engine_module(monkeypatch, use_ml=True)

    breaches = model_liveness.check_model_liveness(
        market_open=True,
        signals_expected_now=False,
        now=datetime.now(UTC) + timedelta(seconds=5),
    )

    assert breaches == []


def test_liveness_does_not_enforce_ml_when_model_disabled(monkeypatch) -> None:
    _set_liveness_defaults(monkeypatch)
    _install_bot_engine_module(monkeypatch, use_ml=False)

    breaches = model_liveness.check_model_liveness(
        market_open=True,
        signals_expected_now=True,
        now=datetime.now(UTC) + timedelta(seconds=5),
    )

    assert breaches == []


def test_liveness_payload_includes_phase_and_age_fields(monkeypatch) -> None:
    _set_liveness_defaults(monkeypatch)
    _install_bot_engine_module(monkeypatch, use_ml=True)

    breaches = model_liveness.check_model_liveness(
        market_open=True,
        signals_expected_now=True,
        phase="active",
        execution_gate_open=True,
        warmup_complete=True,
        now=datetime.now(UTC) + timedelta(seconds=5),
    )

    assert breaches
    breach = breaches[0]
    assert breach["metric"] == "ml_signal"
    assert breach["severity"] == "warning"
    assert "last_ml_signal_ts" in breach
    assert "ml_age_s" in breach
    assert "ml_since_start_s" in breach
    assert "ml_max_age_s" in breach
    assert breach["ml_age_s"] is None
    assert float(breach["ml_since_start_s"]) >= 0.0
    assert breach["market_open"] is True
    assert breach["signals_expected_now"] is True
    assert breach["phase"] == "active"
    assert breach["execution_gate_open"] is True
    assert breach["warmup_complete"] is True


def test_liveness_alert_cooldown_suppresses_repeat(monkeypatch) -> None:
    _set_liveness_defaults(monkeypatch)
    _install_bot_engine_module(monkeypatch, use_ml=True)
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_ALERT_COOLDOWN_SECONDS", "60")
    model_liveness._reset_model_liveness_state_for_tests()

    first_now = datetime.now(UTC) + timedelta(seconds=5)
    first = model_liveness.check_model_liveness(
        market_open=True,
        signals_expected_now=True,
        now=first_now,
    )
    second = model_liveness.check_model_liveness(
        market_open=True,
        signals_expected_now=True,
        now=first_now + timedelta(seconds=10),
    )

    assert first
    assert second == []
