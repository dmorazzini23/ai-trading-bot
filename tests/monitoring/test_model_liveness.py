from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
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


def _install_bot_engine_module(monkeypatch, *, rl_agent: object | None = None) -> None:
    module = types.ModuleType("ai_trading.core.bot_engine")
    module.RL_AGENT = rl_agent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_trading.core.bot_engine", module)


def test_liveness_skips_when_signals_not_expected(monkeypatch) -> None:
    _set_liveness_defaults(monkeypatch)
    _install_bot_engine_module(monkeypatch)

    breaches = model_liveness.check_model_liveness(
        market_open=True,
        signals_expected_now=False,
        now=datetime.now(UTC) + timedelta(seconds=5),
    )

    assert breaches == []


def test_after_hours_training_liveness_ignores_signal_market_gates(monkeypatch) -> None:
    _set_liveness_defaults(monkeypatch)
    _install_bot_engine_module(monkeypatch)
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_MAX_AGE_SECONDS", "60")
    model_liveness._reset_model_liveness_state_for_tests()

    breaches = model_liveness.check_model_liveness(
        market_open=False,
        signals_expected_now=False,
        now=datetime.now(UTC) + timedelta(seconds=120),
    )

    assert [breach["metric"] for breach in breaches] == ["after_hours_training"]
    assert breaches[0]["event"] == "AFTER_HOURS_TRAINING_COMPLETE"
    assert breaches[0]["market_open"] is False
    assert breaches[0]["signals_expected_now"] is False


def test_after_hours_training_liveness_seeds_from_durable_artifact(
    monkeypatch,
    tmp_path,
) -> None:
    _set_liveness_defaults(monkeypatch)
    _install_bot_engine_module(monkeypatch)
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_MAX_AGE_SECONDS", "60")
    artifact = tmp_path / "after_hours_training_latest.json"
    now = datetime.now(UTC)
    artifact.write_text(
        json.dumps(
            {
                "report": {
                    "status": "skipped",
                    "reason": "no_qualified_candidate",
                    "timestamp": (now - timedelta(seconds=10)).isoformat(),
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_REPORT_LATEST_PATH", str(artifact))
    model_liveness._reset_model_liveness_state_for_tests()

    breaches = model_liveness.check_model_liveness(
        market_open=False,
        signals_expected_now=False,
        now=now,
    )

    assert breaches == []


def test_liveness_never_enforces_ml_even_with_stale_use_ml_marker(monkeypatch) -> None:
    _set_liveness_defaults(monkeypatch)
    _install_bot_engine_module(monkeypatch)
    # An old-style marker on the module must not bring the ML leg back.
    monkeypatch.setattr(sys.modules["ai_trading.core.bot_engine"], "USE_ML", True, raising=False)
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_ENFORCE_ML", "1")

    breaches = model_liveness.check_model_liveness(
        market_open=True,
        signals_expected_now=True,
        now=datetime.now(UTC) + timedelta(seconds=5),
    )

    assert breaches == []


def test_liveness_payload_includes_phase_and_age_fields(monkeypatch) -> None:
    _set_liveness_defaults(monkeypatch)
    _install_bot_engine_module(monkeypatch)
    monkeypatch.setenv("USE_RL_AGENT", "1")

    breaches = model_liveness.check_model_liveness(
        market_open=True,
        signals_expected_now=True,
        phase="active",
        execution_gate_open=True,
        warmup_complete=True,
        rl_expected=True,
        now=datetime.now(UTC) + timedelta(seconds=5),
    )

    assert [breach["metric"] for breach in breaches] == ["rl_signal"]
    breach = breaches[0]
    assert breach["severity"] == "warning"
    assert breach["ml_expected"] is False
    assert breach["ml_status"] == "no_model_expected"
    assert "last_rl_signal_ts" in breach
    assert breach["rl_age_s"] is None
    assert float(breach["rl_since_start_s"]) >= 0.0
    assert breach["rl_max_age_s"] == 1.0
    assert breach["market_open"] is True
    assert breach["signals_expected_now"] is True
    assert breach["phase"] == "active"
    assert breach["execution_gate_open"] is True
    assert breach["warmup_complete"] is True


def test_liveness_alert_cooldown_suppresses_repeat(monkeypatch) -> None:
    _set_liveness_defaults(monkeypatch)
    _install_bot_engine_module(monkeypatch)
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_ALERT_COOLDOWN_SECONDS", "60")
    monkeypatch.setenv("USE_RL_AGENT", "1")
    model_liveness._reset_model_liveness_state_for_tests()

    first_now = datetime.now(UTC) + timedelta(seconds=5)
    first = model_liveness.check_model_liveness(
        market_open=True,
        signals_expected_now=True,
        rl_expected=True,
        now=first_now,
    )
    second = model_liveness.check_model_liveness(
        market_open=True,
        signals_expected_now=True,
        rl_expected=True,
        now=first_now + timedelta(seconds=10),
    )

    assert first
    assert second == []
