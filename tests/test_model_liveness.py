from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trading.monitoring import model_liveness


def _set_default_liveness_env(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_REQUIRE_MARKET_OPEN", "1")
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_ALERT_COOLDOWN_SECONDS", "0")
    monkeypatch.setenv("AI_TRADING_ML_SIGNAL_MAX_AGE_SECONDS", "1")
    monkeypatch.setenv("AI_TRADING_RL_SIGNAL_MAX_AGE_SECONDS", "1")
    monkeypatch.setenv("USE_RL_AGENT", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_CANARY_AUTO_ROLLBACK_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_ON_MODEL_LIVENESS_BREACH", "1")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_COOLDOWN_SECONDS", "300")
    model_liveness._reset_model_liveness_state_for_tests()


def test_model_liveness_breaches_when_rl_signals_are_stale(monkeypatch) -> None:
    _set_default_liveness_env(monkeypatch)

    now = datetime.now(UTC) + timedelta(seconds=2)
    breaches = model_liveness.check_model_liveness(
        market_open=True,
        signals_expected_now=True,
        rl_expected=True,
        now=now,
    )
    metrics = {entry["metric"] for entry in breaches}

    # No ML model serves runtime decisions: there is no ML heartbeat to go stale.
    assert metrics == {"rl_signal"}
    assert all(entry["ml_expected"] is False for entry in breaches)


def test_model_liveness_skips_intraday_checks_when_market_closed(monkeypatch) -> None:
    _set_default_liveness_env(monkeypatch)

    now = datetime.now(UTC) + timedelta(seconds=2)
    breaches = model_liveness.check_model_liveness(market_open=False, now=now)

    assert breaches == []


def test_ml_leg_never_breaches_even_when_never_observed(monkeypatch) -> None:
    _set_default_liveness_env(monkeypatch)
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_ENFORCE_ML", "1")
    monkeypatch.setenv("USE_RL_AGENT", "0")
    assert not hasattr(model_liveness, "note_ml_signal")

    breaches = model_liveness.check_model_liveness(
        market_open=True,
        signals_expected_now=True,
        execution_gate_open=True,
        warmup_complete=True,
        now=datetime.now(UTC) + timedelta(days=1),
    )

    assert breaches == []


def test_recorded_rl_signal_suppresses_rl_breach(monkeypatch) -> None:
    _set_default_liveness_env(monkeypatch)
    monkeypatch.setenv("AI_TRADING_RL_SIGNAL_MAX_AGE_SECONDS", "10")
    base = datetime.now(UTC)
    model_liveness.note_rl_signals_emitted(now=base)

    breaches = model_liveness.check_model_liveness(
        market_open=True,
        signals_expected_now=True,
        rl_expected=True,
        now=base + timedelta(seconds=2),
    )

    assert breaches == []


def test_after_hours_training_liveness_is_monitored_when_enabled(monkeypatch) -> None:
    _set_default_liveness_env(monkeypatch)
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_MAX_AGE_SECONDS", "60")
    model_liveness._reset_model_liveness_state_for_tests()

    now = datetime.now(UTC) + timedelta(seconds=120)
    breaches = model_liveness.check_model_liveness(market_open=True, now=now)
    metrics = {entry["metric"] for entry in breaches}

    assert "after_hours_training" in metrics


def test_model_liveness_snapshot_says_no_ml_model_is_expected(monkeypatch) -> None:
    _set_default_liveness_env(monkeypatch)
    snapshot = model_liveness.get_model_liveness_snapshot()

    assert snapshot["ml_expected"] is False
    assert snapshot["ml_status"] == "no_model_expected"
    for key in ("last_ml_signal_ts", "ml_age_s", "ml_since_start_s", "ml_max_age_s"):
        assert key not in snapshot
    assert snapshot["last_rl_signal_ts"] is None
    assert snapshot["rl_age_s"] is None
    assert float(snapshot["rl_since_start_s"]) >= 0.0


def test_ml_breach_cannot_trigger_canary_rollback_or_kill_switch(monkeypatch, tmp_path) -> None:
    _set_default_liveness_env(monkeypatch)
    rollback_flag = tmp_path / "canary_rollback.flag"
    kill_switch = tmp_path / "kill_switch.flag"
    monkeypatch.setenv("AI_TRADING_CANARY_SYMBOLS", "AAPL")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_FLAG_PATH", str(rollback_flag))
    monkeypatch.setenv("AI_TRADING_KILL_SWITCH_PATH", str(kill_switch))
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_SET_KILL_SWITCH", "1")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_COMMAND", "")
    model_liveness._reset_model_liveness_state_for_tests()

    result = model_liveness.maybe_trigger_canary_auto_rollback(
        [
            {
                "metric": "ml_signal",
                "event": "ML_SIGNAL",
                "age_seconds": 120.0,
                "threshold_seconds": 60.0,
                "severity": "critical",
                "reason": "stale",
            }
        ],
        now=datetime.now(UTC),
    )

    assert result is None
    assert not rollback_flag.exists()
    assert not kill_switch.exists()


def test_canary_auto_rollback_writes_flag_and_respects_cooldown(monkeypatch, tmp_path) -> None:
    """The RL-signal leg still writes the rollback flag and the kill switch."""
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
            "metric": "rl_signal",
            "event": "RL_SIGNALS_EMITTED",
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


def test_canary_auto_rollback_resolves_default_files_under_runtime_root(
    monkeypatch,
    tmp_path,
) -> None:
    _set_default_liveness_env(monkeypatch)
    data_dir = tmp_path / "data-root"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_dir))
    monkeypatch.delenv("AI_TRADING_CANARY_ROLLBACK_FLAG_PATH", raising=False)
    monkeypatch.delenv("AI_TRADING_KILL_SWITCH_PATH", raising=False)
    monkeypatch.setenv("AI_TRADING_CANARY_SYMBOLS", "AAPL")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_SET_KILL_SWITCH", "1")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_COMMAND", "")
    model_liveness._reset_model_liveness_state_for_tests()

    result = model_liveness.maybe_trigger_canary_auto_rollback(
        [
            {
                "metric": "rl_signal",
                "event": "RL_SIGNALS_EMITTED",
                "age_seconds": 120.0,
                "threshold_seconds": 60.0,
                "severity": "critical",
                "reason": "stale",
            }
        ],
        now=datetime.now(UTC),
    )

    expected_flag = (data_dir / "runtime/canary_rollback.flag").resolve()
    expected_kill_switch = (data_dir / "runtime/kill_switch").resolve()
    assert result is not None
    assert result["flag_path"] == str(expected_flag)
    assert result["kill_switch_path"] == str(expected_kill_switch)
    assert expected_flag.exists()
    assert expected_kill_switch.exists()
    assert not (cwd / "runtime/canary_rollback.flag").exists()
    assert not (cwd / "runtime/kill_switch").exists()


def test_canary_auto_rollback_runs_command_without_shell(monkeypatch, tmp_path) -> None:
    _set_default_liveness_env(monkeypatch)
    rollback_flag = tmp_path / "canary_rollback.flag"
    kill_switch = tmp_path / "kill_switch.flag"
    monkeypatch.setenv("AI_TRADING_CANARY_SYMBOLS", "AAPL")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_FLAG_PATH", str(rollback_flag))
    monkeypatch.setenv("AI_TRADING_KILL_SWITCH_PATH", str(kill_switch))
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_SET_KILL_SWITCH", "1")
    monkeypatch.setenv(
        "AI_TRADING_CANARY_ROLLBACK_COMMAND",
        json.dumps(["/bin/echo", "rollback-now"]),
    )
    captured: dict[str, object] = {}

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = dict(kwargs)
        return _Completed()

    monkeypatch.setattr(model_liveness.subprocess, "run", _fake_run)
    model_liveness._reset_model_liveness_state_for_tests()

    breach_payload = [
        {
            "metric": "rl_signal",
            "event": "RL_SIGNALS_EMITTED",
            "age_seconds": 120.0,
            "threshold_seconds": 60.0,
            "severity": "critical",
            "reason": "stale",
        }
    ]
    result = model_liveness.maybe_trigger_canary_auto_rollback(
        breach_payload,
        now=datetime.now(UTC),
    )

    assert result is not None
    assert result["triggered"] is True
    assert Path(rollback_flag).exists()
    assert Path(kill_switch).exists()
    assert captured["command"] == ["/bin/echo", "rollback-now"]
    kwargs_payload = captured["kwargs"]
    assert isinstance(kwargs_payload, Mapping)
    kwargs = dict(kwargs_payload)
    assert "shell" not in kwargs
    assert kwargs["check"] is False
