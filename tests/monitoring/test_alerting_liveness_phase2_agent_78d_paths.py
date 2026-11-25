from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from ai_trading.monitoring import model_liveness




def _set_liveness_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_REQUIRE_MARKET_OPEN", "1")
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_ALERT_COOLDOWN_SECONDS", "0")
    monkeypatch.setenv("AI_TRADING_ML_SIGNAL_MAX_AGE_SECONDS", "1")
    monkeypatch.setenv("AI_TRADING_RL_SIGNAL_MAX_AGE_SECONDS", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "0")
    monkeypatch.setenv("USE_RL_AGENT", "1")
    monkeypatch.setenv("AI_TRADING_CANARY_AUTO_ROLLBACK_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_ON_MODEL_LIVENESS_BREACH", "1")
    monkeypatch.setenv("AI_TRADING_CANARY_SYMBOLS", "SPY")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_SET_KILL_SWITCH", "0")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_COMMAND", "")
    model_liveness._reset_model_liveness_state_for_tests()










def test_liveness_env_parse_metric_defaults_and_disabled_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object | None]] = []

    def fake_get_env(name: str, default: object = None, *, cast: object | None = None) -> object:
        calls.append((name, cast))
        if cast is not None:
            raise ValueError("cast failed")
        if name == "BOOL":
            return "yes"
        if name == "FLOAT":
            return "7.5"
        raise RuntimeError("raw missing")

    monkeypatch.setattr(model_liveness, "get_env", fake_get_env)
    assert model_liveness._env_bool("BOOL", False) is True
    assert model_liveness._env_float("FLOAT", 1.0) == 7.5
    assert model_liveness._env_float("MISSING", 2.0) == 2.0
    assert calls

    assert model_liveness._parse_rollback_command("") == []
    assert model_liveness._parse_rollback_command("echo rollback") == ["echo", "rollback"]
    with pytest.raises(ValueError, match="list is empty"):
        model_liveness._parse_rollback_command("[]")
    with pytest.raises(ValueError, match="command is empty"):
        model_liveness._parse_rollback_command('""')
    assert model_liveness._event_for_metric("custom_metric") == "custom_metric"

    # The ML leg is gone; nothing in the module decides whether ML is expected.
    assert not hasattr(model_liveness, "_ml_liveness_expected_default")
    assert model_liveness._CRITICAL_CANARY_METRICS == {"rl_signal"}

    from ai_trading.config.management import get_env as config_get_env

    monkeypatch.setattr(model_liveness, "get_env", config_get_env)
    _set_liveness_env(monkeypatch)
    monkeypatch.setenv("AI_TRADING_MODEL_LIVENESS_ENABLED", "0")
    assert model_liveness.check_model_liveness(market_open=True) == []


def test_liveness_naive_timestamps_stale_reason_snapshot_and_rollbacks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _set_liveness_env(monkeypatch)
    now = datetime(2026, 4, 27, 14, 0)
    model_liveness.note_rl_signals_emitted(now=now - timedelta(seconds=5))
    breaches = model_liveness.check_model_liveness(
        market_open=True,
        rl_expected=True,
        now=now,
    )
    assert breaches[0]["reason"] == "stale"
    assert breaches[0]["last_rl_signal_ts"].endswith("+00:00")

    snapshot = model_liveness._MONITOR.snapshot(now=now + timedelta(seconds=1))
    assert snapshot["rl_age_s"] == 6.0

    payload = [
        {
            "metric": "rl_signal",
            "event": "RL_SIGNALS_EMITTED",
            "age_seconds": 5.0,
            "threshold_seconds": 1.0,
            "severity": "warning",
            "reason": "stale",
        }
    ]
    monkeypatch.setenv("AI_TRADING_CANARY_AUTO_ROLLBACK_ENABLED", "0")
    assert model_liveness.maybe_trigger_canary_auto_rollback(payload, now=now) is None
    monkeypatch.setenv("AI_TRADING_CANARY_AUTO_ROLLBACK_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_ON_MODEL_LIVENESS_BREACH", "0")
    assert model_liveness.maybe_trigger_canary_auto_rollback(payload, now=now) is None
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_ON_MODEL_LIVENESS_BREACH", "1")
    monkeypatch.setenv("AI_TRADING_CANARY_SYMBOLS", "")
    assert model_liveness.maybe_trigger_canary_auto_rollback(payload, now=now) is None
    monkeypatch.setenv("AI_TRADING_CANARY_SYMBOLS", "SPY")
    assert (
        model_liveness.maybe_trigger_canary_auto_rollback(
            [{**payload[0], "metric": "after_hours_training"}],
            now=now,
        )
        is None
    )

    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_FLAG_PATH", str(tmp_path / "flag"))
    monkeypatch.setenv("AI_TRADING_KILL_SWITCH_PATH", str(tmp_path / "kill"))
    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_SET_KILL_SWITCH", "1")

    def bad_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("write failed")

    monkeypatch.setattr(model_liveness, "_atomic_write_text", bad_write)
    result = model_liveness.maybe_trigger_canary_auto_rollback(payload, now=now)
    assert result is not None
    assert result["triggered"] is True
    assert "flag_path" not in result
    assert "kill_switch_path" not in result

    monkeypatch.setenv("AI_TRADING_CANARY_ROLLBACK_COMMAND", "[]")
    model_liveness._reset_model_liveness_state_for_tests()
    monkeypatch.setattr(model_liveness, "_atomic_write_text", lambda *_args, **_kwargs: None)
    result = model_liveness.maybe_trigger_canary_auto_rollback(
        [payload[0], {"metric": object()}],
        now=now,
    )
    assert result is not None
    assert result["command_error"]

    class BadEntry:
        def get(self, _key: str) -> object:
            raise ValueError("bad payload")

    assert model_liveness.maybe_trigger_canary_auto_rollback([BadEntry()], now=now) is None


def test_liveness_atomic_write_cleans_failed_tempfile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    removed: list[str] = []

    def bad_replace(_src: str, _dst: model_liveness.Path) -> None:
        raise OSError("replace failed")

    def bad_unlink(path: str) -> None:
        removed.append(path)
        raise OSError("unlink failed")

    monkeypatch.setattr(model_liveness.os, "replace", bad_replace)
    monkeypatch.setattr(model_liveness.os, "unlink", bad_unlink)

    with pytest.raises(OSError, match="replace failed"):
        model_liveness._atomic_write_text(tmp_path / "rollback.flag", "rollback\n")

    assert removed




