from __future__ import annotations

import logging
from pathlib import Path

import ai_trading.main as main


def test_rl_runtime_path_preflight_warns_when_target_not_writable(
    monkeypatch,
    tmp_path: Path,
    caplog,
) -> None:
    target = tmp_path / "runtime" / "rl_agent.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"rl-model")

    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTE_RL_PATH", "1")
    monkeypatch.setenv("AI_TRADING_RL_MODEL_PATH", str(target))

    original_access = main.os.access

    def _fake_access(path: str | Path, mode: int) -> bool:
        if Path(path) == target:
            return False
        return original_access(path, mode)

    monkeypatch.setattr(main.os, "access", _fake_access)
    caplog.set_level(logging.WARNING, logger=main.logger.name)

    assert main._preflight_rl_runtime_model_path_permissions() is False
    records = [
        record
        for record in caplog.records
        if record.getMessage() == "RL_RUNTIME_MODEL_PATH_PERMISSION_WARNING"
    ]
    assert records, "expected RL runtime path permission warning"
    assert getattr(records[0], "path", "") == str(target)
    assert getattr(records[0], "reason", "") == "target_not_writable"


def test_rl_runtime_path_preflight_noop_when_rl_overlay_disabled(
    monkeypatch,
    tmp_path: Path,
    caplog,
) -> None:
    target = tmp_path / "runtime" / "rl_agent.zip"
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_AFTER_HOURS_PROMOTE_RL_PATH", "1")
    monkeypatch.setenv("AI_TRADING_RL_MODEL_PATH", str(target))

    caplog.set_level(logging.WARNING, logger=main.logger.name)

    assert main._preflight_rl_runtime_model_path_permissions() is True
    assert not any(
        record.getMessage() == "RL_RUNTIME_MODEL_PATH_PERMISSION_WARNING"
        for record in caplog.records
    )
