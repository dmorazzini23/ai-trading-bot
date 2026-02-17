from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ai_trading.runtime.run_manifest import write_run_manifest


def test_run_manifest_written(tmp_path: Path) -> None:
    cfg = SimpleNamespace(
        execution_mode="paper",
        alpaca_api_key="PK1234567890",
        run_manifest_path=str(tmp_path / "run_manifest.json"),
        to_dict=lambda: {
            "execution_mode": "paper",
            "recon_enabled": True,
            "kill_switch": False,
        },
    )
    path = write_run_manifest(cfg, runtime_contract={"stubs_enabled": False})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["mode"] == "paper"
    assert payload["runtime_contract"]["stubs_enabled"] is False
    assert payload["resolved_config_hash"]


def test_run_manifest_uses_env_path_when_cfg_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "run_manifest_env.json"
    monkeypatch.setenv("AI_TRADING_RUN_MANIFEST_PATH", str(target))

    cfg = SimpleNamespace(
        execution_mode="paper",
        alpaca_api_key="PK1234567890",
        to_dict=lambda: {
            "execution_mode": "paper",
            "recon_enabled": True,
            "kill_switch": False,
        },
    )

    path = write_run_manifest(cfg, runtime_contract={"stubs_enabled": False})
    assert path == target
    assert path.exists()


def test_run_manifest_relative_path_prefers_state_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.delenv("AI_TRADING_DATA_DIR", raising=False)
    monkeypatch.setenv("STATE_DIRECTORY", str(state_dir))
    monkeypatch.setenv("AI_TRADING_RUN_MANIFEST_PATH", "runtime/run_manifest.jsonl")

    cfg = SimpleNamespace(
        execution_mode="paper",
        alpaca_api_key="PK1234567890",
        to_dict=lambda: {
            "execution_mode": "paper",
            "recon_enabled": True,
            "kill_switch": False,
        },
    )

    path = write_run_manifest(cfg, runtime_contract={"stubs_enabled": False})
    assert path == (state_dir / "runtime" / "run_manifest.jsonl").resolve()
    assert path.exists()
