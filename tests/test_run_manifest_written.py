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
    mirror_path = (state_dir / "runtime" / "run_manifest.json").resolve()
    assert mirror_path.exists()


def test_run_manifest_jsonl_append_and_json_mirror(tmp_path: Path) -> None:
    target = tmp_path / "run_manifest.jsonl"
    cfg = SimpleNamespace(
        execution_mode="paper",
        alpaca_api_key="PK1234567890",
        run_manifest_path=str(target),
        to_dict=lambda: {
            "execution_mode": "paper",
            "recon_enabled": True,
            "kill_switch": False,
        },
    )

    first_path = write_run_manifest(cfg, runtime_contract={"stubs_enabled": False})
    second_path = write_run_manifest(cfg, runtime_contract={"stubs_enabled": False})

    assert first_path == target
    assert second_path == target
    lines = [
        line
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2
    first_payload = json.loads(lines[0])
    second_payload = json.loads(lines[1])
    assert first_payload["runtime_contract"]["stubs_enabled"] is False
    assert second_payload["runtime_contract"]["stubs_enabled"] is False

    mirror = tmp_path / "run_manifest.json"
    assert mirror.exists()
    mirror_payload = json.loads(mirror.read_text(encoding="utf-8"))
    assert mirror_payload == second_payload
