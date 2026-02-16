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
