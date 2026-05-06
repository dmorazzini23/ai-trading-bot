from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_sync_env_runtime_fails_when_explicit_source_missing(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "AI_TRADING_ENV_SRC": str(tmp_path / "missing.env"),
            "AI_TRADING_RUNTIME_ENV_DST": str(tmp_path / "runtime.env"),
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/sync_env_runtime.sh"],
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "AI_TRADING_ENV_SRC does not exist" in proc.stderr
    assert not (tmp_path / "runtime.env").exists()
