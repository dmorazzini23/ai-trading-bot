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


def test_sync_env_runtime_resolves_repo_relative_paths_from_other_cwd(tmp_path: Path) -> None:
    env_src = tmp_path / "source.env"
    runtime_dst = tmp_path / "runtime.env"
    env_src.write_text("ALPACA_API_KEY=key\nALPACA_SECRET_KEY=secret\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "AI_TRADING_ENV_SRC": str(env_src),
            "AI_TRADING_RUNTIME_ENV_DST": str(runtime_dst),
        }
    )

    proc = subprocess.run(
        ["bash", str(_repo_root() / "scripts/sync_env_runtime.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert runtime_dst.exists()


def test_sync_env_runtime_fails_when_packaged_runtime_dir_missing(tmp_path: Path) -> None:
    env_src = tmp_path / "source.env"
    packaged_dir = tmp_path / "run" / "ai-trading-bot"
    runtime_dst = packaged_dir / "ai-trading-runtime.env"
    env_src.write_text("ALPACA_API_KEY=key\nALPACA_SECRET_KEY=secret\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "AI_TRADING_ENV_SRC": str(env_src),
            "AI_TRADING_PACKAGED_RUNTIME_DIR": str(packaged_dir),
            "AI_TRADING_RUNTIME_ENV_DST": str(runtime_dst),
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
    assert "packaged runtime env directory is missing" in proc.stderr
    assert not runtime_dst.exists()
