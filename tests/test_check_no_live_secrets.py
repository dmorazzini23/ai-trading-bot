from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)


def _run_guard(path: Path) -> subprocess.CompletedProcess[str]:
    script = Path(__file__).resolve().parents[1] / "tools" / "check_no_live_secrets.py"
    return subprocess.run(
        [sys.executable, str(script), "--root", str(path)],
        cwd=path,
        capture_output=True,
        text=True,
    )


def test_secret_guard_ignores_placeholders(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / ".env.example").write_text(
        "ALPACA_API_KEY=YOUR_ALPACA_API_KEY_HERE\n"
        "ALPACA_SECRET_KEY=YOUR_ALPACA_SECRET_KEY_HERE\n"
        "WEBHOOK_SECRET=changeme\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", ".env.example"], cwd=tmp_path, check=True, capture_output=True, text=True)

    result = _run_guard(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "no likely live secrets" in result.stdout.lower()


def test_secret_guard_blocks_live_like_values(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / ".env.production").write_text(
        "ALPACA_API_KEY=PKLIVEA1B2C3D4E5F6G7H8J9K0\n"
        "ALPACA_SECRET_KEY=liveSecretValue_1234567890_abcdefghijklmnopqrstuvwxyz\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", ".env.production"], cwd=tmp_path, check=True, capture_output=True, text=True)

    result = _run_guard(tmp_path)
    assert result.returncode == 1
    assert "potential live secrets detected" in result.stderr.lower()
