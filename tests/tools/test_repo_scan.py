"""Tests for the lightweight repository scan helper."""

import subprocess
import sys
from pathlib import Path


def test_repo_scan_clean():
    """The repo scan should report a clean state."""
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo_root / "tools" / "repo_scan.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Repo scan: clean" in result.stdout

