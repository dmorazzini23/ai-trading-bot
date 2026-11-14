"""Smoke tests for CLI commands."""

import subprocess
import sys


def _run(cmd: list[str]) -> int:
    """Run command and return exit code."""
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,  # We want to check the return code manually
    ).returncode


def test_ai_trade_dry_run_exits_zero():
    """Test that ai-trade --dry-run exits with code 0."""
    assert (
        _run(
            [
                sys.executable,
                "-m",
                "ai_trading.__main__",
                "--dry-run",
                "--symbols",
                "AAPL,MSFT",
            ]
        )
        == 0
    )




