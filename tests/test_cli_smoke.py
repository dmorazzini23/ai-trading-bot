"""Smoke tests for CLI commands."""

import subprocess
import sys

def _run(cmd: list[str]) -> int:
    """Run command and return exit code."""
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ).returncode

def test_ai_trade_dry_run_exits_zero():
    """Test that ai-trade --dry-run exits with code 0."""
    assert _run([sys.executable, "-m", "ai_trading.__main__", "--dry-run", "--symbols", "AAPL,MSFT"]) == 0

def test_ai_backtest_dry_run_exits_zero():
    """Test that ai-backtest --dry-run exits with code 0.""" 
    # For now, test the module directly until console scripts are installed
    assert _run([sys.executable, "-c", "from ai_trading.__main__ import run_backtest; run_backtest()", "--dry-run", "--symbols", "AAPL,MSFT"]) == 0

def test_ai_health_dry_run_exits_zero():
    """Test that ai-health --dry-run exits with code 0."""
    # For now, test the module directly until console scripts are installed
    assert _run([sys.executable, "-c", "from ai_trading.__main__ import run_healthcheck; run_healthcheck()", "--dry-run"]) == 0