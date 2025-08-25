import subprocess, sys


def test_help_exits_quietly():
    proc = subprocess.run(
        [sys.executable, "-m", "ai_trading.runner", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    out = (proc.stdout + proc.stderr).lower()
    assert "yfinance" not in out

