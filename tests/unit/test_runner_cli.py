import subprocess
import sys


def test_runner_help_exits_zero():
    proc = subprocess.run(
        [sys.executable, "-m", "ai_trading.runner", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "usage" in proc.stdout.lower()

