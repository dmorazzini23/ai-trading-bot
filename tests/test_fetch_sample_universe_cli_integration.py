"""Integration test for the fetch_sample_universe CLI."""

import os
import subprocess
import sys
from pathlib import Path


def test_cli_runs_with_no_symbols():
    """The CLI should execute and exit cleanly even with no symbols."""

    env = os.environ.copy()
    env["SAMPLE_UNIVERSE"] = ""  # avoid network calls
    cmd = [sys.executable, "-m", "ai_trading.tools.fetch_sample_universe"]
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stdout
