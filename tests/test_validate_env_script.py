import os
import subprocess
import sys
from pathlib import Path


def test_validate_env_main_exits_zero_without_credentials():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    for key in ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL"]:
        env.pop(key, None)
    env["PYTHONPATH"] = str(repo_root)
    script = repo_root / "ai_trading" / "validation" / "validate_env.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
