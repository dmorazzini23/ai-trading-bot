import os
import subprocess
import sys
from pathlib import Path

from ai_trading.validation import validate_env


def test_validate_specific_env_var_redacts_secret(monkeypatch):
    monkeypatch.setenv("ALPACA_SECRET_KEY", "super-secret-value")

    result = validate_env.validate_specific_env_var("ALPACA_SECRET_KEY")

    assert result["status"] == "set"
    assert result["value"] == "***"
    assert result["length"] == len("super-secret-value")


def test_validate_env_main_exits_nonzero_without_credentials():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    for key in ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_TRADING_BASE_URL"]:
        env.pop(key, None)
    for key in ["DRY_RUN", "AI_TRADING_DRY_RUN", "VALIDATE_ENV_DRY_RUN", "PYTEST_RUNNING", "TESTING", "PYTEST_CURRENT_TEST"]:
        env.pop(key, None)
    env["PYTHONPATH"] = str(repo_root)
    script = repo_root / "ai_trading" / "validation" / "validate_env.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1


def test_validate_env_main_allows_explicit_dry_run_without_credentials():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    for key in ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_TRADING_BASE_URL"]:
        env.pop(key, None)
    env["DRY_RUN"] = "1"
    env["PYTHONPATH"] = str(repo_root)
    script = repo_root / "ai_trading" / "validation" / "validate_env.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
