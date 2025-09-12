import os
import ai_trading.utils.exec as exec_utils


def test_ci_env_safeguards(monkeypatch):
    """CI env should force offline, paper-mode tests."""
    monkeypatch.setenv("AI_TRADING_OFFLINE_TESTS", "1")
    monkeypatch.setenv("ALPACA_ENV", "paper")
    env = exec_utils._sanitize_executor_env()
    assert env.get("AI_TRADING_OFFLINE_TESTS") == "1"
    assert os.getenv("ALPACA_ENV") == "paper"
    # Default test fixtures mask API keys with dummy values
    assert os.getenv("ALPACA_API_KEY") == "dummy"
    assert os.getenv("ALPACA_SECRET_KEY") == "dummy"
