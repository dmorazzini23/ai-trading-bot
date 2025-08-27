import os


def test_ci_env_safeguards():
    """CI env should force offline, paper-mode tests."""
    assert os.getenv("AI_TRADING_OFFLINE_TESTS") == "1"
    assert os.getenv("ALPACA_ENV") == "paper"
    # Default test fixtures mask API keys with dummy values
    assert os.getenv("ALPACA_API_KEY") == "dummy"
    assert os.getenv("ALPACA_SECRET_KEY") == "dummy"
