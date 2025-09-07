import ai_trading.main_extended as main_extended
import pytest


def test_validate_environment_missing(monkeypatch):
    """validate_environment errors when required variables are absent."""
    for var in [
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "WEBHOOK_SECRET",
        "CAPITAL_CAP",
        "DOLLAR_RISK_LIMIT",
    ]:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError):
        main_extended.validate_environment()
