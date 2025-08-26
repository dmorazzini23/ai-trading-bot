# AI-AGENT-REF: ensure RiskEngine imports without crash
import pytest


def test_import_risk_engine():
    # Import must not raise on class creation
    from ai_trading.risk.engine import RiskEngine  # noqa: F401

    assert True


def test_risk_engine_validates_env_at_runtime(monkeypatch):
    keys = (
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "ALPACA_BASE_URL",
        "WEBHOOK_SECRET",
        "CAPITAL_CAP",
        "DOLLAR_RISK_LIMIT",
        "MAX_POSITION_SIZE",
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    from ai_trading.risk.engine import RiskEngine

    with pytest.raises(RuntimeError):
        RiskEngine()

