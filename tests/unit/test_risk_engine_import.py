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


def test_risk_engine_tolerates_stubbed_data_client(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("ALPACA_API_URL", "https://paper-api.alpaca.markets")

    class StubClient:
        def __init__(self, *args, **kwargs):  # noqa: D401, ARG002
            raise ImportError("stub client active")

    monkeypatch.setattr("ai_trading.risk.engine.StockHistoricalDataClient", StubClient)

    from ai_trading.risk.engine import RiskEngine

    engine = RiskEngine()

    assert getattr(engine, "data_client", None) is None

