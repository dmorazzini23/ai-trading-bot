# AI-AGENT-REF: ensure RiskEngine imports without crash


def test_import_risk_engine():
    # Import must not raise on class creation
    from ai_trading.risk.engine import RiskEngine  # noqa: F401

    assert True

