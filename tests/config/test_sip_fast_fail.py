import importlib

import pytest


def test_sip_requires_entitlement(monkeypatch):
    monkeypatch.delenv("ALPACA_ALLOW_SIP", raising=False)
    monkeypatch.delenv("ALPACA_HAS_SIP", raising=False)
    monkeypatch.setenv("DATA_FEED_INTRADAY", "sip")

    runtime = importlib.import_module("ai_trading.config.runtime")

    with pytest.raises(ValueError) as excinfo:
        runtime.TradingConfig.from_env(allow_missing_drawdown=True)

    message = str(excinfo.value)
    assert "DATA_FEED_INTRADAY=sip" in message
    assert "DEPLOYING.md#alpaca-intraday-feed" in message
