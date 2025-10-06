import pytest

from ai_trading.config import runtime


def test_sip_requires_entitlement(monkeypatch) -> None:
    monkeypatch.setenv("DATA_FEED_INTRADAY", "sip")
    monkeypatch.delenv("ALPACA_ALLOW_SIP", raising=False)
    monkeypatch.delenv("ALPACA_HAS_SIP", raising=False)
    monkeypatch.setenv("MAX_DRAWDOWN_THRESHOLD", "0.1")
    runtime.get_trading_config.cache_clear()

    with pytest.raises(ValueError) as excinfo:
        runtime.TradingConfig.from_env(allow_missing_drawdown=True)

    assert "DATA_FEED_INTRADAY=sip" in str(excinfo.value)
