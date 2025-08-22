import pytest
from pydantic import ValidationError

from ai_trading.config.settings import get_settings
from ai_trading.core.bot_engine import _current_qty
from ai_trading.main import logger
from ai_trading.settings import Settings


def test_settings_defaults(monkeypatch):
    """Defaults should populate sane values."""  # AI-AGENT-REF
    for key in [
        "ALPACA_DATA_FEED",
        "ALPACA_ADJUSTMENT",
        "CAPITAL_CAP",
        "DOLLAR_RISK_LIMIT",
    ]:
        monkeypatch.delenv(key, raising=False)
    s = Settings()
    assert s.alpaca_data_feed == "iex"
    assert s.alpaca_adjustment == "all"
    assert s.capital_cap == 0.04
    assert s.dollar_risk_limit == 0.05


def test_settings_invalid_risk(monkeypatch):
    """Invalid risk values raise ValidationError."""  # AI-AGENT-REF
    monkeypatch.setenv("CAPITAL_CAP", "0")
    monkeypatch.setenv("DOLLAR_RISK_LIMIT", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_main_startup_log(caplog):
    """Startup log emits DATA_CONFIG line."""  # AI-AGENT-REF
    S = get_settings()
    with caplog.at_level("INFO"):
        logger.info(
            "DATA_CONFIG feed=%s adjustment=%s timeframe=1Day/1Min provider=alpaca",
            S.alpaca_data_feed,
            S.alpaca_adjustment,
        )
    assert "DATA_CONFIG feed=iex adjustment=all timeframe=1Day/1Min provider=alpaca" in caplog.text


def test_current_qty_no_position():
    """Helper returns 0 when position missing."""  # AI-AGENT-REF
    class Ctx:
        position_map = {}

    assert _current_qty(Ctx(), "XYZ") == 0

