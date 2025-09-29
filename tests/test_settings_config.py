import pytest
from ai_trading.logging import logger
from ai_trading.settings import Settings, get_position_size_min_usd, get_settings
from pydantic import ValidationError


def test_settings_defaults(monkeypatch):
    """Defaults should populate sane values."""  # AI-AGENT-REF
    for key in [
        "ALPACA_DATA_FEED",
        "ALPACA_ADJUSTMENT",
        "CAPITAL_CAP",
        "DOLLAR_RISK_LIMIT",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("AI_TRADING_POSITION_SIZE_MIN_USD", raising=False)
    s = Settings()
    assert s.alpaca_data_feed == "iex"
    assert s.alpaca_adjustment == "all"
    assert s.capital_cap == 0.25
    assert s.dollar_risk_limit == 0.05
    assert s.position_size_min_usd == 25.0
    assert get_position_size_min_usd() == 25.0


def test_sip_feed_falls_back(monkeypatch):
    """SIP feed requests fall back to IEX when not explicitly allowed."""  # AI-AGENT-REF
    monkeypatch.delenv("ALPACA_ALLOW_SIP", raising=False)
    s = Settings()
    s.data_feed = "sip"
    assert s.alpaca_data_feed == "iex"
    assert s.data_feed == "iex"


def test_settings_invalid_risk(monkeypatch):
    """Invalid risk values raise ValidationError."""  # AI-AGENT-REF
    import pydantic

    if "tests/stubs" in getattr(pydantic, "__file__", ""):
        pytest.skip("pydantic stub does not validate values")
    monkeypatch.setenv("CAPITAL_CAP", "0")
    monkeypatch.setenv("DOLLAR_RISK_LIMIT", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_main_startup_log(caplog):
    """Startup log emits DATA_CONFIG line."""  # AI-AGENT-REF
    settings = get_settings()
    with caplog.at_level("INFO"):
        logger.info(
            "DATA_CONFIG feed=%s adjustment=%s timeframe=1Day/1Min provider=alpaca",
            settings.alpaca_data_feed,
            settings.alpaca_adjustment,
        )
    assert "DATA_CONFIG feed=iex adjustment=all timeframe=1Day/1Min provider=alpaca" in caplog.text


def test_current_qty_no_position():
    """Helper returns 0 when position missing."""  # AI-AGENT-REF
    pytest.importorskip("numpy")
    from ai_trading.core.bot_engine import _current_qty

    class Ctx:
        position_map = {}

    assert _current_qty(Ctx(), "XYZ") == 0


def test_cfg_data_feed_updates_default_feed(monkeypatch):
    """Mutating ``CFG.data_feed`` propagates to module-level fallbacks."""

    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    pytest.importorskip("numpy")
    import ai_trading.config as config_pkg
    from ai_trading.config import settings as config_settings
    from ai_trading.data import fetch as data_fetch
    from ai_trading.core import bot_engine

    cfg = config_settings.get_settings()
    original_feed = cfg.data_feed
    try:
        cfg.data_feed = "sip"
        assert cfg.data_feed == "sip"
        assert data_fetch.get_default_feed() == "sip"
        assert data_fetch._DEFAULT_FEED == "sip"
        assert bot_engine.get_default_feed() == "sip"
        assert bot_engine._DEFAULT_FEED == "sip"
        assert config_pkg.DATA_FEED_INTRADAY == "sip"
    finally:
        cfg.data_feed = original_feed
        expected_feed = cfg.data_feed
        assert data_fetch.get_default_feed() == expected_feed
        assert data_fetch._DEFAULT_FEED == expected_feed
        assert bot_engine.get_default_feed() == expected_feed
        assert bot_engine._DEFAULT_FEED == expected_feed
        assert config_pkg.DATA_FEED_INTRADAY == expected_feed
