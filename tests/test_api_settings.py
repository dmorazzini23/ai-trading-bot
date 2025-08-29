# AI-AGENT-REF: verify API host/port settings and defaults
from ai_trading.config.settings import get_settings


def test_api_host_port_defaults_present():
    get_settings.cache_clear()
    s = get_settings()
    assert hasattr(s, "api_host")
    assert hasattr(s, "api_port")
    assert s.api_host == "0.0.0.0"
    assert s.api_port == 9001


def test_sentiment_fields_present():
    get_settings.cache_clear()
    s = get_settings()
    assert hasattr(s, "sentiment_api_key")
    assert hasattr(s, "sentiment_api_url")
