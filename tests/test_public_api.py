# AI-AGENT-REF: ensure public config API exports

def test_public_config_api_imports():
    from ai_trading.config import TradingConfig, Settings, get_settings  # noqa: F401
