# Compatibility facade for legacy imports
# Allows: from config import TradingConfig, Settings, get_settings
from ai_trading.config.settings import Settings, get_settings
from ai_trading.config.management import TradingConfig

__all__ = ["Settings", "get_settings", "TradingConfig"]
