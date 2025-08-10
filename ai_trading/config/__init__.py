"""
Public config API for ai_trading.
Use:
    from ai_trading.config import get_settings, Settings
"""
from typing import TYPE_CHECKING

def __getattr__(name: str):
    """
    Lazy import config items to prevent import-time crashes and circulars.
    We resolve symbols via `management.py`, which re-exports from `settings.py`.
    """
    if name in {"get_settings", "Settings"}:
        try:
            from . import management as _management
            return getattr(_management, name)
        except Exception as e:
            raise ImportError(
                "'ai_trading.config.management' has no attribute "
                f"'{name}'"
            ) from e
    
    # Forward other config attributes from management module
    if name in {"MODEL_PATH", "VERBOSE_LOGGING", "BOT_MODE", "USE_RL_AGENT", 
                "FINNHUB_API_KEY", "NEWS_API_KEY", "SENTIMENT_API_KEY", 
                "SENTIMENT_API_URL", "IEX_API_TOKEN", "ALPACA_PAPER",
                "TRADE_LOG_FILE", "RL_MODEL_PATH", "HALT_FLAG_PATH"}:
        try:
            from . import management as _management
            return getattr(_management, name)
        except Exception as e:
            raise ImportError(
                f"'ai_trading.config.management' has no attribute '{name}'"
            ) from e
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

if TYPE_CHECKING:
    # These help static type checkers without triggering imports at runtime.
    from .settings import Settings as Settings  # noqa: F401
    from .management import get_settings as get_settings  # noqa: F401

# Add file path configuration for compatibility
import os

# File path configuration for missing files handling
SLIPPAGE_LOG_PATH = os.getenv("SLIPPAGE_LOG_PATH", "slippage.csv")  
TICKERS_FILE_PATH = os.getenv("TICKERS_FILE_PATH", "tickers.csv")

__all__ = ["get_settings", "Settings", "SLIPPAGE_LOG_PATH", "TICKERS_FILE_PATH"]