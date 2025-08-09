"""
Public config API for ai_trading.
Use:
    from ai_trading.config import get_settings, Settings
"""
# Lazy imports to prevent import-time crashes
get_settings = None
Settings = None

def __getattr__(name):
    """Lazy import config items to prevent import-time crashes."""
    global get_settings, Settings
    if name == "get_settings":
        if get_settings is None:
            try:
                from .settings import get_settings as _gs
                get_settings = _gs
            except ImportError:
                # Fallback for missing dependencies
                def get_settings():
                    return None
        return get_settings
    elif name == "Settings":
        if Settings is None:
            try:
                from .settings import Settings as _S
                Settings = _S
            except ImportError:
                # Fallback for missing dependencies
                class Settings:
                    pass
        return Settings
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Add file path configuration for compatibility
import os

# File path configuration for missing files handling
SLIPPAGE_LOG_PATH = os.getenv("SLIPPAGE_LOG_PATH", "slippage.csv")  
TICKERS_FILE_PATH = os.getenv("TICKERS_FILE_PATH", "tickers.csv")

__all__ = ["get_settings", "Settings", "SLIPPAGE_LOG_PATH", "TICKERS_FILE_PATH"]