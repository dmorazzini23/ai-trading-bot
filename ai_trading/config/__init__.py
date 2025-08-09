"""
Public config API for ai_trading.
Use:
    from ai_trading.config import get_settings, Settings
"""
from typing import TYPE_CHECKING

def __getattr__(name: str):
    """
    Lazy import config items to prevent import-time crashes.
    Crucially, we do NOT predefine 'get_settings'/'Settings' in module globals,
    so 'from ai_trading.config import get_settings' resolves via this hook.
    """
    if name in {"get_settings", "Settings"}:
        try:
            from . import management as _management
        except Exception as e:
            raise ImportError(
                f"Failed to import '{name}' from ai_trading.config.management: {e}"
            ) from e
        try:
            return getattr(_management, name)
        except AttributeError as e:
            raise ImportError(
                f"'ai_trading.config.management' has no attribute '{name}'"
            ) from e
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Make symbols visible to type checkers without importing at runtime.
if TYPE_CHECKING:  # pragma: no cover
    from .management import Settings as Settings
    from .management import get_settings as get_settings

# Add file path configuration for compatibility
import os

# File path configuration for missing files handling
SLIPPAGE_LOG_PATH = os.getenv("SLIPPAGE_LOG_PATH", "slippage.csv")  
TICKERS_FILE_PATH = os.getenv("TICKERS_FILE_PATH", "tickers.csv")

__all__ = ["get_settings", "Settings", "SLIPPAGE_LOG_PATH", "TICKERS_FILE_PATH"]