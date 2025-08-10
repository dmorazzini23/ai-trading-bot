"""
Public config API for ai_trading.
Use:
    from ai_trading.config import get_settings, Settings
"""

from typing import TYPE_CHECKING

# Direct imports - no more magic __getattr__
from .settings import get_settings, Settings

if TYPE_CHECKING:
    # These help static type checkers without triggering imports at runtime.
    from .settings import get_settings as get_settings  # noqa: F401
    from .settings import Settings as Settings  # noqa: F401

# Add file path configuration for compatibility
import os

# File path configuration for missing files handling
SLIPPAGE_LOG_PATH = os.getenv("SLIPPAGE_LOG_PATH", "slippage.csv")
TICKERS_FILE_PATH = os.getenv("TICKERS_FILE_PATH", "tickers.csv")

__all__ = ["get_settings", "Settings", "SLIPPAGE_LOG_PATH", "TICKERS_FILE_PATH"]
