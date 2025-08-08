"""
Public config API for ai_trading.
Use:
    from ai_trading.config import get_settings, Settings
"""
from .settings import get_settings, Settings

__all__ = ["get_settings", "Settings"]