"""Backwards compatibility re-export for ``ai_trading.data_validation``.

This module provides a flat import path for legacy code importing
``data_validation`` directly from the project root.
"""

from ai_trading.data_validation import *  # noqa: F401,F403
from ai_trading.data_validation import __all__ as _all

__all__ = list(_all)

