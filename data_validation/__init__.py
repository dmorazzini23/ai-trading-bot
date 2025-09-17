"""Compatibility alias for :mod:`ai_trading.data_validation`.

This lightweight module re-exports the public API from
``ai_trading.data_validation`` so that both ``import data_validation`` and
``from data_validation import ...`` remain supported.
"""

from ai_trading.data_validation import *  # noqa: F401,F403
from ai_trading.data_validation import __all__  # noqa: F401

__all__ = list(__all__)
