"""Lightweight Alpaca REST helpers with retry and validation.

DEPRECATED: This module will be removed in a future version.
Please import from ai_trading.core.bot_engine or ai_trading.execution instead.
"""

from __future__ import annotations

import warnings
warnings.warn(
    "Importing from root alpaca_api.py is deprecated. Use 'from ai_trading.core import bot_engine' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the original implementation to preserve compatibility
from alpaca_api_original import *  # noqa: F401,F403