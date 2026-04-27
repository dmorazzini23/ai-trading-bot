from __future__ import annotations

import importlib
from types import ModuleType


def ensure_alpaca_api() -> ModuleType:
    """Return the canonical Alpaca API module without mutating imports at module load."""

    return importlib.import_module("ai_trading.alpaca_api")
