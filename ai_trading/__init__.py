"""AI Trading Bot Module - Institutional Grade Trading Platform

This module contains the core trading bot functionality and institutional-grade
components for professional trading operations including:

- Core trading enums and constants
- Database models and connection management
- Kelly Criterion risk management
- Strategy framework and execution engine
- Performance monitoring and alerting
- Institutional-grade order management

The platform is designed for institutional-scale operations with proper
risk controls, monitoring, and compliance capabilities.
"""

from __future__ import annotations

import importlib.util as _ils
import os as _os
import sys as _sys

from .data_fetcher import _MINUTE_CACHE, YFIN_AVAILABLE

__version__ = "2.0.0"


def _module_ok(name: str) -> bool:  # AI-AGENT-REF: lightweight optional import check
    try:
        return _ils.find_spec(name) is not None
    except Exception:  # noqa: BLE001
        return False


ALPACA_AVAILABLE = any(
    [
        _module_ok("alpaca"),
        _module_ok("alpaca_trade_api"),
        _module_ok("alpaca.trading"),
        _module_ok("alpaca.data"),
    ]
) and _os.environ.get("TESTING", "").lower() not in {"1", "true:force_unavailable"}

FINNHUB_AVAILABLE = _module_ok("finnhub")

# Import-light init - only expose version and basic metadata
__all__ = [
    "__version__",
    "_MINUTE_CACHE",
    "ALPACA_AVAILABLE",
    "FINNHUB_AVAILABLE",
    "YFIN_AVAILABLE",
]

# AI-AGENT-REF: expose validate_env module at top-level for tests
try:
    from .tools import validate_env as _validate_env_mod

    _sys.modules.setdefault("validate_env", _validate_env_mod)
except Exception:  # pragma: no cover  # noqa: BLE001
    pass
