from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

"""Utility helpers for handling Alpaca IEX feed fallbacks.

This module implements a very small subset of the logic that the main
``ai_trading.data.fetch`` package provides.  It focuses on situations where the
Alpaca IEX feed returns an empty payload.  When that occurs we record the
attempt, increment the global ``_IEX_EMPTY_COUNTS`` tracker and, when allowed,
retry the request against the SIP feed.  The counter persists even when the
SIP feed succeeds so future calls can skip the IEX feed until it provides
data again.  If both feeds return empty results an error is logged so callers
can react accordingly.

The functions here are intentionally lightweight so that they can be imported in
isolation for unit tests without pulling in the entire ``fetch`` module.
"""

import importlib
import sys
import types
from datetime import datetime
from typing import Any

from ai_trading.utils.env import get_alpaca_data_base_url, get_alpaca_http_headers
from ai_trading.utils.http import clamp_request_timeout
from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.logging import get_logger
from .fallback_order import demote_provider, promote_high_resolution
from .metrics import inc_empty_payload, mark_skipped, inc_unauthorized_sip

# Import shared state from the package's ``__init__``.  These variables are
# defined there and are re-used across modules.
from . import (
    _HTTP_SESSION,
    _ALLOW_SIP,
    _SIP_UNAUTHORIZED,
    _IEX_EMPTY_COUNTS,
    _IEX_EMPTY_THRESHOLD,
    _ALPACA_CONSECUTIVE_FAILURE_THRESHOLD,
    _consecutive_failure_count,
    _record_alpaca_failure_event,
    _clear_alpaca_failure_events,
)

pd = load_pandas()
logger = get_logger(__name__)
_IMPORTED_ALLOW_SIP = _ALLOW_SIP
_IMPORTED_SIP_UNAUTHORIZED = _SIP_UNAUTHORIZED


def _resolve_fetch_callable(name: str, fallback: Any) -> Any:
    """Return the latest callable from ``ai_trading.data.fetch`` when available."""

    try:
        fetch_mod = sys.modules.get("ai_trading.data.fetch")
        if not isinstance(fetch_mod, types.ModuleType):
            fetch_mod = importlib.import_module("ai_trading.data.fetch")
        candidate = getattr(fetch_mod, name, None)
        if callable(candidate):
            return candidate
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        pass
    return fallback


def _resolve_fetch_value(name: str, fallback: Any) -> Any:
    """Return the latest value from ``ai_trading.data.fetch`` when available."""

    try:
        fetch_mod = sys.modules.get("ai_trading.data.fetch")
        if not isinstance(fetch_mod, types.ModuleType):
            fetch_mod = importlib.import_module("ai_trading.data.fetch")
        return getattr(fetch_mod, name, fallback)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return fallback


def _resolve_live_fetch_value(name: str, imported_value: Any) -> Any:
    """Return mutable parent-package state without freezing imported globals."""

    local_value = globals().get(name, imported_value)
    if local_value != imported_value:
        return local_value
    return _resolve_fetch_value(name, imported_value)


def _sip_unauthorized() -> bool:
    """Return the current SIP authorization lockout state."""

    local_value = globals().get("_SIP_UNAUTHORIZED", _IMPORTED_SIP_UNAUTHORIZED)
    if local_value != _IMPORTED_SIP_UNAUTHORIZED:
        return bool(local_value)
    checker = _resolve_fetch_callable("_is_sip_unauthorized", None)
    if callable(checker):
        try:
            return bool(checker())
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            pass
    return bool(_resolve_live_fetch_value("_SIP_UNAUTHORIZED", _IMPORTED_SIP_UNAUTHORIZED))


def _sip_allowed() -> bool:
    """Return whether SIP fallback is currently allowed."""

    return bool(_resolve_live_fetch_value("_ALLOW_SIP", _IMPORTED_ALLOW_SIP)) and not _sip_unauthorized()


__all__: list[str] = []
