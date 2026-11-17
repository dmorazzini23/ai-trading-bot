"""
Advanced position sizing and portfolio allocation strategies.

Provides volatility targeting, risk parity, correlation-based sizing,
and other institutional-grade position sizing methodologies.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
import numpy as np
from json import JSONDecodeError
from ai_trading.logging import logger, logger_once
from ai_trading.exc import RequestException

@dataclass
class _EquityCache:
    ts: datetime | None = None
    equity: float | None = None


_equity_cache = _EquityCache()


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _fetch_equity(ctx, *, force_refresh: bool = False, ttl_seconds: int = 3600) -> float:
    """Fetch account equity once and cache with timestamp.

    Parameters
    ----------
    ctx:
        Runtime context expected to expose ``api`` with ``get_account``.
    force_refresh:
        When ``True`` ignore any cached value and refresh from the API.
    ttl_seconds:
        Maximum age of a cached value before a refresh is attempted.
    """

    now = _now()
    if (
        not force_refresh
        and _equity_cache.equity is not None
        and _equity_cache.ts is not None
        and now - _equity_cache.ts < timedelta(seconds=ttl_seconds)
    ):
        return _equity_cache.equity

    api = getattr(ctx, "api", None)
    try:
        acct = api.get_account()
        eq = float(getattr(acct, "equity", 0.0) or 0.0)
        _equity_cache.equity = eq
        _equity_cache.ts = now
        try:  # set paper flag if available
            base = getattr(ctx, "alpaca_base_url", None) or getattr(api, "base_url", "")
            api.paper = "paper" in str(base).lower()
        except AttributeError:  # best effort only
            pass
        try:  # reset global disabled alert
            import ai_trading.data.fetch as _df

            _df._ALPACA_DISABLED_ALERTED = False
        except ImportError:  # pragma: no cover - defensive
            pass
        return eq
    except (AttributeError, RequestException, TimeoutError, TypeError, ValueError):
        _equity_cache.equity = 0.0
        _equity_cache.ts = now
        logger_once.warning("EQUITY_MISSING", key="equity_missing")
        return 0.0

COMMON_EXC = (TypeError, ValueError, KeyError, JSONDecodeError, RequestException, TimeoutError, ImportError)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

def _risk_fraction(risk_level: Any) -> float:
    """Return the maximum portfolio fraction for a risk-level token."""

    level = str(getattr(risk_level, "value", risk_level) or "moderate").lower()
    return {
        "conservative": 0.02,
        "moderate": 0.05,
        "aggressive": 0.10,
    }.get(level, 0.05)


def position_size(
    symbol: str,
    signal_strength: float,
    account_equity: float,
    risk_level: Any = "moderate",
    *,
    current_price: float | None = None,
) -> int:
    """Calculate a bounded long-side share quantity for bot-engine sizing hooks."""

    try:
        if not str(symbol).strip():
            return 0
        equity = float(account_equity)
        signal = float(signal_strength)
        if not np.isfinite(equity) or equity <= 0 or not np.isfinite(signal) or signal <= 0:
            return 0
        target_notional = equity * _risk_fraction(risk_level) * min(signal, 1.0)
        if current_price is not None:
            price = float(current_price)
            if not np.isfinite(price) or price <= 0:
                return 0
            return max(0, int(target_notional / price))
        return max(0, int(target_notional))
    except COMMON_EXC as e:
        logger.warning("POSITION_SIZE_FAILED symbol=%s error=%s", symbol, e)
        return 0

