"""Lightweight entrypoint for production execution tasks.

The original :func:`run` helper executed asynchronous work internally but was
implemented as a regular function.  When callers attempted to use
``asyncio.run(run())`` the function returned ``None`` immediately, leaving the
background coroutine unawaited.  Exposing ``run`` as an ``async`` coroutine
allows straightforward execution via ``asyncio.run`` and prevents the
"coroutine was never awaited" warning.
"""
from __future__ import annotations

import asyncio
from typing import Any

from ai_trading.core.enums import RiskLevel
from ai_trading.production_system import ProductionTradingSystem


async def run(
    account_equity: float = 100_000.0,
    risk_level: RiskLevel = RiskLevel.MODERATE,
) -> ProductionTradingSystem:
    """Create and return a :class:`ProductionTradingSystem` instance.

    The coroutine performs a tiny ``await`` so that callers can drive it with
    :func:`asyncio.run` in tests or scripts.  A fully initialised trading
    system is returned for further interaction by the caller.
    """
    await asyncio.sleep(0)
    return ProductionTradingSystem(account_equity, risk_level)


__all__ = ["run"]
