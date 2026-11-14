"""
Order Health Monitor - Enhanced monitoring and alerting for order management.

Addresses the critical issues:
- Orders stuck in "NEW" status without proper timeout handling
- Partial fills not being handled optimally
- Need for comprehensive order status monitoring
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from ai_trading.config import management as config
from ai_trading.config.management import get_trading_config
from ai_trading.exc import RequestException
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
from ai_trading.logging import get_logger
from ai_trading.utils.time import safe_utcnow
from ai_trading.utils.timing import sleep as psleep

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from ai_trading.config.runtime import TradingConfig

CONFIG = cast('TradingConfig', get_trading_config())
_ORDER_MONITOR_EXCEPTIONS = AI_TRADING_FALLBACK_EXCEPTIONS + (RequestException,)

@dataclass
class OrderInfo:
    order_id: str
    symbol: str
    side: str
    qty: int
    submitted_time: float
    last_status: str
    cancel_attempted: bool = False

    @property
    def quantity(self) -> int:
        return self.qty
_active_orders: dict[str, OrderInfo] = {}
_order_tracking_lock = threading.Lock()
logger = get_logger(__name__)


_order_health_monitor = None

