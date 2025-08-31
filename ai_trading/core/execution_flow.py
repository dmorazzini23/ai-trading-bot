from __future__ import annotations

"""Execution flow helpers decoupled from bot_engine."""

from json import JSONDecodeError
from typing import Any
import time as pytime

from ai_trading.logging import get_logger

logger = get_logger(__name__)


def poll_order_fill_status(ctx: Any, order_id: str, timeout: int = 120) -> None:
    """Poll broker for order fill status until it is no longer open.

    Honors the provided ``timeout`` by sleeping in short intervals instead of
    a fixed interval so tests can set small timeouts without hanging.
    """
    start = pytime.time()
    interval = 0.2 if timeout <= 1 else 1.0
    while pytime.time() - start < timeout:
        try:
            od = ctx.api.get_order(order_id)  # type: ignore[attr-defined]
            status = getattr(od, "status", "")
            filled = getattr(od, "filled_qty", "0")
            if status not in {"new", "accepted", "partially_filled"}:
                logger.info(
                    "ORDER_FINAL_STATUS",
                    extra={
                        "order_id": order_id,
                        "status": status,
                        "filled_qty": filled,
                    },
                )
                return
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as e:
            logger.warning(f"[poll_order_fill_status] failed for {order_id}: {e}")
            return
        remaining = timeout - (pytime.time() - start)
        if remaining <= 0:
            break
        pytime.sleep(min(interval, remaining))

__all__ = ["poll_order_fill_status"]

