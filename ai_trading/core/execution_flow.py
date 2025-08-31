from __future__ import annotations

"""Execution flow helpers decoupled from bot_engine.

This module provides concrete implementations for selected execution helpers
and re-exports core entry points for backwards compatibility.
"""

from json import JSONDecodeError
from typing import Any
import time as pytime
from threading import Thread
import csv
from datetime import UTC, datetime

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


def twap_submit(
    ctx: Any,
    symbol: str,
    total_qty: int,
    side: str,
    window_secs: int = 600,
    n_slices: int = 10,
) -> None:
    """Submit TWAP slices over the specified window."""
    from ai_trading.core.bot_engine import submit_order, APIError

    slice_qty = max(1, int(total_qty // max(1, n_slices)))
    wait_secs = float(window_secs) / max(1, n_slices)
    for _ in range(max(1, n_slices)):
        try:
            submit_order(ctx, symbol, slice_qty, side)
        except (APIError, TimeoutError, ConnectionError) as e:  # type: ignore[name-defined]
            logger.error(
                "BROKER_OP_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
            raise
        pytime.sleep(wait_secs)


def vwap_pegged_submit(
    ctx: Any, symbol: str, total_qty: int, side: str, duration: int = 300
) -> None:
    """Submit slices pegged to VWAP with simple spread-aware logic."""
    from ai_trading.core.bot_engine import (
        fetch_minute_df_safe,
        DataFetchError,
        ta,
        StockLatestQuoteRequest,
        LimitOrderRequest,
        OrderSide,
        TimeInForce,
        APIError,
        utc_now_iso,
        slippage_total,
        slippage_count,
        _slippage_log,
        slippage_lock,
        SLIPPAGE_LOG_FILE,
        safe_submit_order,
    )

    start_time = pytime.time()
    placed = 0
    while placed < total_qty and pytime.time() - start_time < duration:
        try:
            df = fetch_minute_df_safe(symbol)
        except DataFetchError:
            logger.error("[VWAP] no minute data for %s", symbol)
            break
        if df is None or df.empty:
            logger.warning("[VWAP] missing bars, aborting VWAP slice", extra={"symbol": symbol})
            break
        vwap_price = ta.vwap(df["high"], df["low"], df["close"], df["volume"]).iloc[-1]
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quote = ctx.data_client.get_stock_latest_quote(req)
            spread = (
                (quote.ask_price - quote.bid_price)
                if getattr(quote, "ask_price", None) and getattr(quote, "bid_price", None)
                else 0.0
            )
        except APIError as e:
            logger.warning(f"[vwap_slice] Alpaca quote failed for {symbol}: {e}")
            spread = 0.0
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as _:
            spread = 0.0
        if spread > 0.05:
            slice_qty = max(1, int((total_qty - placed) * 0.5))
        else:
            slice_qty = min(max(1, total_qty // 10), total_qty - placed)
        order = None
        for attempt in range(3):
            try:
                logger.info(
                    "ORDER_SENT",
                    extra={
                        "timestamp": utc_now_iso(),
                        "symbol": symbol,
                        "side": side,
                        "qty": slice_qty,
                        "order_type": "limit",
                    },
                )
                order = safe_submit_order(
                    ctx.api,
                    LimitOrderRequest(
                        symbol=symbol,
                        qty=slice_qty,
                        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                        time_in_force=TimeInForce.IOC,
                        limit_price=round(vwap_price, 2),
                    ),
                )
                logger.info(
                    "ORDER_ACK",
                    extra={
                        "symbol": symbol,
                        "order_id": getattr(order, "id", ""),
                        "status": getattr(order, "status", ""),
                    },
                )
                Thread(
                    target=poll_order_fill_status,  # local function
                    args=(ctx, getattr(order, "id", "")),
                    daemon=True,
                ).start()
                fill_price = float(getattr(order, "filled_avg_price", 0) or 0)
                if fill_price > 0:
                    slip = (fill_price - vwap_price) * 100
                    if slippage_total:
                        try:
                            slippage_total.inc(abs(slip))  # type: ignore[union-attr]
                        except Exception:
                            pass
                    if slippage_count:
                        try:
                            slippage_count.inc()  # type: ignore[union-attr]
                        except Exception:
                            pass
                    _slippage_log.append((symbol, vwap_price, fill_price, datetime.now(UTC)))
                    with slippage_lock:  # type: ignore[arg-type]
                        try:
                            with open(SLIPPAGE_LOG_FILE, "a", newline="") as sf:
                                csv.writer(sf).writerow(
                                    [utc_now_iso(), symbol, vwap_price, fill_price, slip]
                                )
                        except Exception:
                            pass
                break
            except APIError as e:
                logger.warning(f"[VWAP] APIError attempt {attempt + 1} for {symbol}: {e}")
                pytime.sleep(attempt + 1)
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
                logger.exception(f"[VWAP] slice attempt {attempt + 1} failed: {e}")
                pytime.sleep(attempt + 1)
        if order is None:
            break


# Re-export for backwards compatibility
from .bot_engine import submit_order as submit_order  # noqa: E402,F401
from .bot_engine import safe_submit_order as safe_submit_order  # noqa: E402,F401
from .bot_engine import execute_exit as execute_exit  # noqa: E402,F401

__all__ = [
    "submit_order",
    "safe_submit_order",
    "poll_order_fill_status",
    "execute_exit",
    "twap_submit",
    "vwap_pegged_submit",
]


def send_exit_order(
    ctx: Any,
    symbol: str,
    exit_qty: int,
    price: float,
    reason: str,
    raw_positions: list | None = None,
) -> None:
    """Submit an exit order (market or limit) with simple validations."""
    from ai_trading.core.bot_engine import (
        MarketOrderRequest,
        LimitOrderRequest,
        OrderSide,
        TimeInForce,
        safe_submit_order,
    )

    logger.info(
        f"EXIT_SIGNAL | symbol={symbol}  reason={reason}  exit_qty={exit_qty}  price={price}"
    )
    if raw_positions is not None and not any(
        getattr(p, "symbol", "") == symbol for p in raw_positions
    ):
        logger.info("SKIP_NO_POSITION", extra={"symbol": symbol})
        return
    try:
        pos = ctx.api.get_position(symbol)
        held_qty = int(pos.qty)
    except Exception:
        held_qty = 0
    if held_qty < exit_qty:
        logger.warning(
            f"No shares available to exit for {symbol} (requested {exit_qty}, have {held_qty})"
        )
        return
    if price <= 0.0:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=exit_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        safe_submit_order(ctx.api, req)
        return
    limit_order = safe_submit_order(
        ctx.api,
        LimitOrderRequest(
            symbol=symbol,
            qty=exit_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=price,
        ),
    )
    pytime.sleep(5)
    try:
        o2 = ctx.api.get_order(limit_order.id)
        if getattr(o2, "status", "") in {"new", "accepted", "partially_filled"}:
            ctx.api.cancel_order(limit_order.id)
            safe_submit_order(
                ctx.api,
                MarketOrderRequest(
                    symbol=symbol,
                    qty=exit_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                ),
            )
    except Exception as e:
        logger.error(
            "BROKER_OP_FAILED",
            extra={
                "cause": e.__class__.__name__,
                "detail": str(e),
                "op": "cancel",
                "order_id": getattr(limit_order, "id", ""),
            },
        )
        raise

__all__.append("send_exit_order")
