"""Lightweight Alpaca REST helpers with retry and validation.

Orders submitted via :func:`submit_order` are recorded in ``pending_orders``.
The :func:`check_stuck_orders` task periodically scans this registry and will
cancel and resubmit any order that remains in ``PENDING_NEW`` for over
60 seconds to guard against missed status updates.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import requests
from requests import Session
from requests.exceptions import HTTPError
from alpaca.trading.stream import TradingStream
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from alerts import send_slack_alert

logger = logging.getLogger(__name__)

if "ALPACA_API_KEY" in os.environ:
    os.environ.setdefault("APCA_API_KEY_ID", os.environ["ALPACA_API_KEY"])
if "ALPACA_SECRET_KEY" in os.environ:
    os.environ.setdefault("APCA_API_SECRET_KEY", os.environ["ALPACA_SECRET_KEY"])

SHADOW_MODE = os.getenv("SHADOW_MODE", "0") == "1"
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"


_warn_counts = defaultdict(int)

# Track pending orders awaiting fill or rejection
pending_orders: dict[str, dict[str, Any]] = {}

# AI-AGENT-REF: accumulate partial fills for periodic summary
partial_fills: dict[str, dict[str, float]] = {}
# AI-AGENT-REF: track first partial fill per order
partial_fill_tracker: set[str] = set()

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets")


def _build_headers() -> dict:
    """Return authorization headers constructed from the current environment.

    Secrets are read at call time so that a configuration reload will be
    respected without keeping credentials in memory longer than necessary.
    """

    return {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
    }


def _warn_limited(key: str, msg: str, *args, limit: int = 3, **kwargs) -> None:
    """Log ``msg`` up to ``limit`` times for a given ``key``."""

    if _warn_counts[key] < limit:
        logger.warning(msg, *args, **kwargs)
        _warn_counts[key] += 1
        if _warn_counts[key] == limit:
            logger.warning("Further '%s' warnings suppressed", key)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def alpaca_get(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Perform a GET request to the Alpaca API with retries."""

    url = f"{ALPACA_BASE_URL}{endpoint}"
    try:
        response = requests.get(
            url, headers=_build_headers(), params=params, timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.error("alpaca_get request failed for %s: %s", endpoint, exc)
        raise


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def get_account() -> Optional[Dict[str, Any]]:
    """Return account details or ``None`` on failure."""
    try:
        data = alpaca_get("/v2/account")
    except requests.exceptions.RequestException as exc:
        logger.error("API request failed: %s", exc)
        raise
    except Exception as exc:  # pragma: no cover - safety
        logger.error("Unexpected error: %s", exc)
        raise
    if data is None:
        logger.error("Failed to get Alpaca account data")
    return data


def submit_order(api, req, log: logging.Logger | None = None):
    """Submit an order with retry and optional shadow mode."""
    log = log or logger
    if DRY_RUN:
        log.info(
            "DRY RUN: would submit order for %s of size %s",
            getattr(req, "symbol", ""),
            getattr(req, "qty", 0),
        )
        return {
            "status": "dry_run",
            "symbol": getattr(req, "symbol", ""),
            "qty": getattr(req, "qty", 0),
        }
    if SHADOW_MODE:
        log.info(
            "SHADOW_MODE: Would place order: %s %s %s %s %s",
            getattr(req, "symbol", ""),
            getattr(req, "qty", ""),
            getattr(req, "side", ""),
            req.__class__.__name__,
            getattr(req, "time_in_force", ""),
        )
        return {
            "status": "shadow",
            "symbol": getattr(req, "symbol", ""),
            "qty": getattr(req, "qty", 0),
        }

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            order = api.submit_order(order_data=req)
            if hasattr(order, "status_code") and getattr(order, "status_code") == 429:
                raise HTTPError(
                    "API rate limit exceeded (429)"
                )  # AI-AGENT-REF: use explicit HTTPError import
            if getattr(order, "id", None):
                pending_orders[str(order.id)] = {
                    "symbol": getattr(req, "symbol", ""),
                    "req": req,
                    "timestamp": time.monotonic(),
                    "status": "PENDING_NEW",
                }
            return order
        except HTTPError as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                wait = attempt * 2
                _warn_limited(
                    "order-rate-limit",
                    "Rate limit hit for Alpaca order (attempt %s/%s), sleeping %ss",
                    attempt,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
                continue
            log.error("HTTPError in Alpaca submit_order: %s", e, exc_info=True)
            send_slack_alert("HTTP error submitting order: %s" % e)
            if attempt == max_retries:
                raise
        except requests.exceptions.RequestException as exc:
            log.error("API request failed: %s", exc)
            raise
        except Exception as exc:
            log.error(
                "Unexpected error: %s",
                exc,
                exc_info=True,
            )
            if attempt == max_retries:
                send_slack_alert(
                    "Failed to submit order after %s attempts: %s" % (max_retries, exc)
                )
                raise
            time.sleep(attempt * 2)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def fetch_bars(
    api: Any,
    symbols: list[str] | str,
    timeframe: Any,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Return OHLCV bars ``DataFrame`` from Alpaca REST API."""

    bars_df = api.get_bars(symbols, timeframe, start=start, end=end).df

    if bars_df is None or bars_df.empty:
        logger.warning("No data received for symbols: %s", symbols)
        return pd.DataFrame()

    bars_df = bars_df.reset_index()
    logger.info(
        "Fetched data shape: %s cols=%s",
        bars_df.shape,
        bars_df.columns.tolist(),
    )
    if "timestamp" not in bars_df.columns:
        logger.error("Timestamp column missing in bars_df for symbols: %s", symbols)
        return pd.DataFrame()

    return bars_df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]


async def handle_trade_update(event, state=None) -> None:
    """Process order status updates from the Alpaca stream."""
    try:
        order = getattr(event, "order", {})
        order_id = getattr(order, "id", getattr(event, "order_id", ""))
        symbol = getattr(order, "symbol", "")
        status = getattr(event, "event", getattr(order, "status", ""))
        filled_qty = getattr(order, "filled_qty", None)
        fill_price = getattr(order, "filled_avg_price", None)
        extra = (
            f" [filled_qty={filled_qty} @ price={fill_price}]"
            if filled_qty and fill_price
            else ""
        )
        if status in {"partial_fill", "partial", "partially_filled"}:
            if str(order_id) not in partial_fill_tracker:
                logger.debug(
                    "ORDER_PARTIAL_FILL | %s qty=%s price=%s",
                    symbol,
                    filled_qty,
                    fill_price,
                )
                partial_fill_tracker.add(str(order_id))
            partial_fills[str(order_id)] = {
                "symbol": symbol,
                "qty": float(filled_qty or 0),
                "price": float(fill_price or 0),
                "ts": time.monotonic(),
            }
        else:
            logger.info(
                "Order update: %s (ID: %s) -> %s%s",
                symbol,
                order_id,
                status,
                extra,
            )
            if status == "fill":
                logger.info(
                    "ORDER_FILLED | %s qty=%s price=%s",
                    symbol,
                    filled_qty,
                    fill_price,
                )
                partial_fill_tracker.discard(str(order_id))
            if status in {"fill", "canceled", "rejected"}:
                partial_fills.pop(str(order_id), None)
        info = pending_orders.get(str(order_id))
        if info:
            info["status"] = status
            if status not in {"PENDING_NEW", "NEW"}:
                pending_orders.pop(str(order_id), None)
        if state and status == "fill":
            state.trade_cooldowns[symbol] = dt.datetime.now(dt.timezone.utc)
            side = getattr(order, "side", getattr(event, "side", ""))
            if side:
                state.last_trade_direction[symbol] = str(side).lower()
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.exception("Error processing trade update: %s", exc)


async def _partial_fill_summary_loop() -> None:
    """Periodically log consolidated partial fill summaries."""
    while True:
        await asyncio.sleep(2)
        now = time.monotonic()
        stale = {
            oid: info
            for oid, info in list(partial_fills.items())
            if now - info.get("ts", 0) >= 2
        }
        for oid, info in stale.items():
            logger.info(
                "ORDER_PROGRESS: %s partial_fill total=%s @ avg=%.2f",
                info["symbol"],
                int(info["qty"]),
                info["price"],
            )
            partial_fills.pop(oid, None)


async def check_stuck_orders(api) -> None:
    """Cancel and resubmit orders stuck in PENDING_NEW state."""
    while True:
        await asyncio.sleep(10)
        now = time.monotonic()
        for oid, info in list(pending_orders.items()):
            if (
                now - info.get("timestamp", now) > 60
                and info.get("status") == "PENDING_NEW"
            ):
                symbol = info.get("symbol", "")
                req = info.get("req")
                qty = float(getattr(req, "qty", 0)) if req is not None else 0.0
                filled_qty = 0.0
                try:
                    od = api.get_order_by_id(oid)
                    filled_qty = float(getattr(od, "filled_qty", 0))
                except Exception as exc:  # pragma: no cover - network
                    logger.warning("Failed to fetch order %s status: %s", oid, exc)
                if filled_qty >= qty and qty > 0:
                    logger.info(
                        "Order %s already filled with %s/%s, skipping resubmit.",
                        oid,
                        filled_qty,
                        qty,
                    )
                    pending_orders.pop(oid, None)
                    continue
                logger.warning(
                    "\N{WARNING SIGN} Order %s for %s stuck >60s. Cancelling and resubmitting.",
                    oid,
                    symbol,
                )
                try:
                    api.cancel_order_by_id(oid)
                except Exception as exc:  # pragma: no cover - network
                    logger.warning("Failed to cancel stuck order %s: %s", oid, exc)
                if req is not None:
                    try:
                        new_order = api.submit_order(order_data=req)
                        if getattr(new_order, "id", None):
                            pending_orders[str(new_order.id)] = {
                                "symbol": symbol,
                                "req": req,
                                "timestamp": time.monotonic(),
                                "status": "PENDING_NEW",
                            }
                            logger.info("Resubmitted order %s as %s", oid, new_order.id)
                    except Exception as exc:  # pragma: no cover - network
                        logger.exception("Error resubmitting order %s: %s", oid, exc)
                pending_orders.pop(oid, None)


async def start_trade_updates_stream(
    api_key: str, secret_key: str, api, state=None, *, paper: bool = True
) -> None:
    """Start Alpaca trade updates stream and stuck order monitor."""
    stream = TradingStream(api_key, secret_key, paper=paper)

    async def handle_async_trade_update(ev):
        asyncio.create_task(
            handle_trade_update(ev, state)
        )  # AI-AGENT-REF: schedule coroutine

    stream.subscribe_trade_updates(handle_async_trade_update)
    logger.info("\u2705 Subscribed to Alpaca trade updates stream.")
    asyncio.create_task(check_stuck_orders(api))
    asyncio.create_task(_partial_fill_summary_loop())
    # Avoid nested asyncio.run inside TradingStream.run()
    await stream._run_forever()
