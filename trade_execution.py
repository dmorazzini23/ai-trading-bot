"""Order execution engine with Alpaca integration and slippage logging."""

import csv
import logging
import logging.handlers
import math
import os
import random
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential, wait_random)

# Updated Alpaca SDK imports
try:
    from alpaca.common.exceptions import APIError
    from alpaca.data.models import Quote
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.models import Order
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
except Exception:  # TODO: narrow exception type
    TradingClient = object

    class MarketOrderRequest(dict):
        def __init__(self, symbol, qty, side, time_in_force):
            super().__init__(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=time_in_force,
            )

    class LimitOrderRequest(dict):
        def __init__(self, symbol, qty, side, time_in_force, limit_price):
            super().__init__(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=time_in_force,
                limit_price=limit_price,
            )

    class OrderSide:
        BUY = "buy"
        SELL = "sell"

    class TimeInForce:
        DAY = "day"

    class Order(dict):
        pass

    class APIError(Exception):
        pass

    class Quote:
        bid_price = 0
        ask_price = 0

    class StockLatestQuoteRequest:
        def __init__(self, symbol_or_symbols):
            self.symbols = symbol_or_symbols


try:
    from alpaca_api import submit_order
except Exception:  # TODO: narrow exception type

    def submit_order(*args, **kwargs):
        raise RuntimeError("Alpaca API unavailable")


from audit import log_trade as audit_log_trade
from slippage import monitor_slippage
from utils import get_phase_logger

SHADOW_MODE = os.getenv("SHADOW_MODE", "0") == "1"


def log_trade(symbol, quantity, price, order_id, filled_qty, timestamp):
    """Log basic trade execution details."""  # AI-AGENT-REF: logging fix
    import logging
    logging.info(
        f"[TRADE_LOG] {symbol} qty={quantity} price={price} order_id={order_id} "
        f"filled_qty={filled_qty} time={timestamp}"
    )


def log_order(order, status=None, extra=None):
    """Log the result of an order execution.

    Parameters
    ----------
    order : object
        The order or trade object to log.
    status : Optional[str]
        Execution status. Unused in the stub.
    extra : Optional[dict]
        Additional logging context.
    """
    # TODO: Extend with persistent logging, audit trails, etc.


def place_order(symbol: str, qty: int, side: str):
    """Convenience wrapper to place a basic market order."""
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    order = submit_order(None, req)
    log_order(order)
    return order


warnings.filterwarnings("ignore", category=FutureWarning)


def calculate_adv(df: pd.DataFrame, symbol: str, window: int = 30) -> Optional[float]:
    """Return the average daily volume for ``symbol`` using ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a ``volume`` column.
    symbol : str
        Trading symbol for logging context.
    window : int, optional
        Number of trailing days to average, by default ``30``.

    Returns
    -------
    Optional[float]
        The calculated ADV or ``None`` if not enough data.
    """

    if df is None or df.empty or "volume" not in df.columns:
        logging.warning("ADV data unavailable for %s", symbol)
        return None
    if len(df) < 20:
        logging.debug("ADV skipped for %s, only %s rows", symbol, len(df))
        return None
    vol = df["volume"].tail(window)
    if vol.isna().any() or np.isinf(vol).any():
        logging.warning("Invalid volume data for %s during ADV calc", symbol)
        return None
    return float(vol.mean())


class ExecutionEngine:
    """Institutional-grade execution engine for dynamic order routing."""

    def __init__(self, ctx: Any, *, slippage_total=None, slippage_count=None, orders_total=None) -> None:
        self.ctx = ctx
        # Trading client from the new Alpaca SDK
        self.api: TradingClient = ctx.api
        # Use phase logger for execution context
        self.logger = get_phase_logger(__name__, "ORDER_EXEC")
        self.slippage_path = os.path.join(os.path.dirname(__file__), "logs", "slippage.csv")
        if not os.path.exists(self.slippage_path):
            # Protect file creation in case the logs directory is unwritable
            try:
                with open(self.slippage_path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [
                            "timestamp",
                            "symbol",
                            "expected",
                            "actual",
                            "slippage_cents",
                            "band",
                        ]
                    )
            except OSError as exc:
                self.logger.error(
                    "Failed to create slippage log %s: %s",
                    self.slippage_path,
                    exc,
                )
        self.slippage_total = slippage_total
        self.slippage_count = slippage_count
        self.orders_total = orders_total
        self._slippage_checks: dict[str, int] = {}

    # --- helper methods -------------------------------------------------

    def _select_api(self, asset_class: str):
        api = self.api
        if asset_class == "crypto" and hasattr(self.ctx, "crypto_api"):
            api = self.ctx.crypto_api
        elif asset_class == "forex" and hasattr(self.ctx, "forex_api"):
            api = self.ctx.forex_api
        elif asset_class == "commodity" and hasattr(self.ctx, "commodity_api"):
            api = self.ctx.commodity_api
        return api

    def _has_buy_power(self, api: TradingClient, qty: int, price: Optional[float]) -> bool:
        if price is None:
            return True
        try:
            acct = api.get_account()
        except Exception as exc:  # TODO: narrow exception type
            self.logger.error("Error fetching account information: %s", exc)
            return False
        need = qty * price
        if float(getattr(acct, "cash", 0)) < need:
            self.logger.error(
                "Insufficient buying power: need %s, have %s",
                need,
                getattr(acct, "cash", 0),
            )
            return False
        return True

    def _available_qty(self, api: TradingClient, symbol: str) -> float:
        """Return current position quantity for ``symbol`` with retries."""
        try:
            if hasattr(api, "get_all_positions"):
                for _ in range(3):
                    positions = api.get_all_positions()
                    for pos in positions:
                        if getattr(pos, "symbol", "") == symbol:
                            return float(getattr(pos, "qty", 0))
                    time.sleep(1)
            if hasattr(api, "get_open_position"):
                pos = api.get_open_position(symbol)
                return float(getattr(pos, "qty", 0))
            if hasattr(api, "get_position"):
                pos = api.get_position(symbol)
                return float(getattr(pos, "qty", 0))
            acct = api.get_account()
            for p in getattr(acct, "positions", []):
                if getattr(p, "symbol", "") == symbol:
                    return float(getattr(p, "qty", 0))
        except Exception as exc:  # pragma: no cover - network or API errors
            self.logger.error("No position for %s: %s", symbol, exc)
        return 0.0

    def _can_sell(self, api: TradingClient, symbol: str, qty: int) -> bool:
        avail = self._available_qty(api, symbol)
        if avail < qty:
            self.logger.warning(
                "Insufficient qty for %s: have %s, requested %s",
                symbol,
                avail,
                qty,
            )
            return False
        return True

    def submit_order(self, order_request):
        """Placeholder for order submission logic."""
        return submit_order(self.api, order_request, self.logger)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5) + wait_random(0.1, 0.5),
        retry=retry_if_exception_type(Exception),
    )
    def _latest_quote(self, symbol: str) -> Tuple[float, float]:
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            q: Quote = self.ctx.data_client.get_stock_latest_quote(req)
            bid = float(getattr(q, "bid_price", 0) or 0)
            ask = float(getattr(q, "ask_price", 0) or 0)
            return bid, ask
        except APIError as exc:
            self.logger.warning("_latest_quote APIError for %s: %s", symbol, exc)
            raise

    def _adv_volume(self, symbol: str, window: int = 30) -> Optional[float]:
        df = self.ctx.data_fetcher.get_daily_df(self.ctx, symbol)
        return calculate_adv(df, symbol, window)

    def _minute_stats(self, symbol: str) -> Tuple[float, float, float]:
        df = self.ctx.data_fetcher.get_minute_df(self.ctx, symbol)
        if df is None or df.empty or "volume" not in df.columns or "close" not in df.columns:
            # Missing data means we can't compute stats safely
            self.logger.warning("Minute data unavailable for %s", symbol)
            return 0.0, 0.0, 0.0
        if len(df) < 5:
            # Require at least 5 rows for momentum/average calculations
            self.logger.warning(
                "Not enough rows for minute stats of %s: got %s",
                symbol,
                len(df),
            )
            return 0.0, 0.0, 0.0
        vol = float(df["volume"].iloc[-1])
        avg1m = float(df["volume"].tail(5).mean())
        last_valid_close = df["close"].dropna()
        if last_valid_close.empty:
            self.logger.critical(f"All NaNs in close column for {symbol}")
            return 0.0, 0.0, 0.0
        last_close = last_valid_close.iloc[-1]
        prev_idx = -5 if len(last_valid_close) >= 5 else 0
        momentum = float(last_close - last_valid_close.iloc[prev_idx])
        return vol, avg1m, momentum

    def _prepare_order(self, symbol: str, side: str, qty: int) -> Tuple[object, Optional[float]]:
        if qty <= 0:
            raise ValueError("qty must be positive")

        bid, ask = self._latest_quote(symbol)
        spread = (ask - bid) if ask and bid else 0.0
        mid = (ask + bid) / 2 if ask and bid else None
        vol, avg1m, momentum = self._minute_stats(symbol)
        adv = self._adv_volume(symbol)

        adv_pct = getattr(self.ctx, "adv_target_pct", 0.002)
        max_adv = adv * adv_pct if adv else qty
        max_slice = int(vol * 0.1) if vol > 0 else qty
        slice_qty = max(1, min(qty, int(min(max_slice, max_adv))))

        expected = None
        order_request: object
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        aggressive = momentum > 0 if side == "buy" else momentum < 0

        if spread > 0.05 and mid:
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=slice_qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(mid, 2),
            )
            expected = round(mid, 2)
        elif aggressive and spread < 0.02:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=slice_qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            expected = ask if side == "buy" else bid
        elif mid:
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=slice_qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(mid, 2),
            )
            expected = round(mid, 2)
        else:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=slice_qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            expected = ask if side == "buy" else bid

        return order_request, expected

    def _log_slippage(
        self,
        symbol: str,
        expected: Optional[float],
        actual: float,
        *,
        order_id: str | None = None,
    ) -> None:
        """Log slippage after at least three checks."""
        key = order_id or symbol
        count = self._slippage_checks.get(key, 0) + 1
        self._slippage_checks[key] = count
        slip = ((actual - expected) * 100) if expected else 0.0
        if count < 3:
            self.logger.debug("waiting on fill for %s", symbol)
            return
        try:
            # File I/O may fail; handle gracefully
            with open(self.slippage_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        datetime.now(timezone.utc).isoformat(),
                        symbol,
                        expected,
                        actual,
                        slip,
                        getattr(self.ctx, "capital_band", "small"),
                    ]
                )
        except OSError as exc:
            self.logger.error("Failed to write slippage log: %s", exc)
        if self.slippage_total is not None:
            self.slippage_total.inc(abs(slip))
        if self.slippage_count is not None:
            self.slippage_count.inc()
        self.logger.info(
            "SLIPPAGE",
            extra={
                "symbol": symbol,
                "expected": expected,
                "actual": actual,
                "slippage_cents": slip,
                "band": getattr(self.ctx, "capital_band", "small"),
            },
        )
        monitor_slippage(expected, actual, symbol)

    def _check_order_active(self, symbol: str, order: Order, status: str, order_id: str) -> bool:
        """Return ``True`` if order status indicates the order is active."""
        if status == "rejected":
            self.logger.error(
                "ORDER_REJECTED",
                extra={
                    "symbol": symbol,
                    "order_id": order_id,
                    "reason": getattr(order, "reject_reason", ""),
                },
            )
            return False
        if status == "canceled":
            self.logger.error(
                "ORDER_CANCELED", extra={"symbol": symbol, "order_id": order_id}
            )
            return False
        return True

    def _submit_with_retry(
        self,
        api: TradingClient,
        order_req: object,
        symbol: str,
        side: str,
        slice_qty: int,
    ) -> Optional[Order]:
        """Submit ``order_req`` with simple retry handling."""
        self.logger.info(
            "ORDER_SUBMIT",
            extra={
                "symbol": symbol,
                "side": side,
                "qty": slice_qty,
                "type": order_req.__class__.__name__,
            },
        )
        for attempt in range(3):
            try:
                order = submit_order(api, order_req, self.logger)
                self.logger.info("Order submit response for %s: %s", symbol, order)
                if not getattr(order, "id", None) and not SHADOW_MODE:
                    self.logger.error("Order failed for %s: %s", symbol, order)
                return order
            except (APIError, TimeoutError) as e:
                sleep = 1 * (attempt + 1)
                time.sleep(sleep)
                if attempt == 2:
                    self.logger.warning(
                        "submit_order failed for %s after retries: %s",
                        symbol,
                        e,
                    )
                    return None
            except Exception as exc:  # TODO: narrow exception type
                self.logger.exception(
                    "Unexpected error placing order for %s: %s",
                    symbol,
                    exc,
                )
                return None
        return None

    def _handle_order_result(
        self,
        symbol: str,
        side: str,
        order: Order,
        expected_price: Optional[float],
        slice_qty: int,
        start_time: float,
    ) -> int:
        status = getattr(order, "status", "")
        order_id = getattr(order, "id", "")
        if status in ("new", "pending_new"):
            time.sleep(3)
            try:
                refreshed = self.api.get_order_by_id(order_id)
                status = getattr(refreshed, "status", status)
                order = refreshed
            except Exception as exc:  # pragma: no cover - network issues
                self.logger.debug("Order refresh failed for %s: %s", order_id, exc)
        if not self._check_order_active(symbol, order, status, order_id):
            return 0
        fill_price = float(
            getattr(order, "filled_avg_price", expected_price or 0) or 0
        )
        latency = (time.monotonic() - start_time)
        if status in ("new", "pending_new"):
            self.logger.info(
                "ORDER_PENDING", extra={"symbol": symbol, "order_id": order_id, "wait_s": latency}
            )
            return 0
        self._log_slippage(symbol, expected_price, fill_price, order_id=order_id)
        latency *= 1000.0
        filled_qty = 0
        if status == "filled":
            self.logger.info(
                "ORDER_FILLED",
                extra={
                    "symbol": symbol,
                    "order_id": order_id,
                    "latency_ms": latency,
                    "price": fill_price,
                },
            )
            audit_log_trade(
                symbol,
                side,
                slice_qty,
                fill_price,
                status,
                "SHADOW" if SHADOW_MODE else "LIVE",
            )
            log_trade(
                symbol,
                slice_qty,
                fill_price,
                order_id,
                slice_qty,
                datetime.now(timezone.utc).isoformat(),
            )
            filled_qty = slice_qty
        elif status == "partially_filled":
            self.logger.info(
                "ORDER_PARTIAL",
                extra={
                    "symbol": symbol,
                    "order_id": order_id,
                    "filled_qty": getattr(order, "filled_qty", 0),
                },
            )
            filled_qty = int(getattr(order, "filled_qty", 0) or 0)
        elif status in ("pending_new", "new"):
            self.logger.info("Order status for %s: %s", symbol, status)
        elif status in ("rejected", "failed"):
            self.logger.error("Order failed for %s: %s", symbol, status)
        else:
            self.logger.error(
                "ORDER_STATUS",
                extra={"symbol": symbol, "order_id": order_id, "status": status},
            )
        if self.orders_total is not None:
            self.orders_total.inc()
        return filled_qty

    def execute_order(self, symbol: str, qty: int, side: str, asset_class: str = "equity") -> Optional[Order]:
        """Execute an order for the given asset class."""
        remaining = int(round(qty))
        last_order = None
        api = self._select_api(asset_class)
        if side.lower() == "sell":
            avail = self._available_qty(api, symbol)
            if avail <= 0:
                self.logger.error("No position to sell for %s", symbol)
                return None
            if remaining > avail:
                self.logger.warning(
                    "Requested %s but only %s available for %s; adjusting",
                    remaining,
                    avail,
                    symbol,
                )
                remaining = int(round(avail))
        while remaining > 0:
            if side.lower() == "sell":
                avail = self._available_qty(api, symbol)
                if avail < remaining:
                    self.logger.warning(
                        "Available qty reduced for %s: %s -> %s",
                        symbol,
                        remaining,
                        avail,
                    )
                    remaining = int(round(avail))
                    if remaining <= 0:
                        break
            order_req, expected_price = self._prepare_order(
                symbol, side, remaining
            )
            slice_qty = getattr(order_req, "qty", remaining)
            if side.lower() == "sell" and slice_qty > self._available_qty(api, symbol):
                slice_qty = int(round(self._available_qty(api, symbol)))
                if isinstance(order_req, dict):
                    order_req["qty"] = slice_qty
                else:
                    setattr(order_req, "qty", slice_qty)
            if side.lower() == "buy" and not self._has_buy_power(
                api, slice_qty, expected_price
            ):
                break
            if side.lower() == "sell" and not self._can_sell(
                api, symbol, slice_qty
            ):
                break
            start = time.monotonic()
            order = self._submit_with_retry(
                api, order_req, symbol, side, slice_qty
            )
            if order is None:
                break
            filled = self._handle_order_result(
                symbol, side, order, expected_price, slice_qty, start
            )
            if filled <= 0:
                break
            last_order = order
            remaining -= filled
            if remaining > 0:
                time.sleep(random.uniform(0.05, 0.15))
        return last_order


__all__ = ["ExecutionEngine", "log_order"]
