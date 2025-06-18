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
except Exception:  # pragma: no cover - allow running without Alpaca SDK
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
except Exception:  # pragma: no cover - allow running without Alpaca API config

    def submit_order(*args, **kwargs):
        raise RuntimeError("Alpaca API unavailable")


from audit import log_trade
from slippage import monitor_slippage

SHADOW_MODE = os.getenv("SHADOW_MODE", "0") == "1"


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


class ExecutionEngine:
    """Institutional-grade execution engine for dynamic order routing."""

    def __init__(self, ctx: Any, *, slippage_total=None, slippage_count=None, orders_total=None) -> None:
        self.ctx = ctx
        # Trading client from the new Alpaca SDK
        self.api: TradingClient = ctx.api
        log_path = os.path.join(os.path.dirname(__file__), "logs", "execution.log")
        self.logger = logging.getLogger("execution")
        self.logger.setLevel(logging.INFO)
        try:
            handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=2)
            self.logger.addHandler(handler)
        except Exception as e:
            self.logger.error(f"Failed to set up log handler {log_path}: {e}")
        self.slippage_path = os.path.join(os.path.dirname(__file__), "logs", "slippage.csv")
        if not os.path.exists(self.slippage_path):
            # Protect file creation in case the logs directory is unwritable
            try:
                with open(self.slippage_path, "w", newline="") as f:
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
            except OSError as e:
                self.logger.error(f"Failed to create slippage log {self.slippage_path}: {e}")
        self.slippage_total = slippage_total
        self.slippage_count = slippage_count
        self.orders_total = orders_total

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
        except Exception as e:  # pragma: no cover - api may be stubbed
            self.logger.error(f"Error fetching account information: {e}")
            return False
        need = qty * price
        if float(getattr(acct, "cash", 0)) < need:
            self.logger.error(f"Insufficient buying power: need {need}, have {acct.cash}")
            return False
        return True

    def _available_qty(self, api: TradingClient, symbol: str) -> float:
        try:
            pos = api.get_position(symbol)
            return float(getattr(pos, "qty", 0))
        except Exception as e:  # pragma: no cover - position may not exist
            self.logger.error(f"No position for {symbol}: {e}")
            return 0.0

    def _can_sell(self, api: TradingClient, symbol: str, qty: int) -> bool:
        avail = self._available_qty(api, symbol)
        if avail < qty:
            self.logger.error(
                f"Insufficient qty for {symbol}: have {avail}, requested {qty}"
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
        except APIError as e:
            self.logger.warning(f"_latest_quote APIError for {symbol}: {e}")
            raise

    def _adv_volume(self, symbol: str) -> Optional[float]:
        df = self.ctx.data_fetcher.get_daily_df(self.ctx, symbol)
        if df is None or df.empty or "volume" not in df.columns:
            self.logger.warning(f"ADV data unavailable for {symbol}")
            return 0.0
        if len(df) < 20:
            self.logger.warning(
                f"Not enough rows for ADV calculation of {symbol}: got {len(df)}"
            )
            return 0.0
        vol = df["volume"].tail(20)
        if vol.isna().any() or np.isinf(vol).any():
            self.logger.warning(f"Invalid volume data for {symbol} during ADV calc")
            return 0.0
        return float(vol.mean())

    def _minute_stats(self, symbol: str) -> Tuple[float, float, float]:
        df = self.ctx.data_fetcher.get_minute_df(self.ctx, symbol)
        if df is None or df.empty or "volume" not in df.columns or "close" not in df.columns:
            # Missing data means we can't compute stats safely
            self.logger.warning(f"Minute data unavailable for {symbol}")
            return 0.0, 0.0, 0.0
        if len(df) < 5:
            # Require at least 5 rows for momentum/average calculations
            self.logger.warning(f"Not enough rows for minute stats of {symbol}: got {len(df)}")
            return 0.0, 0.0, 0.0
        vol = float(df["volume"].iloc[-1])
        avg1m = float(df["volume"].tail(5).mean())
        momentum = float(df["close"].iloc[-1] - df["close"].iloc[-5])
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

    def _log_slippage(self, symbol: str, expected: Optional[float], actual: float) -> None:
        slip = ((actual - expected) * 100) if expected else 0.0
        try:
            # File I/O may fail; handle gracefully
            with open(self.slippage_path, "a", newline="") as f:
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
        except OSError as e:
            self.logger.error(f"Failed to write slippage log: {e}")
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
                self.logger.info(f"Order submit response for {symbol}: {order}")
                if not getattr(order, "id", None) and not SHADOW_MODE:
                    self.logger.error(f"Order failed for {symbol}: {order}")
                return order
            except (APIError, TimeoutError) as e:
                sleep = 1 * (attempt + 1)
                time.sleep(sleep)
                if attempt == 2:
                    self.logger.warning(
                        f"submit_order failed for {symbol} after retries: {e}"
                    )
                    return None
            except APIError as e:
                self.logger.warning(f"APIError placing order for {symbol}: {e}")
                return None
            except Exception as e:
                self.logger.exception(f"Unexpected error placing order for {symbol}: {e}")
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
    ) -> bool:
        status = getattr(order, "status", "")
        order_id = getattr(order, "id", "")
        if not self._check_order_active(symbol, order, status, order_id):
            return False
        fill_price = float(
            getattr(order, "filled_avg_price", expected_price or 0) or 0
        )
        latency = (time.monotonic() - start_time) * 1000.0
        self._log_slippage(symbol, expected_price, fill_price)
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
            log_trade(
                symbol,
                side,
                slice_qty,
                fill_price,
                status,
                "SHADOW" if SHADOW_MODE else "LIVE",
            )
        elif status == "partially_filled":
            self.logger.info(
                "ORDER_PARTIAL",
                extra={
                    "symbol": symbol,
                    "order_id": order_id,
                    "filled_qty": getattr(order, "filled_qty", 0),
                },
            )
        else:
            self.logger.error(
                "ORDER_STATUS",
                extra={"symbol": symbol, "order_id": order_id, "status": status},
            )
        if self.orders_total is not None:
            self.orders_total.inc()
        return True

    def execute_order(self, symbol: str, qty: int, side: str, asset_class: str = "equity") -> Optional[Order]:
        """Execute an order for the given asset class."""
        remaining = int(math.floor(qty))
        last_order = None
        api = self._select_api(asset_class)
        if side.lower() == "sell":
            avail = self._available_qty(api, symbol)
            if avail <= 0:
                self.logger.error(f"No position to sell for {symbol}")
                return None
            if remaining > avail:
                self.logger.warning(
                    f"Requested {remaining} but only {avail} available for {symbol}; adjusting"
                )
                remaining = int(avail)
        while remaining > 0:
            order_req, expected_price = self._prepare_order(symbol, side, remaining)
            slice_qty = getattr(order_req, "qty", remaining)
            if side.lower() == "buy" and not self._has_buy_power(api, slice_qty, expected_price):
                break
            if side.lower() == "sell" and not self._can_sell(api, symbol, slice_qty):
                break
            start = time.monotonic()
            order = self._submit_with_retry(api, order_req, symbol, side, slice_qty)
            if order is None:
                break
            if not self._handle_order_result(symbol, side, order, expected_price, slice_qty, start):
                break
            last_order = order
            remaining -= slice_qty
            if remaining > 0:
                time.sleep(random.uniform(0.05, 0.15))
        return last_order


__all__ = ["ExecutionEngine", "log_order"]
