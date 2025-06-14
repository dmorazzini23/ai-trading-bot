import os
import csv
import math
import time
import random
import logging
import logging.handlers
import warnings
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception_type,
)
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Optional
from typing import Tuple

# Updated Alpaca SDK imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Order
from alpaca.common.exceptions import APIError
from alpaca.data.models import Quote
from alpaca.data.requests import StockLatestQuoteRequest

from alpaca_api import submit_order
from slippage import monitor_slippage
from audit import log_trade
from config import SHADOW_MODE

__all__ = ["ExecutionEngine", "log_order"]


class ExecutionEngine:
    """
    Template for centralized trade execution, order handling, and slippage tracking.
    """

    def __init__(self, broker_api=None):
        self.broker_api = broker_api

    def place_order(self, symbol, qty, side, **kwargs):
        """
        Place an order via broker API.
        Args:
            symbol (str): Ticker symbol.
            qty (float): Quantity to trade.
            side (str): "buy" or "sell".
        Returns:
            dict: Order confirmation/response.
        """
        # TODO: Implement actual order placement logic here
        return {"symbol": symbol, "qty": qty, "side": side, "status": "stubbed"}


def log_order(order):
    """
    Logging or audit hook for every executed order. Extend for audit, compliance, and event tracking.
    Args:
        order (dict): The order or trade object to log.
    """
    # TODO: Extend with persistent logging, audit trails, etc.
    pass


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

    def __init__(
        self, ctx: Any, *, slippage_total=None, slippage_count=None, orders_total=None
    ) -> None:
        self.ctx = ctx
        # Trading client from the new Alpaca SDK
        self.api: TradingClient = ctx.api
        log_path = os.path.join(os.path.dirname(__file__), "logs", "execution.log")
        self.logger = logging.getLogger("execution")
        self.logger.setLevel(logging.INFO)
        try:
            handler = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=5_000_000, backupCount=2
            )
            self.logger.addHandler(handler)
        except Exception as e:
            self.logger.error(f"Failed to set up log handler {log_path}: {e}")
        self.slippage_path = os.path.join(
            os.path.dirname(__file__), "logs", "slippage.csv"
        )
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
                self.logger.error(
                    f"Failed to create slippage log {self.slippage_path}: {e}"
                )
        self.slippage_total = slippage_total
        self.slippage_count = slippage_count
        self.orders_total = orders_total

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
            # DataFrame missing or empty; return safe default
            self.logger.warning(f"ADV data unavailable for {symbol}")
            return 0.0
        if len(df) < 20:
            # Not enough rows for a 20 day average
            self.logger.warning(
                f"Not enough rows for ADV calculation of {symbol}: got {len(df)}"
            )
            return 0.0
        return float(df["volume"].tail(20).mean())

    def _minute_stats(self, symbol: str) -> Tuple[float, float, float]:
        df = self.ctx.data_fetcher.get_minute_df(self.ctx, symbol)
        if (
            df is None
            or df.empty
            or "volume" not in df.columns
            or "close" not in df.columns
        ):
            # Missing data means we can't compute stats safely
            self.logger.warning(f"Minute data unavailable for {symbol}")
            return 0.0, 0.0, 0.0
        if len(df) < 5:
            # Require at least 5 rows for momentum/average calculations
            self.logger.warning(
                f"Not enough rows for minute stats of {symbol}: got {len(df)}"
            )
            return 0.0, 0.0, 0.0
        vol = float(df["volume"].iloc[-1])
        avg1m = float(df["volume"].tail(5).mean())
        momentum = float(df["close"].iloc[-1] - df["close"].iloc[-5])
        return vol, avg1m, momentum

    def _prepare_order(
        self, symbol: str, side: str, qty: int
    ) -> Tuple[object, Optional[float]]:
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
        self, symbol: str, expected: Optional[float], actual: float
    ) -> None:
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

    def execute_order(
        self, symbol: str, qty: int, side: str, asset_class: str = "equity"
    ) -> Optional[Order]:
        """Execute an order for the given asset class."""
        remaining = int(math.floor(qty))
        last_order = None
        api = self.api
        if asset_class == "crypto" and hasattr(self.ctx, "crypto_api"):
            api = self.ctx.crypto_api
        elif asset_class == "forex" and hasattr(self.ctx, "forex_api"):
            api = self.ctx.forex_api
        elif asset_class == "commodity" and hasattr(self.ctx, "commodity_api"):
            api = self.ctx.commodity_api
        while remaining > 0:
            order_req, expected_price = self._prepare_order(symbol, side, remaining)
            slice_qty = getattr(order_req, "qty", remaining)
            try:
                acct = api.get_account()
            except Exception as e:
                # Log unexpected account retrieval errors
                self.logger.error(f"Error fetching account information: {e}")
                acct = None
            if side.lower() == "buy" and acct:
                need = slice_qty * (expected_price or 0)
                if float(acct.cash) < need:
                    self.logger.error(
                        f"Insufficient buying power for {symbol}: need {need}, have {acct.cash}"
                    )
                    break
            if side.lower() == "sell":
                try:
                    api.get_position(symbol)
                except Exception as e:
                    # Log the exception to aid debugging of sell attempts
                    self.logger.error(f"No position to sell for {symbol}: {e}")
                    break
            start = time.monotonic()
            order = None
            for attempt in range(2):
                try:
                    self.logger.info(
                        "ORDER_SUBMIT",
                        extra={
                            "symbol": symbol,
                            "side": side,
                            "qty": slice_qty,
                            "type": order_req.__class__.__name__,
                        },
                    )
                    try:
                        order = submit_order(api, order_req, self.logger)
                        self.logger.info(
                            f"Order submit response for {symbol}: {order}"
                        )
                        if not getattr(order, "id", None) and not SHADOW_MODE:
                            self.logger.error(f"Order failed for {symbol}: {order}")
                    except Exception as e:
                        self.logger.error(
                            f"Order submission failed for {symbol}: {e}",
                            exc_info=True,
                        )
                        break
                    break
                except (APIError, TimeoutError) as e:
                    time.sleep(1)
                    if attempt == 1:
                        self.logger.warning(f"submit_order failed for {symbol}: {e}")
                        return last_order
                except APIError as e:
                    self.logger.warning(f"APIError placing order for {symbol}: {e}")
                    return last_order
                except Exception as e:
                    self.logger.exception(
                        f"Unexpected error placing order for {symbol}: {e}"
                    )
                    return last_order
            if order is None:
                break
            status = getattr(order, "status", "")
            if status == "rejected":
                self.logger.error(
                    "ORDER_REJECTED",
                    extra={
                        "symbol": symbol,
                        "reason": getattr(order, "reject_reason", ""),
                    },
                )
                break
            if status == "canceled":
                self.logger.error(
                    "ORDER_CANCELED",
                    extra={"symbol": symbol},
                )
                break
            fill_price = float(
                getattr(order, "filled_avg_price", expected_price or 0) or 0
            )
            latency = (time.monotonic() - start) * 1000.0
            self._log_slippage(symbol, expected_price, fill_price)
            if status == "filled":
                self.logger.info(
                    "ORDER_FILLED",
                    extra={
                        "symbol": symbol,
                        "order_id": getattr(order, "id", ""),
                        "latency_ms": latency,
                        "price": fill_price,
                    },
                )
                log_trade(symbol, side, slice_qty, fill_price, status, "SHADOW" if SHADOW_MODE else "LIVE")
            else:
                self.logger.error(
                    "ORDER_STATUS", extra={"symbol": symbol, "status": status}
                )
            if self.orders_total is not None:
                self.orders_total.inc()
            last_order = order
            remaining -= slice_qty
            if remaining > 0:
                time.sleep(random.uniform(0.05, 0.15))
        return last_order
