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
from enum import Enum
from uuid import uuid4

import numpy as np
import pandas as pd
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential, wait_random)

# AI-AGENT-REF: track recent buy timestamps to avoid immediate re-checks
recent_buys: dict[str, float] = {}

# Updated Alpaca SDK imports
try:
    from alpaca.common.exceptions import APIError
    from alpaca.data.models import Quote
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.models import Order
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
except ImportError:
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
except ImportError:

    def submit_order(*args, **kwargs):
        raise RuntimeError("Alpaca API unavailable")


from audit import log_trade as audit_log_trade, log_json_audit
from slippage import monitor_slippage
from utils import get_phase_logger
import config
from collections import deque


class OrderClass(str, Enum):
    """Execution order classification."""

    NORMAL = "normal"
    INITIAL_REBALANCE = "initial_rebalance"

SHADOW_MODE = os.getenv("SHADOW_MODE", "0") == "1"

# AI-AGENT-REF: aggregate partial fills across orders
_partial_fills: dict[str, dict] = {}


def generate_client_order_id(
    symbol: str, side: str, order_class: OrderClass
) -> str:
    """Return a deterministic client order id for retries."""
    seed = f"{symbol}-{side}-{order_class.name}-{time.time_ns()}"
    return f"{symbol}-{hash(seed) & 0xFFFF_FFFF:x}"


def log_trade(
    symbol: str,
    qty: int,
    side: str,
    fill_price: float,
    timestamp: str,
    extra_info: Any | None = None,
) -> None:
    """Log basic trade execution details."""
    logger = logging.getLogger(__name__)
    logger.info(
        "TRADE_EXECUTED",
        extra={
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "price": fill_price,
            "timestamp": timestamp,
            "extra": extra_info,
        },
    )


def log_order(order, status=None, extra=None):
    """Log the result of an order execution and persist to file."""
    path = os.path.join(os.path.dirname(__file__), "logs", "orders.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now(timezone.utc).isoformat(),
                    getattr(order, "symbol", ""),
                    getattr(order, "qty", ""),
                    status or getattr(order, "status", ""),
                    extra or {},
                ]
            )
    except OSError as exc:
        logging.getLogger(__name__).error("Failed to persist order log: %s", exc)


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


def handle_partial_fill(order_id: str, symbol: str, qty: int, price: float) -> None:
    """Accumulate partial fill data for later aggregation."""
    if order_id not in _partial_fills:
        _partial_fills[order_id] = {"symbol": symbol, "qty": 0, "price_sum": 0.0, "fills": 0}
    data = _partial_fills[order_id]
    data["qty"] += qty
    data["price_sum"] += price * qty
    data["fills"] += 1
    logging.getLogger(__name__).debug(
        "Partial fill accumulating: %s total_qty=%s", symbol, data["qty"]
    )


def handle_full_fill(order_id: str) -> None:
    """Log aggregated metrics for a fully filled order."""
    data = _partial_fills.pop(order_id, None)
    if data:
        avg_price = data["price_sum"] / data["qty"]
        logging.getLogger(__name__).info(
            "ORDER_FILLED_AGGREGATED | %s qty=%s avg_price=%.2f",
            data["symbol"],
            data["qty"],
            avg_price,
        )


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
        # AI-AGENT-REF: track fill fragmentation and volatility regime
        self.fill_history: deque[int] = deque(maxlen=config.PARTIAL_FILL_LOOKBACK)
        self.partial_flags: deque[bool] = deque(maxlen=config.PARTIAL_FILL_LOOKBACK)
        self.adaptive_multiplier = 1.0
        self.vol_history: deque[float] = deque(maxlen=50)
        self.baseline_vol: float | None = None
        self.high_vol_regime = False
        self._partial_buffer: dict[str, dict] = {}
        # AI-AGENT-REF: track orders within a cycle to avoid duplicates
        self._cycle_orders: set[str] = set()
        self._cycle_symbols: set[str] = set()

    # --- helper methods -------------------------------------------------

    def start_cycle(self) -> None:
        """Reset duplicate order tracking for a new run cycle."""
        self._cycle_orders.clear()
        self._cycle_symbols.clear()

    def _select_api(self, asset_class: str):
        api = self.api
        if asset_class == "crypto" and hasattr(self.ctx, "crypto_api"):
            api = self.ctx.crypto_api
        elif asset_class == "forex" and hasattr(self.ctx, "forex_api"):
            api = self.ctx.forex_api
        elif asset_class == "commodity" and hasattr(self.ctx, "commodity_api"):
            api = self.ctx.commodity_api
        return api

    def _is_duplicate_order(self, order) -> bool:
        """Return True if ``order`` is a duplicate within this cycle."""
        order_cls = getattr(order, "order_class", OrderClass.NORMAL)
        if order_cls == OrderClass.INITIAL_REBALANCE:
            return False
        cid = getattr(order, "client_order_id", "")
        symbol = getattr(order, "symbol", "")
        return cid in self._cycle_orders or symbol in self._cycle_symbols

    def _has_buy_power(self, api: TradingClient, qty: int, price: Optional[float]) -> bool:
        if price is None:
            return True
        try:
            acct = api.get_account()
        except (APIError, RuntimeError, AttributeError) as exc:
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
                for attempt in range(3):
                    positions = api.get_all_positions()
                    self.logger.info("Raw Alpaca positions: %s", positions)
                    for pos in positions:
                        if getattr(pos, "symbol", "") == symbol:
                            return float(getattr(pos, "qty", 0))
                    time.sleep(1 + random.uniform(0.1, 0.3) * (attempt + 1))
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

        self._update_volatility(symbol)

        bid, ask = self._latest_quote(symbol)
        spread = (ask - bid) if ask and bid else 0.0
        mid = (ask + bid) / 2 if ask and bid else None
        vol, avg1m, momentum = self._minute_stats(symbol)
        adv = self._adv_volume(symbol)

        adv_pct = getattr(self.ctx, "adv_target_pct", 0.002)
        max_adv = adv * adv_pct if adv else qty
        max_slice = int(vol * 0.1) if vol > 0 else qty
        slice_qty = max(1, min(qty, int(min(max_slice, max_adv))))
        slice_qty = int(slice_qty * self.adaptive_multiplier)
        if self.high_vol_regime:
            slice_qty = max(1, int(slice_qty * 0.5))

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

        # AI-AGENT-REF: tag order requests by execution phase
        rebalance_ids = getattr(self.ctx, "rebalance_ids", {})
        is_rebalance = symbol in rebalance_ids and not getattr(self.ctx, "_rebalance_done", False)
        order_class = OrderClass.INITIAL_REBALANCE if is_rebalance else OrderClass.NORMAL
        setattr(order_request, "order_class", order_class)

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

    # --- adaptive helpers -------------------------------------------------

    def _record_fill_steps(self, steps: int) -> None:
        self.fill_history.append(steps)
        partial = steps > 1
        self.partial_flags.append(partial)
        if partial and steps > config.PARTIAL_FILL_FRAGMENT_THRESHOLD:
            self.logger.warning(
                "PARTIAL_FILL_FRAGMENTED", extra={"steps": steps}
            )
        if len(self.partial_flags) >= self.partial_flags.maxlen:
            frag_count = sum(1 for f in self.partial_flags if f)
            if frag_count > config.PARTIAL_FILL_FRAGMENT_THRESHOLD:
                self.logger.warning(
                    "HIGH_FRAGMENTATION", extra={"count": frag_count}
                )
                self.adaptive_multiplier = 1.0 - config.PARTIAL_FILL_REDUCTION_RATIO
            else:
                self.adaptive_multiplier = 1.0

    def _assess_liquidity(
        self, symbol: str, qty: int, *, attempted: bool = False
    ) -> tuple[int, bool]:
        bid, ask = self._latest_quote(symbol)
        spread = ask - bid if ask and bid else 0.0
        df_ticks = self.ctx.data_fetcher.get_minute_df(self.ctx, symbol)
        tick_range = 0.0
        if df_ticks is not None and not df_ticks.empty and "close" in df_ticks.columns:
            tick_range = float(df_ticks["close"].diff().abs().tail(5).max() or 0.0)
        if spread >= config.LIQUIDITY_SPREAD_THRESHOLD * 2 or tick_range >= config.LIQUIDITY_VOL_THRESHOLD * 2:
            reason = "spread" if spread >= config.LIQUIDITY_SPREAD_THRESHOLD * 2 else "volatility"
            if attempted:
                self.logger.info(
                    "LIQUIDITY_SKIP",
                    extra={"symbol": symbol, "spread": spread, "tick_range": tick_range, "reason": reason},
                )
                time.sleep(random.uniform(0.1, 0.3))
                self.logger.info(
                    "LIQUIDITY_SKIP_FINAL",
                    extra={"symbol": symbol, "spread": spread, "tick_range": tick_range, "reason": reason},
                )
                return 0, True
            self.logger.info(
                "LIQUIDITY_RETRY",
                extra={"symbol": symbol, "spread": spread, "tick_range": tick_range, "reason": reason},
            )
            time.sleep(random.uniform(0.1, 0.3))
            return max(1, int(qty * 0.5)), False
        if spread >= config.LIQUIDITY_SPREAD_THRESHOLD or tick_range >= config.LIQUIDITY_VOL_THRESHOLD:
            self.logger.info(
                "LIQUIDITY_REDUCE",
                extra={"symbol": symbol, "spread": spread, "tick_range": tick_range},
            )
            return max(1, int(qty * 0.5)), False
        return qty, False

    def _update_volatility(self, symbol: str) -> None:
        df = self.ctx.data_fetcher.get_minute_df(self.ctx, symbol)
        if df is None or df.empty or "close" not in df.columns:
            return
        returns = df["close"].pct_change().dropna()
        if returns.empty:
            return
        std = float(returns.tail(20).std())
        self.vol_history.append(std)
        if self.baseline_vol is None and len(self.vol_history) >= 20:
            self.baseline_vol = sum(self.vol_history) / len(self.vol_history)
        if self.baseline_vol:
            self.high_vol_regime = std > config.VOL_REGIME_MULTIPLIER * self.baseline_vol

    def _check_exposure_cap(self, asset_class: str, qty: int, price: float, symbol: str) -> bool:
        """Return True if submitting qty would exceed exposure cap."""
        if symbol in recent_buys and time.time() - recent_buys[symbol] < 60:
            # AI-AGENT-REF: skip exposure cap check briefly after a buy
            self.logger.info("EXPOSURE_CAP_SKIP_RECENT_BUY for %s", symbol)
            return False
        eng = getattr(self.ctx, "risk_engine", None)
        if eng is None:
            return False
        try:
            acct = self.api.get_account()
            equity = float(getattr(acct, "equity", 0) or 0)
        except Exception:
            return False
        if equity <= 0 or price <= 0:
            return False
        weight = qty * price / equity
        current = eng.exposure.get(asset_class, 0.0)
        cap = eng._dynamic_cap(asset_class)
        projected = current + weight
        if projected > cap:
            self.logger.info(
                "EXPOSURE_CAP_BREACH", 
                extra={
                    "symbol": symbol,
                    "qty": qty,
                    "projected": projected,
                    "cap": cap,
                },
            )
            return True
        return False

    def _flush_partial_buffers(self, force_id: str | None = None) -> None:
        """Emit consolidated fill logs when ``force_id`` is provided."""
        now = time.monotonic()
        ids = list(self._partial_buffer.keys()) if force_id is None else [force_id]
        for oid in ids:
            buf = self._partial_buffer.get(oid)
            if not buf:
                continue
            if force_id is None:
                if now - buf["ts"] > 60:
                    del self._partial_buffer[oid]
                continue
            avg = buf["total_price"] / buf["qty"] if buf["qty"] else 0.0
            self.logger.info(
                "ORDER_FILL_CONSOLIDATED",
                extra={
                    "symbol": buf["symbol"],
                    "order_id": oid,
                    "total": buf["qty"],
                    "fragments": buf["count"],
                    "avg_price": round(avg, 2),
                },
            )
            del self._partial_buffer[oid]

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
        # AI-AGENT-REF: preserve base client_order_id for INITIAL_REBALANCE
        order_class = getattr(order_req, "order_class", OrderClass.NORMAL)
        base_cid = getattr(
            order_req,
            "client_order_id",
            generate_client_order_id(symbol, side, order_class),
        )
        rebalance_ids = getattr(self.ctx, "rebalance_ids", {})
        rebalance_attempts = getattr(self.ctx, "rebalance_attempts", {})
        use_rebalance_id = symbol in rebalance_ids
        if use_rebalance_id:
            base_cid = rebalance_ids[symbol]
        else:
            if isinstance(order_req, dict):
                order_req.setdefault("client_order_id", base_cid)
            else:
                setattr(order_req, "client_order_id", base_cid)
        backoff = 1
        for attempt in range(3):
            if order_class is OrderClass.INITIAL_REBALANCE:
                cid = f"{base_cid}-{uuid4().hex[:8]}"
                if isinstance(order_req, dict):
                    order_req["client_order_id"] = cid
                else:
                    setattr(order_req, "client_order_id", cid)
            elif use_rebalance_id:
                cid = base_cid if attempt == 0 and not rebalance_attempts.get(symbol) else f"{base_cid}_{uuid4().hex[:8]}"
                if isinstance(order_req, dict):
                    order_req["client_order_id"] = cid
                else:
                    setattr(order_req, "client_order_id", cid)
            elif attempt:
                new_cid = f"{base_cid}-{uuid4().hex[:8]}"
                if isinstance(order_req, dict):
                    order_req["client_order_id"] = new_cid
                else:
                    setattr(order_req, "client_order_id", new_cid)
            try:
                order = submit_order(api, order_req, self.logger)
                self.logger.info("Order submit response for %s: %s", symbol, order)
                if not getattr(order, "id", None) and not SHADOW_MODE:
                    self.logger.error("Order failed for %s: %s", symbol, order)
                return order
            except (APIError, TimeoutError) as e:
                self.logger.error(
                    "ORDER_SUBMIT_ERROR %s | params=%s | %s",
                    symbol,
                    {
                        "symbol": getattr(order_req, "symbol", ""),
                        "qty": getattr(order_req, "qty", 0),
                        "side": getattr(order_req, "side", ""),
                        "type": order_req.__class__.__name__,
                    },
                    e,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                if use_rebalance_id:
                    rebalance_attempts[symbol] = rebalance_attempts.get(symbol, 0) + 1
                if attempt == 2:
                    self.logger.warning(
                        "submit_order failed for %s after retries: %s",
                        symbol,
                        e,
                    )
                    return None
            except (RuntimeError, ValueError) as exc:
                self.logger.exception(
                    "Unexpected error placing order for %s: %s",
                    symbol,
                    exc,
                )
                return None
        return None

    def _wait_for_fill(self, order_id: str) -> None:
        """Block until the order ``order_id`` is filled or canceled."""
        backoff = 1
        while True:
            try:
                od = self.api.get_order_by_id(order_id)
                status = getattr(od, "status", "")
                if status in {"filled", "canceled", "rejected"}:
                    return
                backoff = 1
            except Exception as exc:  # pragma: no cover - network
                self.logger.debug("ORDER_POLL_FAIL %s: %s", order_id, exc)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    def _handle_order_result(
        self,
        symbol: str,
        side: str,
        order: Order,
        expected_price: Optional[float],
        slice_qty: int,
        start_time: float,
    ) -> int:
        self._flush_partial_buffers()
        status = getattr(order, "status", "")
        order_id = getattr(order, "id", "")
        if status in ("new", "pending_new"):
            time.sleep(1)
            try:
                refreshed = self.api.get_order_by_id(order_id)
                status = getattr(refreshed, "status", status)
                order = refreshed
            except Exception as exc:  # pragma: no cover - network issues
                self.logger.debug("Order refresh failed for %s: %s", order_id, exc)
            if status not in {"filled", "canceled", "rejected"}:
                self._wait_for_fill(order_id)
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
                int(getattr(order, "filled_qty", slice_qty)),
                side,
                fill_price,
                datetime.now(timezone.utc).isoformat(),
                {"status": status, "mode": "SHADOW" if SHADOW_MODE else "LIVE"},
            )
            log_json_audit(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "qty": int(getattr(order, "filled_qty", slice_qty)),
                    "price": fill_price,
                    "order_id": order_id,
                    "client_order_id": getattr(order, "client_order_id", order_id),
                    "status": status,
                    "partial_fills": getattr(order, "legs", []),
                }
            )
            actual_qty = int(getattr(order, "filled_qty", slice_qty))
            log_trade(
                symbol,
                actual_qty,
                side,
                fill_price,
                datetime.now(timezone.utc).isoformat(),
                order_id,
            )
            filled_qty = actual_qty
            if order_id in self._partial_buffer:
                buf = self._partial_buffer[order_id]
                add_qty = max(0, slice_qty - buf["qty"])
                buf["qty"] += add_qty
                buf["count"] += 1 if add_qty else 0
                buf["total_price"] += add_qty * fill_price
                self._flush_partial_buffers(order_id)
        elif status == "partially_filled":
            part_qty = int(getattr(order, "filled_qty", 0) or 0)
            self._partial_buffer.setdefault(
                order_id,
                {
                    "qty": 0,
                    "count": 0,
                    "total_price": 0.0,
                    "symbol": symbol,
                    "ts": time.monotonic(),
                },
            )
            buf = self._partial_buffer[order_id]
            buf["qty"] += part_qty
            buf["count"] += 1
            buf["total_price"] += part_qty * fill_price
            buf["ts"] = time.monotonic()
            self._flush_partial_buffers()
            filled_qty = part_qty
        elif status in ("pending_new", "new"):
            self.logger.info("Order status for %s: %s", symbol, status)
        elif status in ("rejected", "failed"):
            self.logger.error("Order failed for %s: %s", symbol, status)
        else:
            self.logger.error(
                "ORDER_STATUS",
                extra={"symbol": symbol, "order_id": order_id, "status": status},
            )

        if status not in ("filled", "accepted", "new", "pending_new", "partially_filled"):
            self.logger.warning(
                "UNEXPECTED_ORDER_STATUS", extra={"symbol": symbol, "status": status}
            )
        if status not in {"filled", "canceled", "rejected"}:
            self._wait_for_fill(order_id)
        if self.orders_total is not None:
            self.orders_total.inc()
        return filled_qty

    def execute_order(self, symbol: str, qty: int, side: str, asset_class: str = "equity") -> Optional[Order]:
        """Execute an order for the given asset class."""
        remaining = int(round(qty))
        last_order = None
        api = self._select_api(asset_class)
        existing = self._available_qty(api, symbol)
        if side.lower() == "buy" and existing > 0:
            self.logger.info("SKIP_HELD_POSITION | already long, skipping buy")
            return None
        if side.lower() == "sell" and existing == 0:
            self.logger.info("SKIP_NO_POSITION | no shares to sell, skipping")
            return None
        if side.lower() == "sell":
            avail = self._available_qty(api, symbol)
            if avail <= 0:
                self.logger.info("No position to sell for %s, skipping.", symbol)
                return None
            if remaining > avail:
                self.logger.warning(
                    "Requested %s but only %s available for %s; adjusting",
                    remaining,
                    avail,
                    symbol,
                )
                remaining = int(round(avail))
        steps = 0
        tried_partial = False
        max_steps = 20
        while remaining > 0 and steps < max_steps:
            remaining, skip = self._assess_liquidity(symbol, remaining, attempted=tried_partial)
            if skip or remaining <= 0:
                break
            if not tried_partial and remaining < qty:
                tried_partial = True
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
            if expected_price and self._check_exposure_cap(asset_class, slice_qty, expected_price, symbol):
                break
            order_key = getattr(order_req, "client_order_id", f"{symbol}-{side}")
            order_cls = getattr(order_req, "order_class", OrderClass.NORMAL)
            if self._is_duplicate_order(order_req):
                self.logger.info(
                    "DUPLICATE_ORDER_SKIP",
                    extra={"symbol": symbol, "client_order_id": order_key},
                )
                break
            if order_cls is not OrderClass.INITIAL_REBALANCE:
                self._cycle_orders.add(order_key)
                self._cycle_symbols.add(symbol)
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
            steps += 1
            if remaining > 0:
                time.sleep(random.uniform(0.05, 0.15))
        if steps >= max_steps and remaining > 0:
            self.logger.error(
                "ORDER_MAX_STEPS_EXCEEDED",
                extra={"symbol": symbol, "remaining": remaining},
            )
        self._record_fill_steps(max(1, steps))
        if last_order:
            oid = getattr(last_order, "id", None)
            if oid:
                self._flush_partial_buffers(oid)
            if side.lower() == "buy":
                # AI-AGENT-REF: track recent buys to prevent immediate sell-offs
                recent_buys[symbol] = time.time()
            else:
                self.logger.info("EXITING %s via order %s", symbol, oid)
        return last_order


__all__ = ["ExecutionEngine", "log_order"]
