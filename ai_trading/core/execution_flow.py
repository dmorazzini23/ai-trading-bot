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
    interval = 0.2 if timeout <= 1 else min(1.0, max(0.2, timeout / 10))
    open_statuses = {"new", "accepted", "partially_filled", "pending_new"}
    last_status = ""
    last_filled: float | None = None
    last_qty: float | None = None

    def _coerce_numeric_attr(obj: Any, primary: str, aliases: tuple[str, ...] = ()) -> float | None:
        names = (primary,) + aliases
        value: float | None = None
        for name in names:
            if not hasattr(obj, name):
                continue
            raw = getattr(obj, name)
            if raw in (None, ""):
                continue
            if isinstance(raw, (int, float)):
                value = float(raw)
            else:
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
            for target in names:
                if hasattr(obj, target):
                    try:
                        setattr(obj, target, value)
                    except Exception:
                        # Ignore read-only attributes from broker SDKs
                        pass
            break
        return value

    while True:
        try:
            od = ctx.api.get_order(order_id)  # type: ignore[attr-defined]
            status_raw = getattr(od, "status", "")
            status = str(status_raw or "").lower()
            filled = _coerce_numeric_attr(od, "filled_qty", ("filled_quantity",))
            qty = _coerce_numeric_attr(od, "qty", ("quantity",))
            last_status = str(status_raw or "")
            last_filled = filled if filled is not None else last_filled
            last_qty = qty if qty is not None else last_qty
            if status not in open_statuses:
                logger.info(
                    "ORDER_FINAL_STATUS",
                    extra={
                        "order_id": order_id,
                        "status": last_status,
                        "filled_qty": filled if filled is not None else getattr(od, "filled_qty", "0"),
                        "qty": qty if qty is not None else getattr(od, "qty", getattr(od, "quantity", "0")),
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

    if last_status:
        logger.warning(
            "ORDER_POLL_TIMEOUT",
            extra={
                "order_id": order_id,
                "status": last_status,
                "timeout_s": timeout,
                "filled_qty": last_filled,
                "qty": last_qty,
            },
        )


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
        APIError,
        utc_now_iso,
        slippage_total,
        slippage_count,
        _slippage_log,
        slippage_lock,
        SLIPPAGE_LOG_FILE,
        safe_submit_order,
    )
    from ai_trading.core import bot_engine as _bot_engine

    _bot_engine._ensure_alpaca_classes()
    OrderSide = _bot_engine.OrderSide
    TimeInForce = _bot_engine.TimeInForce
    LimitOrderRequest = _bot_engine.LimitOrderRequest
    MarketOrderRequest = _bot_engine.MarketOrderRequest

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
            _bot_engine._ensure_alpaca_classes()
            if (
                _bot_engine._ALPACA_IMPORT_ERROR is not None
                or _bot_engine.StockLatestQuoteRequest is None
            ):
                raise RuntimeError("StockLatestQuoteRequest unavailable")
            req = _bot_engine.StockLatestQuoteRequest(symbol_or_symbols=[symbol])
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
            RuntimeError,
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
    from ai_trading.core import bot_engine as _bot_engine

    _bot_engine._ensure_alpaca_classes()
    MarketOrderRequest = _bot_engine.MarketOrderRequest
    LimitOrderRequest = _bot_engine.LimitOrderRequest
    OrderSide = _bot_engine.OrderSide
    TimeInForce = _bot_engine.TimeInForce
    safe_submit_order = _bot_engine.safe_submit_order

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


def pov_submit(
    ctx: Any,
    symbol: str,
    total_qty: int,
    side: str,
    cfg: Any,
) -> bool:
    """Place POV (percentage of volume) slices with backoff on missing data."""
    from ai_trading.core.bot_engine import (
        fetch_minute_df_safe,
        DataFetchError,
        Quote,
        APIError,
        submit_order,
    )
    from ai_trading.core import bot_engine as _bot_engine
    import random

    import os as _os
    test_mode = _os.getenv("PYTEST_RUNNING") in {"1", "true", "True"}

    def _sleep(seconds: float) -> None:
        if not test_mode and seconds > 0:
            pytime.sleep(seconds)

    placed = 0
    intended_total = 0
    partial_fill_summaries: list[dict[str, Any]] = []
    retries = 0
    interval = cfg.sleep_interval
    while placed < total_qty:
        try:
            df = fetch_minute_df_safe(symbol)
        except DataFetchError:
            retries += 1
            if retries > cfg.max_retries:
                logger.warning(
                    f"[pov_submit] no minute data after {cfg.max_retries} retries, aborting",
                    extra={"symbol": symbol},
                )
                return False
            logger.warning(
                f"[pov_submit] missing bars, retry {retries}/{cfg.max_retries} in {interval:.1f}s",
                extra={"symbol": symbol},
            )
            sleep_time = interval * (0.8 + 0.4 * random.random())
            _sleep(sleep_time)
            interval = min(interval * cfg.backoff_factor, cfg.max_backoff_interval)
            continue
        if df is None or df.empty:
            retries += 1
            if retries > cfg.max_retries:
                logger.warning(
                    f"[pov_submit] no minute data after {cfg.max_retries} retries, aborting",
                    extra={"symbol": symbol},
                )
                return False
            logger.warning(
                f"[pov_submit] missing bars, retry {retries}/{cfg.max_retries} in {interval:.1f}s",
                extra={"symbol": symbol},
            )
            sleep_time = interval * (0.8 + 0.4 * random.random())
            _sleep(sleep_time)
            interval = min(interval * cfg.backoff_factor, cfg.max_backoff_interval)
            continue
        retries = 0
        interval = cfg.sleep_interval
        try:
            _bot_engine._ensure_alpaca_classes()
            if (
                _bot_engine._ALPACA_IMPORT_ERROR is not None
                or _bot_engine.StockLatestQuoteRequest is None
            ):
                raise RuntimeError("StockLatestQuoteRequest unavailable")
            req = _bot_engine.StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quote: Quote = ctx.data_client.get_stock_latest_quote(req)  # type: ignore[assignment]
            spread = (
                (quote.ask_price - quote.bid_price)
                if getattr(quote, "ask_price", None) and getattr(quote, "bid_price", None)
                else 0.0
            )
        except APIError as e:
            logger.warning(f"[pov_submit] Alpaca quote failed for {symbol}: {e}")
            spread = 0.0
        except Exception:
            spread = 0.0

        vol = float(df["volume"].iloc[-1])
        dynamic_spread_threshold = min(0.10, max(0.02, vol / 1000000.0 * 0.05))
        if spread > dynamic_spread_threshold:
            slice_qty = min(int(vol * cfg.pct * 0.75), total_qty - placed)
        else:
            slice_qty = min(int(vol * cfg.pct), total_qty - placed)
        if slice_qty < 1:
            logger.debug(
                f"[pov_submit] slice_qty<1 (vol={vol}), waiting",
                extra={"symbol": symbol},
            )
            _sleep(cfg.sleep_interval * (0.8 + 0.4 * random.random()))
            continue
        try:
            order = submit_order(ctx, symbol, slice_qty, side)
            if order is None:
                logger.warning(
                    "[pov_submit] submit_order returned None for slice, skipping",
                    extra={"symbol": symbol, "slice_qty": slice_qty},
                )
                continue
            intended_total += slice_qty
            actual_filled = int(getattr(order, "filled_qty", "0") or "0")
            new_total_filled = placed + actual_filled
            if actual_filled < slice_qty:
                partial_summary = {
                    "symbol": symbol,
                    "slice_index": len(partial_fill_summaries) + 1,
                    "intended_slice_qty": slice_qty,
                    "actual_slice_filled": actual_filled,
                    "running_intended_total": intended_total,
                    "running_actual_total": new_total_filled,
                    "running_fill_gap": max(0, intended_total - new_total_filled),
                    "order_id": getattr(order, "id", ""),
                    "status": getattr(order, "status", ""),
                }
                partial_fill_summaries.append(partial_summary)
                logger.warning("POV_SLICE_PARTIAL_FILL", extra=partial_summary)
            placed = new_total_filled
        except Exception as e:
            logger.exception(
                f"[pov_submit] submit_order failed on slice, aborting: {e}",
                extra={"symbol": symbol},
            )
            return False
        logger.info(
            "POV_SUBMIT_SLICE",
            extra={
                "symbol": symbol,
                "slice_qty": slice_qty,
                "actual_filled": actual_filled,
                "total_placed": placed,
                "intended_total": intended_total,
            },
        )
        _sleep(cfg.sleep_interval * (0.8 + 0.4 * random.random()))
    if partial_fill_summaries:
        summary_payload = {
            "symbol": symbol,
            "total_intended": intended_total,
            "total_actual": placed,
            "fill_gap": max(0, intended_total - placed),
            "partial_fills": partial_fill_summaries,
        }
        logger.info("POV_PARTIAL_FILL_SUMMARY", extra=summary_payload)
        tracker = getattr(ctx, "partial_fill_tracker", None)
        if not isinstance(tracker, dict):
            tracker = {}
            setattr(ctx, "partial_fill_tracker", tracker)
        tracker[symbol] = summary_payload
    logger.info(
        "POV_SUBMIT_COMPLETE",
        extra={"symbol": symbol, "placed": placed, "intended_total": intended_total},
    )
    return True

__all__.append("pov_submit")


def execute_entry(ctx: Any, symbol: str, qty: int, side: str) -> None:
    """Execute entry order with slicing policies and target initialization."""
    from ai_trading.core.bot_engine import (
        submit_order,
        vwap_pegged_submit,
        pov_submit,
        SLICE_THRESHOLD,
        POV_SLICE_PCT,
        fetch_minute_df_safe,
        DataFetchError,
        prepare_indicators,
        get_latest_close,
        get_take_profit_factor,
        is_high_vol_regime,
        scaled_atr_stop,
        targets_lock,
        BotState,
        get_trade_logger,  # AI-AGENT-REF: ensure trade log initialization
    )
    import numpy as np  # local import to avoid global cost
    from datetime import UTC, datetime
    from zoneinfo import ZoneInfo

    PACIFIC = ZoneInfo("America/Los_Angeles")

    # Ensure trade log file exists before logging first entry
    tl = get_trade_logger()
    if getattr(ctx, "trade_logger", None) is None:
        ctx.trade_logger = tl

    if getattr(ctx, "api", None) is None:
        logger.warning("ctx.api is None - cannot execute entry")
        return
    try:
        buying_pw = float(ctx.api.get_account().buying_power)
        if buying_pw <= 0:
            logger.info("NO_BUYING_POWER", extra={"symbol": symbol})
            return
    except Exception as exc:
        logger.warning("Failed to get buying power for %s: %s", symbol, exc)
        return
    if qty is None or qty <= 0 or not np.isfinite(qty):
        logger.error(f"Invalid order quantity for {symbol}: {qty}. Skipping order.")
        return
    if POV_SLICE_PCT > 0 and qty > SLICE_THRESHOLD:
        logger.info("POV_SLICE_ENTRY", extra={"symbol": symbol, "qty": qty})
        pov_submit(ctx, symbol, qty, side)
    elif qty > SLICE_THRESHOLD:
        logger.info("VWAP_SLICE_ENTRY", extra={"symbol": symbol, "qty": qty})
        vwap_pegged_submit(ctx, symbol, qty, side)
    else:
        logger.info("MARKET_ENTRY", extra={"symbol": symbol, "qty": qty})
        submit_order(ctx, symbol, qty, side)

    try:
        raw = fetch_minute_df_safe(symbol)
    except DataFetchError:
        logger.warning("NO_MINUTE_BARS_POST_ENTRY", extra={"symbol": symbol})
        return
    if raw is None or getattr(raw, "empty", True):
        logger.warning("NO_MINUTE_BARS_POST_ENTRY", extra={"symbol": symbol})
        return
    try:
        df_ind = prepare_indicators(raw)
        if df_ind is None or getattr(df_ind, "empty", True):
            logger.warning("INSUFFICIENT_INDICATORS_POST_ENTRY", extra={"symbol": symbol})
            return
    except (ValueError, KeyError) as exc:
        logger.warning(f"Indicator preparation failed for {symbol}: {exc}")
        return
    entry_price = get_latest_close(df_ind)
    ctx.trade_logger.log_entry(symbol, entry_price, qty, side, "", "", confidence=0.5)

    now_pac = datetime.now(UTC).astimezone(PACIFIC)
    mo = datetime.combine(now_pac.date(), ctx.market_open, PACIFIC)
    mc = datetime.combine(now_pac.date(), ctx.market_close, PACIFIC)
    tp_base = get_take_profit_factor()
    tp_factor = tp_base * 1.1 if is_high_vol_regime() else tp_base
    stop, take = scaled_atr_stop(
        entry_price,
        df_ind["atr"].iloc[-1],
        now_pac,
        mo,
        mc,
        max_factor=tp_factor,
        min_factor=0.5,
    )
    with targets_lock:
        ctx.stop_targets[symbol] = stop
        ctx.take_profit_targets[symbol] = take


def execute_exit(ctx: Any, state: Any, symbol: str, qty: int) -> None:
    from ai_trading.core.bot_engine import (
        fetch_minute_df_safe,
        DataFetchError,
        get_latest_close,
        send_exit_order,
        targets_lock,
    )
    import numpy as np
    if qty is None or not np.isfinite(qty) or qty <= 0:
        logger.warning(f"Skipping {symbol}: computed qty <= 0")
        return
    try:
        raw = fetch_minute_df_safe(symbol)
    except DataFetchError:
        logger.warning("NO_MINUTE_BARS_POST_EXIT", extra={"symbol": symbol})
        raw = None
    exit_price = get_latest_close(raw) if raw is not None else 1.0
    send_exit_order(ctx, symbol, qty, exit_price, "manual_exit")
    try:
        ctx.trade_logger.log_exit(state, symbol, exit_price)
    except Exception:
        pass
    with targets_lock:
        ctx.take_profit_targets.pop(symbol, None)
        ctx.stop_targets.pop(symbol, None)


def exit_all_positions(ctx: Any) -> None:
    raw_positions = ctx.api.list_positions()
    for pos in raw_positions:
        qty = abs(int(pos.qty))
        if qty:
            send_exit_order(ctx, pos.symbol, qty, 0.0, "eod_exit", raw_positions=raw_positions)
            logger.info("EOD_EXIT", extra={"symbol": pos.symbol, "qty": qty})


def _liquidate_all_positions(runtime: Any) -> None:
    exit_all_positions(runtime)


def liquidate_positions_if_needed(runtime: Any) -> None:
    from ai_trading.core.bot_engine import check_halt_flag
    if check_halt_flag(runtime):
        logger.info("TRADING_HALTED_VIA_FLAG is active: NOT liquidating positions")
        return
    # normal liquidation logic would be implemented here (intentionally left as stub)
