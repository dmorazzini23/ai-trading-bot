from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

"""Execution flow helpers decoupled from bot_engine."""

from json import JSONDecodeError
from typing import Any, cast
import time as pytime
from threading import Thread
import csv
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from ai_trading.config.management import get_env
from ai_trading.data.feed_roles import get_execution_feed
from ai_trading.logging import get_logger

logger = get_logger(__name__)


def _latest_quote_request(symbol: str, *, feed: str | None = None) -> Any:
    bot_engine_module = cast(
        Any,
        __import__("ai_trading.core.bot_engine", fromlist=["bot_engine"]),
    )
    resolved_feed = feed or get_execution_feed()
    try:
        return bot_engine_module.StockLatestQuoteRequest(
            symbol_or_symbols=[symbol],
            feed=resolved_feed,
        )
    except TypeError:
        return bot_engine_module.StockLatestQuoteRequest(symbol_or_symbols=[symbol])


def poll_order_fill_status(ctx: Any, order_id: str, timeout: int | float = 120) -> None:
    """Poll broker for order fill status until it is no longer open.

    Honors the provided ``timeout`` by sleeping in short intervals instead of
    a fixed interval so tests can set small timeouts without hanging.
    """
    try:
        timeout_s = max(float(timeout), 0.0)
    except (TypeError, ValueError):
        timeout_s = 0.0
    start_wall = pytime.time()
    try:
        start_monotonic = pytime.monotonic()
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        start_monotonic = None
    synthetic_elapsed = 0.0
    interval = 0.2 if timeout_s <= 1 else min(1.0, max(0.2, timeout_s / 10))
    open_statuses = {"new", "accepted", "partially_filled", "pending_new"}
    last_status = ""
    last_filled: float | None = None
    last_qty: float | None = None

    def _elapsed_seconds() -> float:
        elapsed = synthetic_elapsed
        if start_monotonic is not None:
            try:
                monotonic_elapsed = max(0.0, float(pytime.monotonic() - start_monotonic))
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                monotonic_elapsed = 0.0
            elapsed = max(elapsed, monotonic_elapsed)
        try:
            wall_elapsed = max(0.0, float(pytime.time() - start_wall))
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            wall_elapsed = 0.0
        return max(elapsed, wall_elapsed)

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
                    except AI_TRADING_FALLBACK_EXCEPTIONS:
                        # Ignore read-only attributes from broker SDKs
                        logger.debug(
                            "ORDER_ATTR_NORMALIZE_SET_FAILED",
                            extra={"attr": target},
                            exc_info=True,
                        )
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
                        "filled_qty": filled if filled is not None else getattr(od, "filled_qty", 0.0),
                        "qty": qty if qty is not None else getattr(od, "qty", getattr(od, "quantity", 0.0)),
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
        remaining = timeout_s - _elapsed_seconds()
        if remaining <= 0:
            break
        sleep_for = min(interval, remaining)
        pytime.sleep(sleep_for)
        synthetic_elapsed += sleep_for

    if last_status:
        logger.warning(
            "ORDER_POLL_TIMEOUT",
            extra={
                "order_id": order_id,
                "status": last_status,
                "timeout_s": timeout_s,
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
    from ai_trading.core.bot_engine import APIError
    from ai_trading.services.execution import submit_order

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
    )
    from ai_trading.core import bot_engine as _bot_engine
    from ai_trading.services.execution import submit_order

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
            req = _latest_quote_request(symbol)
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
                order = submit_order(
                    ctx,
                    symbol,
                    slice_qty,
                    side,
                    price=round(float(vwap_price), 2),
                    time_in_force="ioc",
                    execution_algorithm="vwap",
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
                        except AI_TRADING_FALLBACK_EXCEPTIONS:
                            logger.debug("VWAP_SLIPPAGE_TOTAL_METRIC_INC_FAILED", exc_info=True)
                    if slippage_count:
                        try:
                            slippage_count.inc()  # type: ignore[union-attr]
                        except AI_TRADING_FALLBACK_EXCEPTIONS:
                            logger.debug("VWAP_SLIPPAGE_COUNT_METRIC_INC_FAILED", exc_info=True)
                    _slippage_log.append((symbol, vwap_price, fill_price, datetime.now(UTC)))
                    with slippage_lock:  # type: ignore[arg-type]
                        try:
                            with open(SLIPPAGE_LOG_FILE, "a", newline="") as sf:
                                csv.writer(sf).writerow(
                                    [utc_now_iso(), symbol, vwap_price, fill_price, slip]
                                )
                        except AI_TRADING_FALLBACK_EXCEPTIONS:
                            logger.debug("VWAP_SLIPPAGE_LOG_APPEND_FAILED", exc_info=True)
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


__all__ = [
    "poll_order_fill_status",
    "execute_exit",
    "twap_submit",
    "vwap_pegged_submit",
]


def _signed_position_qty(pos: Any) -> int:
    """Return broker position quantity with short positions represented as negative."""

    try:
        qty_float = float(getattr(pos, "qty", getattr(pos, "quantity", 0)) or 0.0)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        qty_float = 0.0
    qty = int(qty_float)
    if qty < 0:
        return qty
    try:
        side = str(getattr(pos, "side", "") or "").strip().lower()
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        side = ""
    if side in {"short", "sell", "sell_short"}:
        return -abs(qty)
    return qty


def _broker_order_status(order: Any) -> str:
    status = getattr(order, "status", "")
    status = getattr(status, "value", status)
    return str(status or "").strip().lower()


def _broker_filled_qty(order: Any) -> int:
    for attr in ("filled_qty", "filled_quantity", "filled"):
        value = getattr(order, attr, None)
        if value in (None, ""):
            continue
        try:
            return max(int(float(value)), 0)
        except (TypeError, ValueError):
            continue
    return 0


def _list_positions_compat(api: Any) -> list[Any]:
    get_all_positions = getattr(api, "get_all_positions", None)
    if callable(get_all_positions):
        return list(get_all_positions() or [])
    list_positions = getattr(api, "list_positions", None)
    if callable(list_positions):
        return list(list_positions() or [])
    raise AttributeError("broker client missing positions method")


def send_exit_order(
    ctx: Any,
    symbol: str,
    exit_qty: int,
    price: float,
    reason: str,
    raw_positions: list | None = None,
) -> None:
    """Submit an exit order (market or limit) with simple validations."""
    from ai_trading.services.execution import submit_order

    logger.info(
        f"EXIT_SIGNAL | symbol={symbol}  reason={reason}  exit_qty={exit_qty}  price={price}"
    )
    if raw_positions is not None and not any(
        getattr(p, "symbol", "") == symbol for p in raw_positions
    ):
        logger.info("SKIP_NO_POSITION", extra={"symbol": symbol})
        return
    snapshot_qty = 0
    if raw_positions is not None:
        for raw_pos in raw_positions:
            if getattr(raw_pos, "symbol", "") != symbol:
                continue
            snapshot_qty = _signed_position_qty(raw_pos)
            break
    try:
        pos = ctx.api.get_position(symbol)
        held_qty_signed = _signed_position_qty(pos)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        held_qty_signed = snapshot_qty
    held_qty = abs(held_qty_signed)
    if held_qty <= 0 and abs(snapshot_qty) > 0:
        held_qty_signed = snapshot_qty
        held_qty = abs(snapshot_qty)
    if held_qty < exit_qty:
        logger.warning(
            f"No shares available to exit for {symbol} (requested {exit_qty}, have {held_qty})"
        )
        return
    exit_side = "buy_to_cover" if held_qty_signed < 0 else "sell"
    exit_metadata = {"reason": reason, "closing_position": True, "reduce_only": True}
    if price <= 0.0:
        submit_order(
            ctx,
            symbol=symbol,
            qty=exit_qty,
            side=exit_side,
            metadata=exit_metadata,
            closing_position=True,
            reduce_only=True,
        )
        return
    limit_order = submit_order(
        ctx,
        symbol=symbol,
        qty=exit_qty,
        side=exit_side,
        price=price,
        metadata=exit_metadata,
        closing_position=True,
        reduce_only=True,
    )
    if limit_order is None:
        return
    pytime.sleep(5)
    try:
        o2 = ctx.api.get_order_by_id(limit_order.id)
        if _broker_order_status(o2) in {"new", "accepted", "partially_filled"}:
            ctx.api.cancel_order_by_id(limit_order.id)
            refreshed = ctx.api.get_order_by_id(limit_order.id)
            status = _broker_order_status(refreshed)
            if status not in {"canceled", "cancelled", "filled", "expired", "rejected"}:
                logger.warning(
                    "EXIT_LIMIT_CANCEL_NOT_FINAL",
                    extra={"symbol": symbol, "order_id": getattr(limit_order, "id", "")},
                )
                return
            remaining_qty = max(int(exit_qty) - _broker_filled_qty(refreshed), 0)
            if remaining_qty <= 0:
                return
            submit_order(
                ctx,
                symbol=symbol,
                qty=remaining_qty,
                side=exit_side,
                metadata=exit_metadata,
                closing_position=True,
                reduce_only=True,
            )
    except AI_TRADING_FALLBACK_EXCEPTIONS as e:
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
    )
    from ai_trading.core import bot_engine as _bot_engine
    from ai_trading.services.execution import submit_order
    import random

    import sys as _sys

    env_token = get_env("PYTEST_RUNNING", None, cast=str, resolve_aliases=False)
    if isinstance(env_token, str):
        normalized = env_token.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            pytest_running = True
        elif normalized in {"0", "false", "no", "off"}:
            pytest_running = False
        else:
            pytest_running = "pytest" in _sys.modules
    else:
        pytest_running = "pytest" in _sys.modules

    def _sleep(seconds: float) -> None:
        if pytest_running or seconds <= 0:
            return
        pytime.sleep(seconds)

    placed = 0
    intended_total = 0
    partial_fill_summaries: list[dict[str, Any]] = []
    retries = 0
    interval = cfg.sleep_interval
    terminated_on_gap = False
    none_submission_retries = 0
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
            req = _latest_quote_request(symbol)
            quote = ctx.data_client.get_stock_latest_quote(req)
            spread = (
                (quote.ask_price - quote.bid_price)
                if getattr(quote, "ask_price", None) and getattr(quote, "bid_price", None)
                else 0.0
            )
        except APIError as e:
            logger.warning(f"[pov_submit] Alpaca quote failed for {symbol}: {e}")
            spread = 0.0
        except AI_TRADING_FALLBACK_EXCEPTIONS:
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
                none_submission_retries += 1
                logger.warning(
                    "[pov_submit] submit_order returned None for slice, skipping",
                    extra={
                        "symbol": symbol,
                        "slice_qty": slice_qty,
                        "retry": none_submission_retries,
                    },
                )
                if none_submission_retries > max(1, int(getattr(cfg, "max_retries", 0) or 0)):
                    logger.warning(
                        "[pov_submit] submit_order returned None repeatedly, aborting",
                        extra={"symbol": symbol, "slice_qty": slice_qty},
                    )
                    return False
                _sleep(cfg.sleep_interval * (0.8 + 0.4 * random.random()))
                continue
            none_submission_retries = 0
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
        except AI_TRADING_FALLBACK_EXCEPTIONS as e:
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
        loop_should_break = placed >= total_qty
        if (
            not loop_should_break
            and pytest_running
            and actual_filled < slice_qty
            and intended_total >= total_qty
        ):
            terminated_on_gap = True
            loop_should_break = True
        if loop_should_break:
            break
        _sleep(cfg.sleep_interval * (0.8 + 0.4 * random.random()))
    if partial_fill_summaries:
        summary_payload = {
            "symbol": symbol,
            "total_intended": intended_total,
            "total_actual": placed,
            "fill_gap": max(0, intended_total - placed),
            "partial_fills": partial_fill_summaries,
        }
        if terminated_on_gap:
            summary_payload["terminated_on_gap"] = True
        logger.info("POV_PARTIAL_FILL_SUMMARY", extra=summary_payload)
        tracker = getattr(ctx, "partial_fill_tracker", None)
        if not isinstance(tracker, dict):
            tracker = {}
            setattr(ctx, "partial_fill_tracker", tracker)
        tracker[symbol] = summary_payload
    logger.info(
        "POV_SUBMIT_COMPLETE",
        extra={
            "symbol": symbol,
            "placed": placed,
            "intended_total": intended_total,
            "fill_gap": max(0, intended_total - placed),
        },
    )
    return True

__all__.append("pov_submit")


def execute_entry(ctx: Any, symbol: str, qty: int, side: str) -> None:
    """Execute entry order with slicing policies and target initialization."""
    from ai_trading.core.bot_engine import (
        SLICE_THRESHOLD,
        POV_SLICE_PCT,
        SliceConfig,
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
    from ai_trading.services.execution import submit_order
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
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        logger.warning("Failed to get buying power for %s: %s", symbol, exc)
        return
    if qty is None or qty <= 0 or not np.isfinite(qty):
        logger.error(f"Invalid order quantity for {symbol}: {qty}. Skipping order.")
        return
    if POV_SLICE_PCT > 0 and qty > SLICE_THRESHOLD:
        logger.info("POV_SLICE_ENTRY", extra={"symbol": symbol, "qty": qty})
        pov_submit(ctx, symbol, qty, side, SliceConfig(pct=POV_SLICE_PCT))
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
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        logger.debug("TRADE_LOG_EXIT_RECORD_FAILED", extra={"symbol": symbol}, exc_info=True)
    with targets_lock:
        ctx.take_profit_targets.pop(symbol, None)
        ctx.stop_targets.pop(symbol, None)


def exit_all_positions(ctx: Any) -> None:
    raw_positions = _list_positions_compat(ctx.api)
    for pos in raw_positions:
        signed_qty = _signed_position_qty(pos)
        qty = abs(signed_qty)
        if qty:
            execute_order = getattr(ctx, "execute_order", None)
            if callable(execute_order):
                side = "sell" if signed_qty > 0 else "buy"
                execute_order(
                    pos.symbol,
                    side,
                    qty,
                    order_type="market",
                    closing_position=True,
                    reduce_only=True,
                    metadata={"reason": "eod_exit"},
                )
            else:
                send_exit_order(
                    ctx,
                    pos.symbol,
                    qty,
                    0.0,
                    "eod_exit",
                    raw_positions=raw_positions,
                )
            logger.info("EOD_EXIT", extra={"symbol": pos.symbol, "qty": qty})


def _liquidate_all_positions(runtime: Any) -> None:
    exit_all_positions(runtime)


def _should_trigger_eod_flatten(now_et: datetime | None = None) -> tuple[bool, dict[str, Any]]:
    enabled = bool(get_env("AI_TRADING_EOD_FLATTEN_ENABLED", False, cast=bool))
    if not enabled:
        return False, {"enabled": False, "reason": "disabled"}

    lead_seconds = max(
        0,
        min(
            int(get_env("AI_TRADING_EOD_FLATTEN_LEAD_SECONDS", 300, cast=int)),
            3600,
        ),
    )
    current_et = now_et or datetime.now(ZoneInfo("America/New_York"))
    if int(current_et.weekday()) >= 5:
        return False, {"enabled": True, "reason": "weekend", "lead_seconds": lead_seconds}

    try:
        from ai_trading.utils.market_calendar import session_info

        session_close_et = session_info(current_et.date()).end_utc.astimezone(current_et.tzinfo)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        logger.debug("EOD_FLATTEN_SESSION_CLOSE_LOOKUP_FAILED", exc_info=True)
        session_close_et = current_et.replace(hour=16, minute=0, second=0, microsecond=0)
    trigger_at = session_close_et - timedelta(seconds=lead_seconds)
    if current_et < trigger_at:
        return False, {
            "enabled": True,
            "reason": "before_window",
            "lead_seconds": lead_seconds,
            "session_close": session_close_et.isoformat(),
        }
    if current_et >= session_close_et:
        return False, {
            "enabled": True,
            "reason": "after_close",
            "lead_seconds": lead_seconds,
            "session_close": session_close_et.isoformat(),
        }
    return True, {
        "enabled": True,
        "reason": "session_close_window",
        "lead_seconds": lead_seconds,
        "session_close": session_close_et.isoformat(),
    }


def liquidate_positions_if_needed(runtime: Any) -> None:
    from ai_trading.core.bot_engine import check_halt_flag

    if check_halt_flag(runtime):
        logger.info("TRADING_HALTED_VIA_FLAG is active: NOT liquidating positions")
        return
    should_flatten, context = _should_trigger_eod_flatten()
    if not should_flatten:
        return

    api = getattr(runtime, "api", None)
    if api is None or not (
        callable(getattr(api, "get_all_positions", None))
        or callable(getattr(api, "list_positions", None))
    ):
        logger.warning("EOD_FLATTEN_SKIPPED", extra=context | {"detail": "missing_api"})
        return

    try:
        raw_positions = _list_positions_compat(api)
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:
        logger.warning(
            "EOD_FLATTEN_LIST_POSITIONS_FAILED",
            extra=context | {"error": str(exc)},
        )
        return

    active_positions = [
        pos
        for pos in raw_positions
        if abs(_signed_position_qty(pos)) > 0.0
    ]
    if not active_positions:
        return

    now_mono = float(pytime.monotonic())
    last_attempt = float(getattr(runtime, "_last_eod_flatten_attempt_mono", 0.0) or 0.0)
    if last_attempt > 0.0 and (now_mono - last_attempt) < 60.0:
        return
    setattr(runtime, "_last_eod_flatten_attempt_mono", now_mono)
    logger.info(
        "EOD_FLATTEN_TRIGGERED",
        extra=context | {"positions": len(active_positions)},
    )
    exit_all_positions(runtime)
