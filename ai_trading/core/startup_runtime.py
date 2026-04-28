"""Legacy startup/control-plane helpers extracted from ``bot_engine.py``."""

from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import asyncio
import importlib
import math
import time
from json import JSONDecodeError
from threading import Thread
from typing import Any

from ai_trading.data.reference_reconcile import run_reference_reconciliation_once


def _bot_engine() -> Any:
    return importlib.import_module("ai_trading.core.bot_engine")


def _normalized_sleep_seconds(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return max(0.0, float(parsed))


def _ensure_rebalance_tracking(ctx: Any) -> None:
    for attr_name in ("rebalance_ids", "rebalance_attempts", "rebalance_buys"):
        current = getattr(ctx, attr_name, None)
        if isinstance(current, dict):
            continue
        setattr(ctx, attr_name, {})


def _update_service_status_safe(be: Any, *, status: str, reason: str) -> None:
    runtime_state = getattr(be, "runtime_state", None)
    update_service_status = getattr(runtime_state, "update_service_status", None)
    if not callable(update_service_status):
        return
    try:
        update_service_status(status=status, reason=reason)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        log_debug = getattr(getattr(be, "logger", None), "debug", None)
        if callable(log_debug):
            log_debug("RUNTIME_SERVICE_STATUS_UPDATE_FAILED", exc_info=True)


def _run_trade_updates_stream(be: Any, ctx: Any, start_trade_updates_stream: Any) -> None:
    try:
        asyncio.run(
            start_trade_updates_stream(
                be.ALPACA_API_KEY,
                be.ALPACA_SECRET_KEY,
                be.trading_client,
                be.state,
                paper=True,
                running=ctx.stream_event,
            )
        )
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        be.logger.warning(
            "TRADE_UPDATES_STREAM_FAILED",
            extra={
                "error_type": exc.__class__.__name__,
                "detail": str(exc),
            },
        )
        _update_service_status_safe(
            be,
            status="degraded",
            reason="trade_updates_stream_failed",
        )
        return
    stream_event = getattr(ctx, "stream_event", None)
    stream_still_running = False
    is_set = getattr(stream_event, "is_set", None)
    if callable(is_set):
        try:
            stream_still_running = bool(is_set())
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            stream_still_running = False
    if stream_still_running:
        be.logger.warning(
            "TRADE_UPDATES_STREAM_EXITED",
            extra={"reason": "unexpected_return"},
        )
        _update_service_status_safe(
            be,
            status="degraded",
            reason="trade_updates_stream_exited",
        )


def schedule_run_all_trades_runtime(runtime: Any) -> None:
    """Spawn ``run_all_trades_worker`` if the market is open."""

    be = _bot_engine()
    if not be.is_market_open():
        be._log_market_closed("Market closed—skipping run_all_trades.")
        return
    be._LAST_MARKET_CLOSED_LOG = 0.0
    be.ensure_alpaca_attached(runtime)
    if not be._validate_trading_api(getattr(runtime, "api", None)):
        return
    thread = be.threading.Thread(
        target=be.run_all_trades_worker,
        args=(be.state, runtime),
        daemon=True,
    )
    thread.start()


def schedule_run_all_trades_with_delay_runtime(runtime: Any) -> None:
    time.sleep(30.0)
    schedule_run_all_trades_runtime(runtime)


def initial_rebalance_runtime(ctx: Any, symbols: list[str]) -> None:
    """Initial portfolio rebalancing."""

    be = _bot_engine()
    _ensure_rebalance_tracking(ctx)

    if ctx.api is None:
        be.logger.warning("ctx.api is None - cannot perform initial rebalance")
        return

    try:
        be.datetime.now(be.UTC).astimezone(be.PACIFIC)
        acct = ctx.api.get_account()
        float(acct.equity)
        cash = float(acct.cash)
        buying_power = float(getattr(acct, "buying_power", cash))
        n = len(symbols)
        if n == 0 or cash <= 0 or buying_power <= 0:
            be.logger.info("INITIAL_REBALANCE_NO_SYMBOLS_OR_NO_CASH")
            return
    except (be.APIError, TimeoutError, ConnectionError, AttributeError, TypeError, ValueError) as exc:
        be.logger.warning(
            "INITIAL_REBALANCE_ACCOUNT_FAIL",
            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
        )
        return

    now_utc = be.datetime.now(be.UTC)
    if now_utc.hour == 0 and now_utc.minute < 15:
        be.logger.info("INITIAL_REBALANCE: Too early—daily bars not live yet.")
    else:
        valid_symbols = []
        valid_prices: dict[str, float] = {}
        for symbol in symbols:
            try:
                df_daily = ctx.data_fetcher.get_daily_df(ctx, symbol)
                price = be.get_latest_close(df_daily)
            except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
                be.logger.warning(
                    "INITIAL_REBALANCE_PRICE_LOAD_FAILED",
                    extra={
                        "symbol": str(symbol or "").strip().upper(),
                        "cause": exc.__class__.__name__,
                        "detail": str(exc),
                    },
                )
                continue
            if price <= 0:
                continue
            valid_symbols.append(symbol)
            valid_prices[symbol] = price

        if not valid_symbols:
            log_level = (
                be.logging.ERROR if be.in_trading_hours(now_utc) else be.logging.WARNING
            )
            be.logger.log(
                log_level,
                (
                    "INITIAL_REBALANCE: No valid prices for any symbol—skipping "
                    "rebalance. Possible data outage or market holiday. "
                    "Check data provider/API status."
                ),
            )
        else:
            total_capital = cash
            weight_per = 1.0 / len(valid_symbols)
            if hasattr(ctx.api, "list_positions"):
                raw_positions = ctx.api.list_positions()
            elif hasattr(ctx.api, "get_all_positions"):
                raw_positions = ctx.api.get_all_positions()
            else:
                raw_positions = []
            positions: dict[str, int] = {}
            for raw_position in raw_positions:
                position_symbol = str(getattr(raw_position, "symbol", "") or "").strip().upper()
                if not position_symbol:
                    continue
                try:
                    positions[position_symbol] = int(float(getattr(raw_position, "qty", 0) or 0))
                except (TypeError, ValueError):
                    be.logger.warning(
                        "INITIAL_REBALANCE_POSITION_PARSE_FAILED",
                        extra={"symbol": position_symbol},
                    )
            for sym in valid_symbols:
                price = valid_prices[sym]
                target_qty = max(1, int((total_capital * weight_per) / price))
                current_qty = int(positions.get(sym, 0))
                if current_qty < target_qty:
                    qty_to_buy = target_qty
                    if qty_to_buy < 1:
                        continue
                    try:
                        cid = ctx.rebalance_ids.get(sym)
                        if not cid:
                            cid = f"{sym}-{be.uuid.uuid4().hex[:8]}"
                            ctx.rebalance_ids[sym] = cid
                            ctx.rebalance_attempts[sym] = 0
                        order_id = be.submit_order(ctx, sym, qty_to_buy, "buy")
                        if order_id:
                            be.logger.info("INITIAL_REBALANCE: Bought %s %s", qty_to_buy, sym)
                            ctx.rebalance_buys[sym] = be.datetime.now(be.UTC)
                        else:
                            be.logger.error(
                                "INITIAL_REBALANCE: Buy failed for %s: order not placed",
                                sym,
                            )
                    except (be.APIError, TimeoutError, ConnectionError) as exc:
                        be.logger.error(
                            "INITIAL_REBALANCE_BUY_FAILED",
                            extra={
                                "symbol": sym,
                                "cause": exc.__class__.__name__,
                                "detail": str(exc),
                            },
                        )
                elif current_qty > target_qty:
                    qty_to_sell = current_qty - target_qty
                    if qty_to_sell < 1:
                        continue
                    try:
                        be.submit_order(ctx, sym, qty_to_sell, "sell")
                        be.logger.info("INITIAL_REBALANCE: Sold %s %s", qty_to_sell, sym)
                    except (be.APIError, TimeoutError, ConnectionError) as exc:
                        be.logger.error(
                            "INITIAL_REBALANCE_SELL_FAILED",
                            extra={
                                "symbol": sym,
                                "cause": exc.__class__.__name__,
                                "detail": str(exc),
                            },
                        )

    ctx.initial_rebalance_done = True
    try:
        if hasattr(ctx.api, "list_positions"):
            pos_list = ctx.api.list_positions()
        elif hasattr(ctx.api, "get_all_positions"):
            pos_list = ctx.api.get_all_positions()
        else:
            pos_list = []
        refreshed_positions: dict[str, int] = {}
        for raw_position in pos_list:
            position_symbol = str(getattr(raw_position, "symbol", "") or "").strip().upper()
            if not position_symbol:
                continue
            try:
                refreshed_positions[position_symbol] = int(
                    float(getattr(raw_position, "qty", 0) or 0)
                )
            except (TypeError, ValueError):
                be.logger.warning(
                    "INITIAL_REBALANCE_REFRESH_POSITION_PARSE_FAILED",
                    extra={"symbol": position_symbol},
                )
        be.state.position_cache = refreshed_positions
        be.state.long_positions = {
            symbol for symbol, qty in be.state.position_cache.items() if qty > 0
        }
        be.state.short_positions = {
            symbol for symbol, qty in be.state.position_cache.items() if qty < 0
        }
    except (be.APIError, TimeoutError, ConnectionError) as exc:
        be.logger.error(
            "Failed to refresh position cache after rebalance",
            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
        )
        be.state.position_cache = {}
        be.state.long_positions = set()
        be.state.short_positions = set()


def run_main_startup_runtime(ctx: Any) -> None:
    """Run legacy startup health checks, cache warmup, and initial rebalance."""

    be = _bot_engine()

    be.cancel_all_open_orders(ctx)
    be.audit_positions(ctx)
    try:
        initial_list = be.load_tickers(be.TICKERS_FILE)
        summary = be.pre_trade_health_check(ctx, initial_list)
        be.logger.info("STARTUP_HEALTH", extra=summary)
        failures = (
            summary["failures"]
            or summary["insufficient_rows"]
            or summary["missing_columns"]
            or summary.get("invalid_values")
            or summary["timezone_issues"]
        )
        stale_data = summary.get("stale_data", [])
        allow_stale_on_startup = (
            str(be.get_env("ALLOW_STALE_DATA_STARTUP", "false") or "").strip().lower()
            == "true"
        )
        if stale_data and allow_stale_on_startup:
            be.logger.warning(
                "BYPASS_STALE_DATA_STARTUP: Allowing trading with stale data for initial deployment",
                extra={"stale_symbols": stale_data, "count": len(stale_data)},
            )
        elif stale_data and not allow_stale_on_startup:
            failures = failures or stale_data

        health_ok = not failures
        if not health_ok:
            be.logger.error("HEALTH_CHECK_FAILED", extra=summary)
            raise SystemExit(1)
        be.logger.info("HEALTH_OK")
        for sym in initial_list:
            try:
                ctx.data_fetcher.get_minute_df(
                    ctx,
                    sym,
                    lookback_minutes=getattr(be.CFG, "min_health_rows", 120),
                )
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
                be.logger.warning("Initial minute prefetch failed for %s: %s", sym, exc)
    except (
        FileNotFoundError,
        OSError,
        KeyError,
        ValueError,
        TypeError,
        TimeoutError,
        ConnectionError,
    ) as exc:
        be.logger.warning(
            "HEALTH_DATA_PROBE_FAILED",
            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
        )
        raise SystemExit(1) from exc

    all_tickers = be.load_tickers(be.TICKERS_FILE)
    now_ts = be.pytime.time()
    with be.sentiment_lock:
        for ticker in all_tickers:
            be._SENTIMENT_CACHE[ticker] = (now_ts, 0.0)

    try:
        if not getattr(ctx, "_rebalance_done", False):
            universe = be.load_tickers(be.TICKERS_FILE)
            initial_rebalance_runtime(ctx, universe)
            ctx._rebalance_done = True
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
        be.logger.warning("[REBALANCE] aborted due to error: %s", exc)


def configure_main_runtime_jobs(ctx: Any) -> None:
    """Configure the legacy runtime scheduler and trade update stream."""

    be = _bot_engine()
    if bool(getattr(ctx, "_runtime_jobs_configured", False)):
        be.logger.info("RUNTIME_JOBS_ALREADY_CONFIGURED")
        return
    setattr(ctx, "_runtime_jobs_configured", True)

    try:
        def gather_minute_data_with_delay() -> None:
            try:
                delay_seconds = _normalized_sleep_seconds(
                    getattr(be.CFG, "scheduler_sleep_seconds", 0.0),
                    default=0.0,
                )
                if delay_seconds > 0.0:
                    time.sleep(delay_seconds)
                schedule_run_all_trades_runtime(ctx)
            except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
                be.logger.exception("gather_minute_data_with_delay failed: %s", exc)

        be.schedule.every(1).minutes.do(
            lambda: Thread(target=gather_minute_data_with_delay, daemon=True).start()
        )
        try:
            gather_minute_data_with_delay()
        except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
            be.logger.exception("Initial data fetch failed", exc_info=exc)

        be.schedule.every(1).minutes.do(
            lambda: Thread(target=be.validate_open_orders, args=(ctx,), daemon=True).start()
        )
        be.schedule.every(1).minutes.do(
            lambda: Thread(target=be._update_risk_engine_exposure, daemon=True).start()
        )
        be.schedule.every(5).minutes.do(
            lambda: Thread(target=be._emit_periodic_metrics, daemon=True).start()
        )
        reference_reconcile_minutes = max(
            1,
            int(
                be.get_env(
                    "AI_TRADING_REFERENCE_RECONCILE_MINUTES",
                    10,
                    cast=int,
                )
                or 10
            ),
        )

        def _run_reference_reconcile_with_guard() -> None:
            try:
                run_reference_reconciliation_once()
            except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
                be.logger.warning(
                    "REFERENCE_RECONCILE_FAILED",
                    extra={
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                    },
                )

        be.schedule.every(reference_reconcile_minutes).minutes.do(
            lambda: Thread(target=_run_reference_reconcile_with_guard, daemon=True).start()
        )
        be.schedule.every(6).hours.do(
            lambda: Thread(target=be.update_signal_weights, daemon=True).start()
        )
        be.schedule.every(30).minutes.do(
            lambda: Thread(target=be.update_bot_mode, args=(be.state,), daemon=True).start()
        )
        be.schedule.every(30).minutes.do(
            lambda: Thread(target=be.adaptive_risk_scaling, args=(ctx,), daemon=True).start()
        )
        be.schedule.every(be.get_rebalance_interval_min()).minutes.do(
            lambda: Thread(target=be.maybe_rebalance, args=(ctx,), daemon=True).start()
        )
        be.schedule.every().day.at("23:55").do(
            lambda: Thread(target=be.check_disaster_halt, daemon=True).start()
        )

        ctx.stream_event = asyncio.Event()
        ctx.stream_event.set()
        _, start_trade_updates_stream = be._alpaca_symbols()

        be.threading.Thread(
            target=lambda: _run_trade_updates_stream(be, ctx, start_trade_updates_stream),
            daemon=True,
        ).start()
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        setattr(ctx, "_runtime_jobs_configured", False)
        raise


__all__ = [
    "configure_main_runtime_jobs",
    "initial_rebalance_runtime",
    "run_main_startup_runtime",
    "schedule_run_all_trades_runtime",
    "schedule_run_all_trades_with_delay_runtime",
    "_run_trade_updates_stream",
]
