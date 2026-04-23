"""Execute the main run-all-trades worker cycle outside ``bot_engine.py``."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import importlib
import time
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.core.legacy_decision_journal import LegacyDecisionJournalRecorder


def _evaluate_replay_live_parity_gate(*, state: Any, runtime: Any, be: Any) -> dict[str, Any]:
    default_required = not bool(
        str(get_env("PYTEST_CURRENT_TEST", "", cast=str) or "").strip()
        or bool(get_env("PYTEST_RUNNING", False, cast=bool))
    )
    enabled = bool(
        get_env("AI_TRADING_REPLAY_LIVE_PARITY_GATE_ENABLED", True, cast=bool)
    )
    required = bool(
        get_env(
            "AI_TRADING_RUNTIME_GONOGO_REQUIRE_REPLAY_LIVE_PARITY_GATE",
            default_required,
            cast=bool,
        )
    )
    if not enabled and not required:
        return {"enabled": False, "available": False, "ok": True, "status": "disabled"}
    try:
        from ai_trading.governance.replay_live_parity import summarize_replay_live_parity_gate
        from ai_trading.tools.runtime_performance_report import summarize_oms_lifecycle_parity

        oms_lifecycle_parity = summarize_oms_lifecycle_parity()
        gate = summarize_replay_live_parity_gate(
            oms_lifecycle_parity=oms_lifecycle_parity,
        )
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        gate = {
            "enabled": True,
            "available": False,
            "ok": False,
            "status": "fail",
            "reason": "replay_live_parity_gate_error",
            "error": str(exc),
        }
    setattr(state, "replay_live_parity_gate", gate)
    setattr(runtime, "replay_live_parity_gate", gate)
    if required and gate.get("enabled", False) and not bool(gate.get("ok")):
        be.logger.warning(
            "REPLAY_LIVE_PARITY_GATE_BLOCKED",
            extra={
                "reason": gate.get("reason"),
                "failed_checks": gate.get("failed_checks", []),
            },
        )
        try:
            be.runtime_state.update_service_status(
                status="degraded",
                reason="replay_live_parity_gate_failed",
            )
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            be.logger.debug("REPLAY_PARITY_RUNTIME_STATE_UPDATE_FAILED", exc_info=True)
    return gate


def execute_run_all_trades_cycle(
    *,
    state: Any,
    runtime: Any,
    cfg_runtime: Any,
    loop_id: str,
    loop_start: float,
    api: Any,
    restore_last_run_timestamp: Any,
) -> None:
    """Execute the main worker cycle after bootstrap has completed."""
    be = importlib.import_module("ai_trading.core.bot_engine")
    default_required = not bool(
        str(get_env("PYTEST_CURRENT_TEST", "", cast=str) or "").strip()
        or bool(get_env("PYTEST_RUNNING", False, cast=bool))
    )
    parity_gate = _evaluate_replay_live_parity_gate(state=state, runtime=runtime, be=be)
    require_replay_live_parity_gate = bool(
        get_env(
            "AI_TRADING_RUNTIME_GONOGO_REQUIRE_REPLAY_LIVE_PARITY_GATE",
            default_required,
            cast=bool,
        )
    )
    if require_replay_live_parity_gate and parity_gate.get("enabled", False) and not bool(
        parity_gate.get("ok")
    ):
        return

    legacy_recorder = LegacyDecisionJournalRecorder(
        path=str(be._decision_log_runtime_path()),
        write_impl=be._write_decision_record,
    )
    setattr(state, "_legacy_decision_recorder", legacy_recorder)
    try:
        can_list_orders = hasattr(api, "list_orders") and callable(
            getattr(api, "list_orders", None)
        )
        if not can_list_orders:
            if not getattr(state, "_warned_missing_list_orders", False):
                be.logger.warning("API capability unavailable: list_orders")
                setattr(state, "_warned_missing_list_orders", True)
            open_orders = []
        else:
            try:
                open_orders = be.list_open_orders(api)
            except (
                be.APIError,
                TimeoutError,
                ConnectionError,
                AttributeError,
            ) as e:
                be.logger.warning(
                    "api.list_orders failed during order check",
                    extra={"cause": e.__class__.__name__, "detail": str(e)},
                )
                breaker_info = be.classify_exception(e, dependency="broker_open_orders")
                breakers = be._dependency_breakers(state)
                breakers.record_failure("broker_open_orders", breaker_info)
                be._handle_error(breaker_info, state=state, ctx=runtime)
                open_orders = []
        pending_skip_cycle = be._handle_pending_orders(open_orders, runtime)
        if pending_skip_cycle:
            blocked_symbols_raw = getattr(runtime, be._PENDING_ORDER_BLOCKED_SYMBOLS_ATTR, ())
            blocked_symbols = {
                str(sym).strip().upper()
                for sym in blocked_symbols_raw
                if str(sym).strip()
            }
            if be._pending_orders_block_scope() == "symbol" and blocked_symbols:
                blocked_symbols_sample = sorted(blocked_symbols)[: be._PENDING_ORDER_SAMPLE_LIMIT]
                blocked_payload = {
                    "blocked_symbols_count": len(blocked_symbols),
                    "blocked_symbols": blocked_symbols_sample,
                }
                blocked_ttl_s = be._resolve_runtime_info_log_ttl_seconds(
                    "AI_TRADING_PENDING_SYMBOL_BLOCK_ACTIVE_LOG_TTL_SEC",
                    be._PENDING_SYMBOL_BLOCK_ACTIVE_LOG_TTL_SEC_DEFAULT,
                )
                blocked_signature = (
                    f"{blocked_payload['blocked_symbols_count']}:"
                    f"{','.join(blocked_symbols_sample[:3])}"
                )
                if be._should_emit_runtime_info_log(
                    runtime,
                    f"PENDING_ORDERS_SYMBOL_BLOCK_ACTIVE:{blocked_signature}",
                    ttl_seconds=blocked_ttl_s,
                ):
                    be.log_throttled_event(
                        be.logger,
                        "PENDING_ORDERS_SYMBOL_BLOCK_ACTIVE",
                        level=be.logging.INFO,
                        message="PENDING_ORDERS_SYMBOL_BLOCK_ACTIVE",
                        extra=blocked_payload,
                    )
            else:
                return

        if be._netting_pipeline_enabled(runtime):
            cycle_engine = getattr(runtime, "execution_engine", None)
            cycle_started = False
            broker_snapshot = None
            if cycle_engine is not None:
                start_hook = getattr(cycle_engine, "start_cycle", None)
                if callable(start_hook):
                    try:
                        start_hook()
                        cycle_started = True
                    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
                        be.logger.warning(
                            "EXECUTION_START_CYCLE_FAILED",
                            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
                            exc_info=True,
                        )
            try:
                be._run_netting_cycle(state, runtime, loop_id, loop_start)
                provider_status_after = ""
                provider_using_backup = False
                try:
                    provider_state_after = be.runtime_state.observe_data_provider_state()
                except AI_TRADING_FALLBACK_EXCEPTIONS:
                    provider_state_after = {}
                if isinstance(provider_state_after, dict):
                    try:
                        provider_status_after = str(
                            provider_state_after.get("status") or ""
                        ).strip().lower()
                    except AI_TRADING_FALLBACK_EXCEPTIONS:
                        provider_status_after = ""
                    provider_using_backup = bool(provider_state_after.get("using_backup"))
                    timeframe_state_after = provider_state_after.get("timeframes")
                    if isinstance(timeframe_state_after, be.Mapping):
                        minute_backup = timeframe_state_after.get("1Min")
                        if isinstance(minute_backup, bool):
                            provider_using_backup = minute_backup
                if provider_status_after not in {
                    "down",
                    "disabled",
                    "failed",
                    "unreachable",
                    "error",
                }:
                    be.runtime_state.update_data_provider_state(
                        status="degraded" if provider_using_backup else "healthy",
                        data_status="ready",
                        reason=(
                            "data_available_via_backup"
                            if provider_using_backup
                            else "data_available_netting"
                        ),
                        safe_mode=be.is_safe_mode_active(),
                    )
                post_sync_enabled = True
                try:
                    post_sync_enabled = bool(
                        getattr(cfg_runtime, "post_submit_broker_sync", True)
                    )
                except (TypeError, ValueError):
                    post_sync_enabled = True
                if post_sync_enabled:
                    engine_for_sync = getattr(runtime, "execution_engine", None) or cycle_engine
                    if engine_for_sync is not None and hasattr(
                        engine_for_sync, "synchronize_broker_state"
                    ):
                        try:
                            broker_snapshot = engine_for_sync.synchronize_broker_state()
                        except AI_TRADING_FALLBACK_EXCEPTIONS:
                            be.logger.debug("BROKER_SYNC_REFRESH_FAILED", exc_info=True)
            finally:
                if cycle_started and cycle_engine is not None:
                    end_hook = getattr(cycle_engine, "end_cycle", None)
                    if callable(end_hook):
                        try:
                            end_hook()
                        except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
                            be.logger.warning(
                                "EXECUTION_END_CYCLE_FAILED",
                                extra={"cause": exc.__class__.__name__, "detail": str(exc)},
                                exc_info=True,
                            )
            if broker_snapshot is not None:
                be._record_broker_sync_metrics(state, broker_snapshot)
            return

        if be.get_verbose_logging():
            be.logger.info(
                "RUN_ALL_TRADES_START",
                extra={"timestamp": be.utc_now_iso()},
            )
        if bool(getattr(be.CFG, "log_market_fetch", True)):
            be.logger.info("MARKET_FETCH")
        else:
            be.logger.debug("MARKET_FETCH")

        ctx = be._get_runtime_context_or_none()
        if ctx and getattr(runtime, "data_fetcher", None) is None:
            runtime.data_fetcher = getattr(ctx, "data_fetcher", None)
        be.ensure_data_fetcher(runtime)
        be.get_trade_logger()
        manager = getattr(runtime, "signal_manager", None)
        if manager is None and ctx is not None:
            manager = getattr(ctx, "signal_manager", None)
        if manager is None:
            manager = be.signal_manager
        if hasattr(manager, "begin_cycle"):
            manager.begin_cycle()

        for attempt in range(3):
            try:
                current_cash, regime_ok, symbols = be._prepare_run(
                    runtime,
                    state,
                    getattr(runtime, "base_universe_tickers", None),
                )
                break
            except be.DataFetchError as e:
                be.logger.warning(
                    "DATA_FETCHER_UNAVAILABLE",
                    extra={"detail": str(e), "attempt": attempt + 1},
                )
                if attempt == 2:
                    restore_last_run_timestamp()
                    return
                time.sleep(1.0)
            except be.APIError as e:
                be.logger.warning(
                    "PREPARE_RUN_API_ERROR",
                    extra={"detail": str(e), "attempt": attempt + 1},
                )
                if attempt == 2:
                    restore_last_run_timestamp()
                    return
                time.sleep(1.0)

        if be.MEMORY_OPTIMIZATION_AVAILABLE:
            try:
                memory_stats = be.optimize_memory()
                if memory_stats.get("memory_usage_mb", 0) > 512:
                    be.logger.warning(
                        "HIGH_MEMORY_USAGE_DETECTED",
                        extra={
                            "memory_usage_mb": memory_stats.get("memory_usage_mb", 0),
                            "symbols_count": len(symbols),
                        },
                    )
                    if memory_stats.get("memory_usage_mb", 0) > 1024:
                        be.logger.critical("EMERGENCY_MEMORY_CLEANUP_TRIGGERED")
                        be.emergency_memory_cleanup()
            except (RuntimeError, ValueError, TypeError) as e:
                be.logger.debug(
                    "MEMORY_OPTIMIZATION_FAILED",
                    extra={"cause": e.__class__.__name__, "detail": str(e)},
                )

        get_account = getattr(api, "get_account", None)
        can_fetch_account = callable(get_account)
        account_snapshot = (
            be.safe_alpaca_get_account(runtime) if can_fetch_account else None
        )
        dbc = getattr(runtime, "drawdown_circuit_breaker", None)
        if dbc and can_fetch_account:
            try:
                acct = account_snapshot
                current_equity = float(getattr(acct, "equity", 0.0)) if acct else 0.0
                trading_allowed = dbc.update_equity(current_equity)
                status = dbc.get_status()
                if not trading_allowed:
                    be.logger.critical(
                        "TRADING_HALTED_DRAWDOWN_PROTECTION",
                        extra={
                            "current_drawdown": status["current_drawdown"],
                            "max_drawdown": status["max_drawdown"],
                            "peak_equity": status["peak_equity"],
                            "current_equity": current_equity,
                        },
                    )
                    try:
                        portfolio = runtime.api.list_positions()
                        for pos in portfolio:
                            be.manage_position_risk(runtime, pos)
                    except (be.APIError, TimeoutError, ConnectionError) as e:
                        be.logger.warning(
                            "HALT_MANAGE_FAIL",
                            extra={"cause": e.__class__.__name__, "detail": str(e)},
                        )
                    return
                be.logger.debug(
                    "DRAWDOWN_STATUS_OK",
                    extra={
                        "current_drawdown": status["current_drawdown"],
                        "max_drawdown": status["max_drawdown"],
                        "trading_allowed": status["trading_allowed"],
                    },
                )
            except (be.APIError, TimeoutError, ConnectionError) as e:
                be.logger.error(
                    "DRAWDOWN_CHECK_FAILED",
                    extra={"cause": e.__class__.__name__, "detail": str(e)},
                )

        if be.check_halt_flag(runtime):
            be._log_health_diagnostics(runtime, "halt_flag_loop")
            be.logger.info("TRADING_HALTED_VIA_FLAG: Managing existing positions only.")
            try:
                portfolio = runtime.api.list_positions()
                for pos in portfolio:
                    be.manage_position_risk(runtime, pos)
            except (be.APIError, TimeoutError, ConnectionError) as e:
                be.logger.warning(
                    "HALT_MANAGE_FAIL",
                    extra={"cause": e.__class__.__name__, "detail": str(e)},
                )
            return

        be._maybe_hot_reload_runtime_models(runtime)
        alpha_model = getattr(runtime, "model", None)
        if not alpha_model:
            be.logger.warning(
                "ALPHA_MODEL_UNAVAILABLE - skipping compute stage for this cycle"
            )
            return

        if not symbols:
            be.logger_once.warning(
                "RUN_ALL_TRADES_NO_SYMBOLS",
                key="run_all_trades_no_symbols_cycle",
            )
            be.logger.info("SKIP_MINUTE_FETCH", extra={"reason": "no_symbols"})
            time.sleep(1.0)
            return

        try:
            provider_state = be.runtime_state.observe_data_provider_state()
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            provider_state = {}
        data_status = None
        try:
            data_status = str(provider_state.get("data_status") or "").strip().lower()
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            data_status = None
        if data_status in {"empty", "degraded"} and be.is_safe_mode_active():
            be.logger.warning(
                "DATA_STATUS_EMPTY_SHORT_CIRCUIT",
                extra={"data_status": data_status or "unknown"},
            )
            be.runtime_state.update_service_status(
                status="degraded",
                reason=data_status or "data_empty",
            )
            time.sleep(1.0)
            return

        base_attempts, base_delay = be._resolve_data_retry_settings()
        attempts_limit = max(1, base_attempts)
        retry_delay = base_delay
        prefer_backup_quotes = bool(getattr(state, "prefer_backup_quotes", False))
        primary_disabled = False
        primary_provider_fn = getattr(
            be.data_fetcher_module,
            "is_primary_provider_enabled",
            None,
        )
        if callable(primary_provider_fn):
            try:
                primary_disabled = not bool(primary_provider_fn())
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                primary_disabled = False
        if primary_disabled:
            be.runtime_state.update_data_provider_state(
                status="degraded",
                reason="primary_provider_disabled",
                data_status="degraded",
                safe_mode=be.is_safe_mode_active(),
            )
        if primary_disabled and be.is_safe_mode_active():
            be.logger.warning(
                "SAFE_MODE_DATA_SKIP",
                extra={"reason": be.safe_mode_reason() or "provider_disabled"},
            )
            be.runtime_state.update_service_status(status="degraded", reason="data_empty")
            time.sleep(1.0)
            return
        attempts_limit, retry_delay, short_circuit_reason = be._short_circuit_retry_budget(
            prefer_backup=prefer_backup_quotes,
            primary_disabled=primary_disabled,
            attempts=attempts_limit,
            delay=retry_delay,
        )
        if short_circuit_reason:
            be.logger.info(
                "DATA_SOURCE_RETRY_BYPASS",
                extra={"reason": short_circuit_reason, "attempts": attempts_limit},
            )
        if be._warmup_data_only_mode_active() and attempts_limit > 1:
            attempts_limit = 1
            retry_delay = 0.0
            be.logger.info(
                "WARMUP_DATA_ONLY_RETRY_BYPASS",
                extra={"attempts": attempts_limit},
            )

        processed: list[str] = []
        row_counts: dict[str, int] = {}
        attempts_used = 0
        fetch_attempts_total = 0
        for attempt in range(attempts_limit):
            attempts_used = attempt + 1
            try:
                with be.StageTimer(be.logger, "CYCLE_DATA_MS", symbols=len(symbols)):
                    processed, row_counts, fetch_attempts = be._process_symbols(
                        symbols,
                        current_cash,
                        alpha_model,
                        regime_ok,
                    )
                    fetch_attempts_total += fetch_attempts
            except be.CycleAbortSafeMode as exc:
                be.logger.warning(
                    "CYCLE_EARLY_EXIT_SAFE_MODE",
                    extra={
                        "reason": str(exc)
                        or (be.safe_mode_reason() or "provider_safe_mode")
                    },
                )
                return
            if processed:
                if attempt:
                    be.logger.info(
                        "DATA_SOURCE_RETRY_SUCCESS",
                        extra={"attempt": attempts_used, "symbols": symbols},
                    )
                break
            if attempt < attempts_limit - 1 and retry_delay > 0.0:
                time.sleep(retry_delay)

        try:
            cfg_for_summary = be.get_trading_config()
        except be.COMMON_EXC:
            cfg_for_summary = None

        def _sync_broker_snapshot_if_enabled() -> Any | None:
            post_sync_enabled = True
            if cfg_for_summary is not None:
                try:
                    post_sync_enabled = bool(
                        getattr(cfg_for_summary, "post_submit_broker_sync", True)
                    )
                except (TypeError, ValueError):
                    post_sync_enabled = True
            if not post_sync_enabled:
                return None
            engine_obj = getattr(runtime, "execution_engine", None)
            if engine_obj is None or not hasattr(engine_obj, "synchronize_broker_state"):
                return None
            try:
                return engine_obj.synchronize_broker_state()
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                be.logger.debug("BROKER_SYNC_REFRESH_FAILED", exc_info=True)
                return None

        if fetch_attempts_total == 0:
            be.logger.info(
                "CYCLE_DATA_SKIP_NO_FETCH",
                extra={"symbols": symbols, "attempts": attempts_used or attempts_limit},
            )
            broker_snapshot = _sync_broker_snapshot_if_enabled()
            if broker_snapshot is not None:
                be._record_broker_sync_metrics(state, broker_snapshot)
            return

        if sum(row_counts.values()) == 0:
            last_ts = None
            for sym in symbols:
                ts = runtime.data_fetcher._minute_timestamps.get(sym)
                if last_ts is None or (ts and ts > last_ts):
                    last_ts = ts
            be.logger.critical(
                "DATA_SOURCE_EMPTY",
                extra={
                    "symbols": symbols,
                    "endpoint": "minute",
                    "last_success": last_ts.isoformat() if last_ts else "unknown",
                    "row_counts": row_counts,
                },
            )
            be.logger.info(
                "DATA_SOURCE_RETRY_FAILED",
                extra={"attempts": attempts_used or attempts_limit, "symbols": symbols},
            )
            state.skipped_cycles += 1
            be.runtime_state.update_data_provider_state(
                status="degraded",
                reason="data_source_empty",
                data_status="empty",
                safe_mode=be.is_safe_mode_active(),
            )
            be.runtime_state.update_service_status(
                status="degraded",
                reason="data_source_empty",
            )
            broker_snapshot = _sync_broker_snapshot_if_enabled()
            if broker_snapshot is not None:
                be._record_broker_sync_metrics(state, broker_snapshot)
            return

        zero_row_symbols = [s for s in symbols if row_counts.get(s, 0) == 0]
        skipped = [s for s in symbols if s not in processed]
        if attempts_used == 0:
            attempts_used = attempts_limit
        success = not skipped and not zero_row_symbols
        if attempts_used > 1 or skipped or zero_row_symbols:
            be.logger.info(
                "DATA_SOURCE_RETRY_FINAL",
                extra={"success": success, "attempts": attempts_used},
            )

        provider_using_backup = False
        try:
            provider_state_post = be.runtime_state.observe_data_provider_state()
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            provider_state_post = {}
        if isinstance(provider_state_post, dict):
            provider_using_backup = bool(provider_state_post.get("using_backup"))
            timeframe_state_post = provider_state_post.get("timeframes")
            if isinstance(timeframe_state_post, be.Mapping):
                minute_backup = timeframe_state_post.get("1Min")
                if isinstance(minute_backup, bool):
                    provider_using_backup = minute_backup
        be.runtime_state.update_data_provider_state(
            status="degraded" if provider_using_backup else "healthy",
            data_status="ready",
            reason=(
                "data_available_via_backup"
                if provider_using_backup
                else "data_available"
            ),
            safe_mode=be.is_safe_mode_active(),
        )

        if skipped:
            be.logger.info(
                "CYCLE_SKIPPED_SUMMARY",
                extra={"count": len(skipped), "symbols": skipped},
            )
            if len(skipped) == len(symbols):
                state.skipped_cycles += 1
            else:
                state.skipped_cycles = 0
        else:
            state.skipped_cycles = 0
        if state.skipped_cycles >= 2:
            be.logger.critical(
                "ALL_SYMBOLS_SKIPPED_TWO_CYCLES",
                extra={
                    "hint": "Check data provider API keys and entitlements; test data fetch manually from the server; review data fetcher logs",
                },
            )

        be.run_multi_strategy(runtime)
        broker_snapshot = _sync_broker_snapshot_if_enabled()
        if broker_snapshot is not None:
            be._record_broker_sync_metrics(state, broker_snapshot)
        try:
            be.get_risk_engine().refresh_positions(runtime.api)
            pos_list = runtime.api.list_positions()
            state.position_cache = {p.symbol: int(p.qty) for p in pos_list}
            state.long_positions = {s for s, q in state.position_cache.items() if q > 0}
            state.short_positions = {s for s, q in state.position_cache.items() if q < 0}
            try:
                state.execution_metrics.positions = len(pos_list)
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                be.logger.debug(
                    "EXECUTION_METRIC_POSITIONS_ASSIGN_FAILED",
                    exc_info=True,
                )
            if runtime.execution_engine:
                trailing_hook = getattr(runtime.execution_engine, "check_trailing_stops", None)
                if callable(trailing_hook):
                    try:
                        trailing_hook()
                    except (ValueError, TypeError) as e:
                        be.logger.info(
                            "TRAILING_STOP_CHECK_SUPPRESSED",
                            extra={
                                "cause": e.__class__.__name__,
                                "detail": str(e),
                            },
                        )
        except (be.APIError, TimeoutError, ConnectionError) as e:
            be.logger.warning(
                "REFRESH_POSITIONS_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
        be.logger.info(
            "RUN_ALL_TRADES_COMPLETE | processed=%s symbols, total_rows=%s",
            len(row_counts),
            sum(row_counts.values()),
        )
        try:
            acct = runtime.api.get_account()
            cash = float(acct.cash)
            equity = float(acct.equity)
            positions = runtime.api.list_positions()
            be.logger.debug("Raw Alpaca positions: %s", positions)
            from ai_trading import portfolio
            from ai_trading.utils import portfolio_lock

            try:
                with portfolio_lock:
                    runtime.portfolio_weights = portfolio.compute_portfolio_weights(
                        runtime,
                        [p.symbol for p in positions],
                    )
            except (ZeroDivisionError, ValueError, KeyError) as e:
                be.logger.warning(
                    "WEIGHT_RECOMPUTE_FAILED",
                    extra={"cause": e.__class__.__name__, "detail": str(e)},
                    exc_info=True,
                )
            exposure = (
                sum(abs(float(p.market_value)) for p in positions) / equity * 100
                if equity > 0
                else 0.0
            )
            try:
                state.execution_metrics.exposure_pct = float(exposure)
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                be.logger.debug("EXECUTION_METRIC_EXPOSURE_ASSIGN_FAILED", exc_info=True)
            provider_mode = "alpaca"
            try:
                if getattr(state, "prefer_backup_quotes", False) or be.provider_monitor.is_disabled("alpaca"):
                    provider_mode = "backup/synthetic"
                elif be.provider_monitor.is_disabled("alpaca_sip"):
                    provider_mode = "backup/synthetic"
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                if getattr(state, "prefer_backup_quotes", False):
                    provider_mode = "backup/synthetic"
            state.execution_metrics.provider_mode = provider_mode
            be.logger.info(
                "Portfolio summary: cash=$%.2f, equity=$%.2f, exposure=%.2f%%, positions=%d",
                cash,
                equity,
                exposure,
                len(positions),
            )
            be.logger.info(
                "POSITIONS_DETAIL",
                extra={
                    "positions": [
                        {
                            "symbol": p.symbol,
                            "qty": int(p.qty),
                            "avg_price": float(p.avg_entry_price),
                            "market_value": float(p.market_value),
                        }
                        for p in positions
                    ],
                },
            )
            be.logger.info(
                "WEIGHTS_VS_POSITIONS",
                extra={
                    "weights": runtime.portfolio_weights,
                    "positions": {p.symbol: int(p.qty) for p in positions},
                    "cash": cash,
                },
            )
            try:
                adaptive_cap = be.get_risk_engine()._adaptive_global_cap()
            except (ZeroDivisionError, ValueError, KeyError) as e:
                be.logger.warning(
                    "ADAPTIVE_CAP_FAILED",
                    extra={"cause": e.__class__.__name__, "detail": str(e)},
                )
                adaptive_cap = 0.0
            be.logger.info(
                "CYCLE SUMMARY: cash=$%.0f equity=$%.0f exposure=%.0f%% positions=%d adaptive_cap=%.1f",
                cash,
                equity,
                exposure,
                len(positions),
                adaptive_cap,
            )
            log_exec_summary = True
            if cfg_for_summary is not None:
                try:
                    log_exec_summary = bool(
                        getattr(cfg_for_summary, "log_exec_summary_enabled", True)
                    )
                except (TypeError, ValueError):
                    log_exec_summary = True
            if log_exec_summary:
                be._log_execution_summary(state.execution_metrics)
        except (be.APIError, TimeoutError, ConnectionError) as e:
            be.logger.warning(
                "SUMMARY_FAIL",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
        try:
            acct = runtime.api.get_account()
            last_equity = getattr(acct, "last_equity", acct.equity)
            pnl = float(acct.equity) - float(last_equity)
            be.logger.info(
                "LOOP_PNL",
                extra={
                    "loop_id": loop_id,
                    "pnl": pnl,
                    "mode": "SHADOW" if be.CFG.shadow_mode else "LIVE",
                },
            )
        except (be.APIError, TimeoutError, ConnectionError, ValueError) as e:
            be.logger.warning(
                "PNL_RETRIEVAL_FAILED",
                extra={"cause": e.__class__.__name__, "detail": str(e)},
            )
    finally:
        if getattr(state, "_legacy_decision_recorder", None) is legacy_recorder:
            delattr(state, "_legacy_decision_recorder")


__all__ = ["execute_run_all_trades_cycle"]
