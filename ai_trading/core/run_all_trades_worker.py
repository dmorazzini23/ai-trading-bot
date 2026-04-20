"""Runtime worker orchestration extracted from ``bot_engine.py``."""

from __future__ import annotations

import importlib
import logging
import uuid
from typing import Any

from ai_trading.core.run_all_trades_execution import execute_run_all_trades_cycle
from ai_trading.core.run_all_trades_prelude import prepare_run_all_trades_cycle


def _log_overlap(be: Any) -> None:
    be.logger.info("RUN_ALL_TRADES_SKIPPED_OVERLAP")
    logging.getLogger("ai_trading.core.bot_engine").info(
        "RUN_ALL_TRADES_SKIPPED_OVERLAP"
    )
    be._emit_pytest_capture(logging.INFO, "RUN_ALL_TRADES_SKIPPED_OVERLAP")


def _restore_last_run_timestamp(state: Any, previous_last_run_at: Any) -> None:
    state.last_run_at = previous_last_run_at


def _log_finalizer_hook_failure(be: Any, *, hook: str, exc: BaseException) -> None:
    be.logger.warning(
        "RUN_ALL_TRADES_FINALIZER_HOOK_FAILED",
        extra={
            "hook": hook,
            "cause": exc.__class__.__name__,
            "detail": str(exc),
        },
        exc_info=True,
    )


def _run_finalizer_hook(
    be: Any,
    hook_name: str,
    func: Any,
    /,
    *args: Any,
    **kwargs: Any,
) -> None:
    try:
        func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive boundary
        _log_finalizer_hook_failure(be, hook=hook_name, exc=exc)


def _observe_cycle_execute_latency(be: Any, execution_stage_start: float | None) -> None:
    if execution_stage_start is None:
        return
    try:
        be.record_cycle_wall(
            max(0.0, be.monotonic_time() - execution_stage_start),
            {"stage": "cycle_execute"},
        )
    except Exception:
        be.logger.debug("CYCLE_EXECUTE_LATENCY_OBSERVE_FAILED", exc_info=True)


def _resolve_stale_ratio(be: Any) -> float:
    try:
        cfg = be.get_trading_config()
        return float(getattr(cfg, "execution_stale_ratio_shadow", 0.30))
    except be.COMMON_EXC:
        return 0.30


def _log_cycle_gates(be: Any) -> None:
    be.logger.info(
        "CYCLE_GATES",
        extra={
            "shadow": be.guard_shadow_active(),
            "stale": getattr(be.EXEC_GUARD_STATE, "stale_symbols", "na"),
            "universe": getattr(be.EXEC_GUARD_STATE, "universe_size", "na"),
        },
    )


def _finalize_run_all_trades_cycle(
    *,
    be: Any,
    state: Any,
    runtime: Any,
    loop_id: str,
    loop_start: float,
    execution_stage_start: float | None,
) -> None:
    _observe_cycle_execute_latency(be, execution_stage_start)

    try:
        last_loop_duration = be.monotonic_time() - loop_start
    except Exception as exc:  # pragma: no cover - defensive boundary
        _log_finalizer_hook_failure(be, hook="last_loop_duration", exc=exc)
        last_loop_duration = getattr(state, "last_loop_duration", 0.0)

    state.running = False
    state._strategies_loaded = False
    state.last_loop_duration = last_loop_duration

    stale_ratio = _resolve_stale_ratio(be)
    _run_finalizer_hook(
        be,
        "guard_end_cycle",
        be.guard_end_cycle,
        stale_threshold_ratio=stale_ratio,
    )
    _run_finalizer_hook(be, "cycle_gates_log", _log_cycle_gates, be)
    _run_finalizer_hook(
        be,
        "loop_heartbeat",
        be._log_loop_heartbeat,
        loop_id,
        loop_start,
    )
    _run_finalizer_hook(
        be,
        "flush_log_throttle_summaries",
        be.flush_log_throttle_summaries,
    )
    _run_finalizer_hook(be, "runtime_stop_check", be._check_runtime_stops, runtime)

    if not be.MEMORY_OPTIMIZATION_AVAILABLE:
        return
    try:
        gc_result = be.optimize_memory()
        if gc_result.get("objects_collected", 0) > 50:
            be.logger.info(
                f"Post-cycle GC: {gc_result['objects_collected']} objects collected"
            )
    except (RuntimeError, ValueError, TypeError) as exc:
        be.logger.warning(
            "MEMORY_OPTIMIZATION_FAILED",
            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
        )


def run_all_trades_worker_cycle(state: Any, runtime: Any) -> None:
    """Execute the main all-symbol worker cycle outside ``bot_engine.py``."""

    be = importlib.import_module("ai_trading.core.bot_engine")
    be._ensure_alpaca_classes()
    if be._ALPACA_IMPORT_ERROR is not None:
        raise RuntimeError("Alpaca SDK is required") from be._ALPACA_IMPORT_ERROR

    risk_engine = getattr(runtime, "risk_engine", None)
    if risk_engine is None:
        be.logger.error("RISK_ENGINE_MISSING")
        return

    be._init_metrics()
    loop_id = str(uuid.uuid4())
    execution_stage_start: float | None = None
    acquired = be.run_lock.acquire(blocking=False)
    if not acquired:
        _log_overlap(be)
        return

    try:
        prelude = prepare_run_all_trades_cycle(
            state=state,
            runtime=runtime,
            risk_engine=risk_engine,
            logger=be.logger,
            loop_id=loop_id,
            run_interval_seconds=be.RUN_INTERVAL_SECONDS,
            ensure_execution_engine_func=be._ensure_execution_engine,
            enforce_dependency_preflight_func=be._enforce_dependency_preflight,
            resolve_trading_config_func=be._resolve_trading_config,
            active_effective_policy_func=be._active_effective_policy,
            policy_config_error_type=be.PolicyConfigError,
            write_run_manifest_func=be.write_run_manifest,
            restore_exit_policy_state_func=be._restore_exit_policy_state,
            ensure_exit_policy_state_func=be._ensure_exit_policy_state,
            get_trade_cooldown_min_func=be.get_trade_cooldown_min,
            is_market_open_func=be.is_market_open,
            log_market_closed_func=be._log_market_closed,
            record_broker_sync_metrics_func=be._record_broker_sync_metrics,
            utc_now_func=lambda: be.datetime.now(be.UTC),
            monotonic_time_func=be.monotonic_time,
            persist_effective_policy_snapshot_func=be._persist_effective_policy_snapshot,
            execution_metrics_factory=be.ExecutionCycleMetrics,
            ensure_alpaca_attached_func=be.ensure_alpaca_attached,
            validate_trading_api_func=be._validate_trading_api,
            startup_cancel_mode_func=be._startup_cancel_mode,
            list_open_orders_func=be.list_open_orders,
            startup_cancel_decision_func=be._startup_cancel_decision,
            cancel_open_orders_subset_func=be._cancel_open_orders_subset,
            select_startup_stale_orders_func=be._select_startup_stale_orders,
            cancel_all_open_orders_oms_func=be.cancel_all_open_orders_oms,
            arm_pending_cleanup_warmup_func=be._arm_pending_cleanup_warmup,
            provider_monitor=be.provider_monitor,
            safe_mode_blocks_trading_func=be._safe_mode_blocks_trading,
            safe_mode_reason_func=be.safe_mode_reason,
            cancel_all_open_orders_func=be.cancel_all_open_orders,
            reset_cycle_cache_func=be._reset_cycle_cache,
            get_strategies_func=be.get_strategies,
            log_loop_heartbeat_func=be._log_loop_heartbeat,
            emit_test_capture_func=be._emit_test_capture,
            common_exceptions=be.COMMON_EXC,
        )
        if not prelude.ready:
            return

        cfg_runtime = prelude.cfg_runtime
        loop_start = prelude.loop_start
        api = prelude.api

        assert cfg_runtime is not None
        assert loop_start is not None
        assert api is not None

        try:
            execution_stage_start = be.monotonic_time()
            execute_run_all_trades_cycle(
                state=state,
                runtime=runtime,
                cfg_runtime=cfg_runtime,
                loop_id=loop_id,
                loop_start=loop_start,
                api=api,
                restore_last_run_timestamp=lambda: _restore_last_run_timestamp(
                    state,
                    prelude.previous_last_run_at,
                ),
            )
        except be.APIError as exc:
            be.logger.warning(
                "TRADING_CYCLE_API_ERROR",
                extra={"cause": exc.__class__.__name__, "detail": str(exc)},
            )
            _restore_last_run_timestamp(state, prelude.previous_last_run_at)
            return
        except (
            TimeoutError,
            ConnectionError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            be.logger.error(
                "TRADING_CYCLE_FAILED",
                extra={"cause": exc.__class__.__name__, "detail": str(exc)},
                exc_info=True,
            )
            _restore_last_run_timestamp(state, prelude.previous_last_run_at)
            raise
        finally:
            _finalize_run_all_trades_cycle(
                be=be,
                state=state,
                runtime=runtime,
                loop_id=loop_id,
                loop_start=loop_start,
                execution_stage_start=execution_stage_start,
            )
    finally:
        if acquired:
            be.run_lock.release()


__all__ = ["run_all_trades_worker_cycle"]
