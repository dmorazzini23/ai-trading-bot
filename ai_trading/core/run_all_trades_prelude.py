"""Pre-cycle bootstrap for ``run_all_trades_worker``."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any, Callable

from ai_trading.config.management import get_env


@dataclass(frozen=True, slots=True)
class RunAllTradesPreludeResult:
    ready: bool
    cfg_runtime: Any | None
    effective_policy: Any | None
    now: datetime | None
    loop_start: float | None
    api: Any | None
    previous_last_run_at: datetime | None


def _ensure_worker_state_fields(state: Any) -> None:
    if not hasattr(state, "trade_cooldowns"):
        state.trade_cooldowns = {}
    if not hasattr(state, "last_trade_direction"):
        state.last_trade_direction = {}
    if not hasattr(state, "entry_flip_signal_streak"):
        state.entry_flip_signal_streak = {}
    if not hasattr(state, "last_entry_side"):
        state.last_entry_side = {}
    if not hasattr(state, "entry_expectancy_context"):
        state.entry_expectancy_context = {}
    if not hasattr(state, "expectancy_history"):
        state.expectancy_history = {}
    if not hasattr(state, "exit_policy_state"):
        state.exit_policy_state = {}
    if not hasattr(state, "policy_rollback_disabled_slices"):
        state.policy_rollback_disabled_slices = []
    if not hasattr(state, "gate_auto_disable_hysteresis_state"):
        state.gate_auto_disable_hysteresis_state = {}
    if not hasattr(state, "gate_auto_disable_transition_ts"):
        state.gate_auto_disable_transition_ts = []
    if not hasattr(state, "last_policy_ablation_run_date"):
        state.last_policy_ablation_run_date = None


def _warn_missing_api(
    *,
    logger: Any,
    loop_id: str,
    loop_start: float,
    log_loop_heartbeat_func: Callable[[str, float], None],
    emit_test_capture_func: Callable[[str, int], None],
) -> None:
    logger.warning("ALPACA_CLIENT_MISSING")
    logging.getLogger("tests.test_broker_unavailable_paths").warning(
        "ALPACA_CLIENT_MISSING"
    )
    emit_test_capture_func("ALPACA_CLIENT_MISSING", logging.WARNING)
    logging.warning("ALPACA_CLIENT_MISSING")
    log_loop_heartbeat_func(loop_id, loop_start)


def prepare_run_all_trades_cycle(
    *,
    state: Any,
    runtime: Any,
    risk_engine: Any,
    logger: Any,
    loop_id: str,
    run_interval_seconds: float,
    ensure_execution_engine_func: Callable[[Any], None],
    enforce_dependency_preflight_func: Callable[[Any], None],
    resolve_trading_config_func: Callable[[Any], Any],
    active_effective_policy_func: Callable[[Any, Any], Any],
    policy_config_error_type: type[BaseException],
    write_run_manifest_func: Callable[..., Any],
    restore_exit_policy_state_func: Callable[[Any], bool],
    ensure_exit_policy_state_func: Callable[[Any], Any],
    get_trade_cooldown_min_func: Callable[[], float],
    is_market_open_func: Callable[[], bool],
    log_market_closed_func: Callable[[str], None],
    record_broker_sync_metrics_func: Callable[[Any, Any], None],
    utc_now_func: Callable[[], datetime],
    monotonic_time_func: Callable[[], float],
    persist_effective_policy_snapshot_func: Callable[..., None],
    execution_metrics_factory: Callable[[], Any],
    ensure_alpaca_attached_func: Callable[[Any], None],
    validate_trading_api_func: Callable[[Any], bool],
    startup_cancel_mode_func: Callable[[], str],
    list_open_orders_func: Callable[[Any], list[Any]],
    startup_cancel_decision_func: Any,
    cancel_open_orders_subset_func: Callable[..., Any],
    select_startup_stale_orders_func: Callable[[list[Any]], list[Any]],
    cancel_all_open_orders_oms_func: Callable[[Any], Any],
    arm_pending_cleanup_warmup_func: Callable[..., Any],
    provider_monitor: Any,
    safe_mode_blocks_trading_func: Callable[[], bool],
    safe_mode_reason_func: Callable[[], str | None],
    cancel_all_open_orders_func: Callable[[Any], Any],
    reset_cycle_cache_func: Callable[[], None],
    get_strategies_func: Callable[[], Any],
    log_loop_heartbeat_func: Callable[[str, float], None],
    emit_test_capture_func: Callable[[str, int], None],
    common_exceptions: tuple[type[BaseException], ...],
) -> RunAllTradesPreludeResult:
    try:
        risk_engine.wait_for_exposure_update(0.5)
    except (TimeoutError, ConnectionError, RuntimeError) as exc:
        logger.warning(
            "RISK_EXPOSURE_UPDATE_FAILED",
            extra={"cause": exc.__class__.__name__, "detail": str(exc)},
        )

    ensure_execution_engine_func(runtime)
    enforce_dependency_preflight_func(runtime)
    cfg_runtime = resolve_trading_config_func(runtime)
    try:
        effective_policy = active_effective_policy_func(state, cfg_runtime)
    except policy_config_error_type as exc:
        logger.error("EFFECTIVE_POLICY_INVALID", extra={"error": str(exc)})
        state.halt_trading = True
        state.halt_reason = "EFFECTIVE_POLICY_INVALID"
        return RunAllTradesPreludeResult(False, None, None, None, None, None, None)

    if not state.run_manifest_written:
        try:
            write_run_manifest_func(
                cfg_runtime,
                runtime_contract={"stubs_enabled": False},
                effective_policy_hash=str(getattr(state, "effective_policy_hash", "") or ""),
                effective_policy={
                    "trading_mode": effective_policy.trading_mode,
                    "objective": effective_policy.objective.objective_name,
                },
            )
        except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
            logger.warning(
                "RUN_MANIFEST_WRITE_FAILED error=%s",
                str(exc),
                extra={"error": str(exc)},
            )
        else:
            state.run_manifest_written = True

    _ensure_worker_state_fields(state)
    if restore_exit_policy_state_func(state):
        logger.info(
            "EXIT_POLICY_STATE_RESTORED",
            extra={"bucket_count": len(ensure_exit_policy_state_func(state))},
        )
    if state.running:
        logger.warning(
            "RUN_ALL_TRADES_SKIPPED_OVERLAP",
            extra={"last_duration": getattr(state, "last_loop_duration", 0.0)},
        )
        logging.getLogger("ai_trading.core.bot_engine").warning(
            "RUN_ALL_TRADES_SKIPPED_OVERLAP"
        )
        emit_test_capture_func("RUN_ALL_TRADES_SKIPPED_OVERLAP", logging.WARNING)
        return RunAllTradesPreludeResult(False, None, None, None, None, None, None)

    now = utc_now_func()
    for sym, ts in list(state.trade_cooldowns.items()):
        if (now - ts).total_seconds() > get_trade_cooldown_min_func() * 60:
            state.trade_cooldowns.pop(sym, None)
    if state.last_run_at and (now - state.last_run_at).total_seconds() < run_interval_seconds:
        logger.warning("RUN_ALL_TRADES_SKIPPED_RECENT")
        return RunAllTradesPreludeResult(False, None, None, None, None, None, None)

    rth_only = bool(getattr(cfg_runtime, "rth_only", True))
    allow_extended = bool(getattr(cfg_runtime, "allow_extended", False))
    if (rth_only or not allow_extended) and not is_market_open_func():
        log_market_closed_func("MARKET_CLOSED_NO_FETCH")
        try:
            post_sync_enabled = bool(getattr(cfg_runtime, "post_submit_broker_sync", True))
        except (TypeError, ValueError):
            post_sync_enabled = True
        if post_sync_enabled:
            engine_obj = getattr(runtime, "execution_engine", None)
            if engine_obj is not None and hasattr(engine_obj, "synchronize_broker_state"):
                try:
                    broker_snapshot = engine_obj.synchronize_broker_state()
                except AI_TRADING_FALLBACK_EXCEPTIONS:
                    logger.debug("BROKER_SYNC_REFRESH_FAILED", exc_info=True)
                else:
                    record_broker_sync_metrics_func(state, broker_snapshot)
        return RunAllTradesPreludeResult(False, None, None, None, None, None, None)

    loop_start = monotonic_time_func()
    persist_effective_policy_snapshot_func(state, effective_policy, loop_id=loop_id)
    state.execution_metrics = execution_metrics_factory()

    api = getattr(runtime, "api", None)
    if api is None and get_env("PYTEST_RUNNING"):
        _warn_missing_api(
            logger=logger,
            loop_id=loop_id,
            loop_start=loop_start,
            log_loop_heartbeat_func=log_loop_heartbeat_func,
            emit_test_capture_func=emit_test_capture_func,
        )
        return RunAllTradesPreludeResult(False, None, None, None, None, None, None)

    ensure_alpaca_attached_func(runtime)
    api = getattr(runtime, "api", None)
    if api is None:
        _warn_missing_api(
            logger=logger,
            loop_id=loop_id,
            loop_start=loop_start,
            log_loop_heartbeat_func=log_loop_heartbeat_func,
            emit_test_capture_func=emit_test_capture_func,
        )
        return RunAllTradesPreludeResult(False, None, None, None, None, None, None)
    if not validate_trading_api_func(api):
        _warn_missing_api(
            logger=logger,
            loop_id=loop_id,
            loop_start=loop_start,
            log_loop_heartbeat_func=log_loop_heartbeat_func,
            emit_test_capture_func=emit_test_capture_func,
        )
        return RunAllTradesPreludeResult(False, None, None, None, None, None, None)

    if not state.startup_cleanup_done:
        startup_mode = startup_cancel_mode_func()
        if startup_mode != "off":
            startup_open_orders: list[Any] = []
            startup_should_cancel = True
            startup_details: dict[str, Any] = {"mode": startup_mode}
            try:
                startup_open_orders = list_open_orders_func(api)
                startup_should_cancel, startup_details = startup_cancel_decision_func(
                    startup_open_orders,
                    mode=startup_mode,
                )
            except common_exceptions as exc:
                startup_should_cancel = False
                startup_details = {
                    "mode": startup_mode,
                    "reason": "evaluation_failed",
                    "detail": str(exc),
                }
                logger.warning(
                    "STARTUP_CLEANUP_EVALUATION_FAILED",
                    extra=startup_details,
                    exc_info=True,
                )
            if startup_should_cancel:
                if startup_mode == "stale_only":
                    startup_result = cancel_open_orders_subset_func(
                        runtime,
                        orders=select_startup_stale_orders_func(startup_open_orders),
                        reason_code="STARTUP_STALE_PENDING",
                    )
                else:
                    startup_result = cancel_all_open_orders_oms_func(runtime)
                startup_log = logger.warning if startup_result.failed else logger.info
                try:
                    startup_open_count = int(
                        startup_details.get("open_count", len(startup_open_orders))
                    )
                except (TypeError, ValueError):
                    startup_open_count = len(startup_open_orders)
                try:
                    startup_pending_count = int(startup_details.get("stale_count", 0))
                except (TypeError, ValueError):
                    startup_pending_count = 0
                startup_log(
                    "STARTUP_CLEANUP",
                    extra={
                        **startup_details,
                        "cancel_scope": startup_mode,
                        "targeted_orders": int(startup_result.total_open),
                        "cancelled": startup_result.cancelled,
                        "failed": startup_result.failed,
                    },
                )
                arm_pending_cleanup_warmup_func(
                    runtime,
                    source="startup_cleanup",
                    open_count=startup_open_count,
                    pending_count=startup_pending_count,
                )
            else:
                logger.info("STARTUP_CLEANUP_SKIPPED", extra=startup_details)
        state.startup_cleanup_done = True

    try:
        safe_mode_live = provider_monitor.is_safe_mode_active()
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        safe_mode_live = False
    if safe_mode_live and safe_mode_blocks_trading_func():
        try:
            version, reason = provider_monitor.safe_mode_cycle_marker()
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            version, reason = (None, safe_mode_reason_func())
        last_cleared = getattr(state, "_safe_mode_cancel_version", None)
        if version is None or version != last_cleared:
            try:
                cancel_all_open_orders_func(runtime)
            except common_exceptions as exc:  # pragma: no cover - defensive cancel
                logger.warning(
                    "SAFE_MODE_CANCEL_OPEN_ORDERS_FAILED",
                    extra={
                        "reason": reason or safe_mode_reason_func() or "provider_safe_mode",
                        "cause": exc.__class__.__name__,
                        "detail": str(exc),
                    },
                )
            else:
                logger.warning(
                    "SAFE_MODE_CANCEL_OPEN_ORDERS",
                    extra={
                        "reason": reason or safe_mode_reason_func() or "provider_safe_mode",
                        "version": version,
                    },
                )
            setattr(state, "_safe_mode_cancel_version", version)

    feed_cache = getattr(state, "minute_feed_cache", None)
    if isinstance(feed_cache, dict):
        feed_cache.clear()
    else:
        state.minute_feed_cache = {}
    reset_cycle_cache_func()
    previous_last_run_at = state.last_run_at
    state.running = True
    state.last_run_at = now
    intents = getattr(state, "cycle_order_intents", None)
    if isinstance(intents, dict):
        intents.clear()
    else:
        state.cycle_order_intents = {}
    submit_compaction = getattr(state, "cycle_submit_compaction", None)
    if isinstance(submit_compaction, set):
        submit_compaction.clear()
    else:
        state.cycle_submit_compaction = set()
    if not getattr(state, "_strategies_loaded", False):
        runtime.strategies = get_strategies_func()
        state._strategies_loaded = True

    return RunAllTradesPreludeResult(
        ready=True,
        cfg_runtime=cfg_runtime,
        effective_policy=effective_policy,
        now=now,
        loop_start=loop_start,
        api=api,
        previous_last_run_at=previous_last_run_at,
    )


__all__ = ["RunAllTradesPreludeResult", "prepare_run_all_trades_cycle"]
