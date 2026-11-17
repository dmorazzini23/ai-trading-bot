"""Execute the main run-all-trades worker cycle outside ``bot_engine.py``."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import importlib
import time
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.core.decision_journal import DecisionJournalRecorder
from ai_trading.governance.entry_control import (
    REPLAY_LIVE_ENTRY_CONTROL_ATTR,
    REPLAY_LIVE_PARITY_GATE_ATTR,
    build_replay_live_entry_control,
)

_REPLAY_LIVE_PARITY_GATE_CACHE: dict[str, Any] = {
    "updated_mono": 0.0,
    "gate": None,
}


def _replay_live_parity_gate_cache_ttl_seconds() -> float:
    return max(
        0.0,
        min(
            float(
                get_env(
                    "AI_TRADING_REPLAY_LIVE_PARITY_GATE_CYCLE_TTL_SEC",
                    300.0,
                    cast=float,
                )
                or 0.0
            ),
            900.0,
        ),
    )


def _cacheable_replay_live_parity_gate(gate: Any) -> bool:
    return (
        isinstance(gate, dict)
        and bool(gate.get("enabled", False))
        and bool(gate.get("available", True))
    )


def _replay_entry_control_execution_mode(*, runtime: Any, cfg_runtime: Any) -> str:
    engine = getattr(runtime, "execution_engine", None) or getattr(
        runtime, "exec_engine", None
    )
    for candidate in (
        getattr(engine, "execution_mode", None),
        getattr(runtime, "execution_mode", None),
        getattr(cfg_runtime, "execution_mode", None),
        get_env("EXECUTION_MODE", "paper", cast=str),
    ):
        token = str(candidate or "").strip().lower()
        if token:
            return token
    return "paper"


def _propagate_replay_entry_control(
    *,
    state: Any,
    runtime: Any,
    be: Any,
    gate: dict[str, Any],
    control: dict[str, Any],
) -> None:
    targets = (
        state,
        runtime,
        getattr(runtime, "execution_engine", None),
        getattr(runtime, "exec_engine", None),
        getattr(be, "_exec_engine", None),
    )
    seen: set[int] = set()
    for target in targets:
        if target is None or id(target) in seen:
            continue
        seen.add(id(target))
        setattr(target, REPLAY_LIVE_PARITY_GATE_ATTR, dict(gate))
        setattr(target, REPLAY_LIVE_ENTRY_CONTROL_ATTR, dict(control))


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
    ttl_seconds = _replay_live_parity_gate_cache_ttl_seconds()
    now_mono = time.monotonic()
    cached_gate = _REPLAY_LIVE_PARITY_GATE_CACHE.get("gate")
    cached_age = now_mono - float(_REPLAY_LIVE_PARITY_GATE_CACHE.get("updated_mono") or 0.0)
    gate: dict[str, Any] | None = None
    if (
        ttl_seconds > 0.0
        and _cacheable_replay_live_parity_gate(cached_gate)
        and cached_age <= ttl_seconds
    ):
        gate = dict(cached_gate)
        gate["cache_age_s"] = round(float(cached_age), 3)
    if gate is None:
        try:
            from ai_trading.governance.replay_live_parity import summarize_replay_live_parity_gate
            from ai_trading.tools.runtime_performance_report import summarize_oms_lifecycle_parity

            oms_lifecycle_parity = summarize_oms_lifecycle_parity()
            gate = summarize_replay_live_parity_gate(
                oms_lifecycle_parity=oms_lifecycle_parity,
                expected_policy_hash=str(getattr(state, "effective_policy_hash", "") or ""),
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
        if _cacheable_replay_live_parity_gate(gate):
            _REPLAY_LIVE_PARITY_GATE_CACHE["updated_mono"] = now_mono
            _REPLAY_LIVE_PARITY_GATE_CACHE["gate"] = dict(gate)
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
) -> None:
    """Execute the netting worker cycle after bootstrap has completed.

    The netting pipeline is the only decision path; the cycle does not read
    ``netting_enabled``. Startup rejects a config that disables it.
    """
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
    entry_control = build_replay_live_entry_control(
        gate=parity_gate,
        required=require_replay_live_parity_gate,
        execution_mode=_replay_entry_control_execution_mode(
            runtime=runtime,
            cfg_runtime=cfg_runtime,
        ),
    )
    _propagate_replay_entry_control(
        state=state,
        runtime=runtime,
        be=be,
        gate=parity_gate,
        control=entry_control,
    )

    decision_recorder = DecisionJournalRecorder(
        path=str(be._decision_log_runtime_path()),
        write_impl=be._write_decision_record,
    )
    setattr(state, "_decision_journal_recorder", decision_recorder)

    def _record_open_order_unknown(exc: BaseException) -> None:
        breaker_info = be.classify_exception(exc, dependency="broker_open_orders")
        breakers = be._dependency_breakers(state)
        breakers.record_failure("broker_open_orders", breaker_info)
        be._handle_error(breaker_info, state=state, ctx=runtime)

    try:
        can_list_orders = (
            hasattr(api, "list_orders") and callable(getattr(api, "list_orders", None))
        ) or (hasattr(api, "get_orders") and callable(getattr(api, "get_orders", None)))
        if not can_list_orders:
            if not getattr(state, "_warned_missing_list_orders", False):
                be.logger.warning("API capability unavailable: list_orders/get_orders")
                setattr(state, "_warned_missing_list_orders", True)
            _record_open_order_unknown(
                AttributeError("broker client missing list_orders/get_orders")
            )
            return
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
                _record_open_order_unknown(e)
                return
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
            _propagate_replay_entry_control(
                state=state,
                runtime=runtime,
                be=be,
                gate=parity_gate,
                control=entry_control,
            )
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
    finally:
        if getattr(state, "_decision_journal_recorder", None) is decision_recorder:
            delattr(state, "_decision_journal_recorder")


__all__ = ["execute_run_all_trades_cycle"]
