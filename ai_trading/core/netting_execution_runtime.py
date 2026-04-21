"""Execution-context runtime helpers extracted from ``bot_engine.py``."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ai_trading.core.netting_execution_context import NettingExecutionContext, build_netting_execution_context
from ai_trading.oms.ledger import OrderLedger


def _bot_engine() -> Any:
    return importlib.import_module("ai_trading.core.bot_engine")


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class NettingExecutionRuntime:
    ledger: OrderLedger | None
    execution_context: NettingExecutionContext


def _prepare_oms_ledger(state: Any, cfg: Any) -> OrderLedger | None:
    be = _bot_engine()

    ledger: OrderLedger | None = None
    execution_mode = str(getattr(cfg, "execution_mode", "sim") or "sim").strip().lower()
    intent_store_enabled = bool(
        be.get_env("AI_TRADING_OMS_INTENT_STORE_ENABLED", True, cast=bool)
    )
    ledger_enabled = bool(getattr(cfg, "ledger_enabled", False))
    if execution_mode == "live":
        be.emit_once(
            be.logger,
            "OMS_DURABILITY_HIERARCHY",
            "info",
            "OMS durability hierarchy resolved",
            authoritative_store="intent_store",
            intent_store_enabled=intent_store_enabled,
            ledger_enabled=ledger_enabled,
            execution_mode=execution_mode,
        )
    if execution_mode == "live":
        if getattr(state, "_oms_ledger", None) is not None:
            setattr(state, "_oms_ledger", None)
        be.emit_once(
            be.logger,
            "OMS_LEDGER_DISABLED_LIVE",
            "info",
            "Live execution disables JSONL OMS ledger side paths; intent store is authoritative",
        )
        return None
    if not ledger_enabled:
        return None

    ledger_path = be._resolve_runtime_artifact_path(
        str(getattr(cfg, "ledger_path", "runtime/oms_ledger.jsonl"))
    )
    lookback_hours = float(getattr(cfg, "ledger_lookback_hours", 24.0))

    cached_ledger = getattr(state, "_oms_ledger", None)
    cached_path = str(getattr(cached_ledger, "_path", "") or "")
    cached_lookback = getattr(cached_ledger, "_configured_lookback_hours", None)
    configured_lookback = _safe_float(lookback_hours, default=24.0)
    cached_lookback_value = _parse_optional_float(cached_lookback)
    cache_usable = (
        isinstance(cached_ledger, OrderLedger)
        and hasattr(cached_ledger, "record")
        and hasattr(cached_ledger, "seen_client_order_id")
        and cached_path == str(ledger_path)
        and cached_lookback_value is not None
        and float(cached_lookback_value) == configured_lookback
    )
    if cache_usable:
        ledger = cached_ledger
    else:
        ledger = OrderLedger(str(ledger_path), configured_lookback)
        setattr(ledger, "_configured_lookback_hours", configured_lookback)
        setattr(state, "_oms_ledger", ledger)
    return ledger


def build_netting_execution_runtime(
    *,
    cfg: Any,
    state: Any,
    runtime: Any,
    now: datetime,
    targets: Mapping[str, Any],
    positions: Mapping[str, float],
    latest_price: Mapping[str, float],
    blocked_symbols: set[str],
    candidate_expected_net_edge: Mapping[str, float],
    allocation_weights: Mapping[str, float],
    learned_overrides: Mapping[str, Any],
    sleeve_snapshot: Mapping[str, Any],
    effective_policy: Any,
    kill_switch: bool,
    policy_disabled_gate_roots: set[str],
) -> NettingExecutionRuntime:
    """Build ledger durability and shared execution context for a netting cycle."""

    be = _bot_engine()
    ledger = _prepare_oms_ledger(state, cfg)
    execution_context = build_netting_execution_context(
        cfg=cfg,
        state=state,
        runtime=runtime,
        now=now,
        targets=targets,
        positions=positions,
        latest_price=latest_price,
        blocked_symbols=blocked_symbols,
        candidate_expected_net_edge=candidate_expected_net_edge,
        allocation_weights=allocation_weights,
        learned_overrides=learned_overrides,
        sleeve_snapshot=sleeve_snapshot,
        effective_policy=effective_policy,
        kill_switch=kill_switch,
        logger=be.logger,
        policy_disabled_gate_roots=policy_disabled_gate_roots,
        decision_record_config_snapshot_func=be._decision_record_config_snapshot,
        execution_model_lineage_func=be._execution_model_lineage,
        pretrade_rate_limiter_func=be._pretrade_rate_limiter,
        tca_stale_block_reason_func=be._tca_stale_block_reason,
        resolve_slo_derisk_effective_mode_func=be._resolve_slo_derisk_effective_mode,
        resolve_operational_safety_tier_func=be.resolve_operational_safety_tier,
        apply_operational_safety_hysteresis_func=be._apply_operational_safety_hysteresis,
        update_rollout_governance_state_func=be._update_rollout_governance_state,
        resolve_capacity_throttle_adaptive_params_func=be._resolve_capacity_throttle_adaptive_params,
        resolve_primary_feed_derisk_state_func=be._resolve_primary_feed_derisk_state,
        resolve_runtime_info_log_ttl_seconds_func=be._resolve_runtime_info_log_ttl_seconds,
        should_emit_runtime_info_log_func=be._should_emit_runtime_info_log,
        read_jsonl_records_func=be._read_jsonl_records,
        gate_effectiveness_log_path_func=be._gate_effectiveness_log_path,
        apply_gate_auto_disable_hysteresis_func=be._apply_gate_auto_disable_hysteresis,
        symbol_adaptive_sizing_profiles_func=be._symbol_adaptive_sizing_profiles,
        get_sector_func=be.get_sector,
        load_uncertainty_capital_state_func=be._load_uncertainty_capital_state,
    )
    return NettingExecutionRuntime(ledger=ledger, execution_context=execution_context)


__all__ = [
    "NettingExecutionRuntime",
    "build_netting_execution_runtime",
]
