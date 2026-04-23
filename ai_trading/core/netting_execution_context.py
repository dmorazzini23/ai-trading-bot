"""Global execution-control context for the live netting cycle."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from typing import Any, Callable, Mapping

from ai_trading.config.management import get_env
from ai_trading.policy.compiler import SafetyTier


@dataclass(slots=True)
class NettingExecutionContext:
    decision_snapshot_template: dict[str, Any]
    execution_model_lineage: dict[str, Any]
    rate_limiter: Any
    tca_stale_reason: str | None
    slo_derisk_scale: float
    slo_derisk_details: dict[str, Any]
    operational_tier: SafetyTier
    tier_reasons: tuple[str, ...]
    rollout_summary: dict[str, Any]
    ramp_summary: dict[str, Any]
    ramp_live_multiplier: float
    burn_in_live_ready: bool
    burn_in_live_reason: str
    live_execution_mode: bool
    liq_regime_enabled: bool
    event_blackout_enabled: bool
    event_blackout_days: int
    event_blackout_cache: dict[str, bool]
    alpha_decay_deweight_enabled: bool
    alpha_decay_qty_step: float
    alpha_decay_qty_max_deweight: float
    capacity_throttle_enabled: bool
    capacity_spread_soft_bps: float
    capacity_spread_hard_bps: float
    capacity_volume_soft_participation: float
    capacity_volume_hard_participation: float
    capacity_min_scale: float
    capacity_adaptive_details: dict[str, Any]
    thin_spread_bps: float
    thin_vol_mult: float
    primary_feed_derisk: dict[str, Any]
    portfolio_current_gross: float
    sector_gross: dict[str, float]
    symbol_adaptive_profiles: dict[str, dict[str, Any]]
    ineffective_gate_blocklist: set[str]
    ineffective_gate_diagnostics: dict[str, dict[str, float]]
    gate_auto_disable_hysteresis_context: dict[str, Any]
    uncertainty_capital_state: dict[str, Any]
    uncertainty_cycle_events: list[dict[str, Any]]
    penalty_overlap_coordination_enabled: bool
    penalty_overlap_weight_dampen: float
    penalty_overlap_min_scale_floor: float


def _extract_slo_derisk_metric(status: Any) -> tuple[float, int]:
    if not isinstance(status, MappingABC):
        return (0.0, 0)
    try:
        value = float(status.get("current_value") or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    try:
        samples = int(status.get("sample_count") or 0)
    except (TypeError, ValueError):
        samples = 0
    return (value, max(samples, 0))


def _build_slo_derisk_state(
    *,
    state: Any,
    logger: Any,
    resolve_slo_derisk_effective_mode_func: Callable[..., tuple[str, float, dict[str, Any]]],
) -> tuple[float, dict[str, Any]]:
    slo_derisk_scale = 1.0
    slo_derisk_details: dict[str, Any] = {}
    if not bool(get_env("AI_TRADING_DERISK_ON_SLO_BREACH_ENABLED", True, cast=bool)):
        return (slo_derisk_scale, slo_derisk_details)

    mode = str(get_env("AI_TRADING_DERISK_SLO_MODE", "adaptive", cast=str) or "adaptive").strip().lower()
    if mode not in {"block", "scale", "adaptive"}:
        mode = "block"
    min_samples = max(1, int(get_env("AI_TRADING_DERISK_SLO_MIN_SAMPLES", 5, cast=int)))
    max_reject_rate_pct = float(get_env("AI_TRADING_DERISK_SLO_MAX_REJECT_RATE_PCT", 5.0, cast=float))
    max_execution_drift_bps = float(
        get_env("AI_TRADING_DERISK_SLO_MAX_EXEC_DRIFT_BPS", 35.0, cast=float)
    )
    max_slippage_bps = float(get_env("AI_TRADING_DERISK_SLO_MAX_SLIPPAGE_BPS", 25.0, cast=float))
    max_calibration_ece = float(get_env("AI_TRADING_DERISK_SLO_MAX_CALIB_ECE", 0.15, cast=float))
    max_calibration_brier = float(
        get_env("AI_TRADING_DERISK_SLO_MAX_CALIB_BRIER", 0.35, cast=float)
    )
    max_feature_drift_psi = float(
        get_env("AI_TRADING_DERISK_SLO_MAX_FEATURE_DRIFT_PSI", 0.30, cast=float)
    )
    max_label_drift_psi = float(
        get_env("AI_TRADING_DERISK_SLO_MAX_LABEL_DRIFT_PSI", 0.30, cast=float)
    )
    max_residual_drift_psi = float(
        get_env("AI_TRADING_DERISK_SLO_MAX_RESIDUAL_DRIFT_PSI", 0.30, cast=float)
    )
    max_pacing_cap_hit_rate_pct = float(
        get_env("AI_TRADING_DERISK_SLO_MAX_PACING_CAP_HIT_RATE_PCT", 40.0, cast=float)
    )
    max_pending_oldest_age_sec = float(
        get_env("AI_TRADING_DERISK_SLO_MAX_PENDING_OLDEST_AGE_SEC", 300.0, cast=float)
    )
    pending_min_samples = max(
        1,
        int(get_env("AI_TRADING_DERISK_SLO_PENDING_MIN_SAMPLES", 1, cast=int)),
    )

    reject_rate_pct = 0.0
    execution_drift_bps = 0.0
    slippage_bps = 0.0
    calibration_ece = 0.0
    calibration_brier = 0.0
    feature_drift_psi = 0.0
    label_drift_psi = 0.0
    residual_drift_psi = 0.0
    pacing_cap_hit_rate_pct = 0.0
    pending_oldest_age_sec = 0.0
    reject_samples = 0
    drift_samples = 0
    slippage_samples = 0
    calibration_ece_samples = 0
    calibration_brier_samples = 0
    feature_drift_samples = 0
    label_drift_samples = 0
    residual_drift_samples = 0
    pacing_samples = 0
    pending_samples = 0
    try:
        from ai_trading.monitoring.slo import get_slo_monitor

        monitor = get_slo_monitor()
        reject_rate_pct, reject_samples = _extract_slo_derisk_metric(
            monitor.get_slo_status("order_reject_rate_pct")
        )
        execution_drift_bps, drift_samples = _extract_slo_derisk_metric(
            monitor.get_slo_status("execution_drift_bps")
        )
        slippage_bps, slippage_samples = _extract_slo_derisk_metric(
            monitor.get_slo_status("realized_slippage_bps")
        )
        calibration_ece, calibration_ece_samples = _extract_slo_derisk_metric(
            monitor.get_slo_status("live_calibration_ece")
        )
        calibration_brier, calibration_brier_samples = _extract_slo_derisk_metric(
            monitor.get_slo_status("live_calibration_brier")
        )
        feature_drift_psi, feature_drift_samples = _extract_slo_derisk_metric(
            monitor.get_slo_status("drift_psi")
        )
        label_drift_psi, label_drift_samples = _extract_slo_derisk_metric(
            monitor.get_slo_status("label_drift_psi")
        )
        residual_drift_psi, residual_drift_samples = _extract_slo_derisk_metric(
            monitor.get_slo_status("residual_drift_psi")
        )
        pacing_cap_hit_rate_pct, pacing_samples = _extract_slo_derisk_metric(
            monitor.get_slo_status("order_pacing_cap_hit_rate_pct")
        )
        pending_oldest_age_sec, pending_samples = _extract_slo_derisk_metric(
            monitor.get_slo_status("pending_oldest_age_sec")
        )
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        logger.debug("SLO_DERISK_SNAPSHOT_FAILED", exc_info=True)

    reject_breached = reject_samples >= min_samples and reject_rate_pct >= max_reject_rate_pct
    drift_breached = drift_samples >= min_samples and execution_drift_bps >= max_execution_drift_bps
    slippage_breached = slippage_samples >= min_samples and slippage_bps >= max_slippage_bps
    calibration_ece_breached = calibration_ece_samples >= min_samples and calibration_ece >= max_calibration_ece
    calibration_brier_breached = (
        calibration_brier_samples >= min_samples and calibration_brier >= max_calibration_brier
    )
    feature_drift_breached = (
        feature_drift_samples >= min_samples and feature_drift_psi >= max_feature_drift_psi
    )
    label_drift_breached = (
        label_drift_samples >= min_samples and label_drift_psi >= max_label_drift_psi
    )
    residual_drift_breached = (
        residual_drift_samples >= min_samples and residual_drift_psi >= max_residual_drift_psi
    )
    pacing_breached = (
        pacing_samples >= min_samples and pacing_cap_hit_rate_pct >= max_pacing_cap_hit_rate_pct
    )
    pending_breached = (
        pending_samples >= pending_min_samples
        and pending_oldest_age_sec >= max_pending_oldest_age_sec
    )

    drift_calibration_guard_enabled = bool(
        get_env("AI_TRADING_DRIFT_CALIBRATION_GUARD_ENABLED", True, cast=bool)
    )
    stale_required_cycles = max(
        1,
        int(get_env("AI_TRADING_DRIFT_CALIBRATION_STALE_CYCLES", 3, cast=int)),
    )
    stale_derisk_scale = max(
        0.05,
        min(
            1.0,
            float(get_env("AI_TRADING_DRIFT_CALIBRATION_STALE_DERISK_SCALE", 0.50, cast=float)),
        ),
    )
    stale_detected = bool(
        drift_samples < min_samples
        or calibration_ece_samples < min_samples
        or calibration_brier_samples < min_samples
        or feature_drift_samples < min_samples
        or label_drift_samples < min_samples
        or residual_drift_samples < min_samples
    )
    stale_cycles = int(getattr(state, "_drift_calibration_stale_cycles", 0) or 0)
    if drift_calibration_guard_enabled:
        stale_cycles = stale_cycles + 1 if stale_detected else max(0, stale_cycles - 1)
    else:
        stale_cycles = 0
    setattr(state, "_drift_calibration_stale_cycles", stale_cycles)
    drift_calibration_stale_triggered = bool(
        drift_calibration_guard_enabled and stale_cycles >= stale_required_cycles
    )
    breached = bool(
        reject_breached
        or drift_breached
        or slippage_breached
        or calibration_ece_breached
        or calibration_brier_breached
        or feature_drift_breached
        or label_drift_breached
        or residual_drift_breached
        or pacing_breached
        or pending_breached
    )
    slo_derisk_details = {
        "mode": mode,
        "reject_rate_pct": reject_rate_pct,
        "execution_drift_bps": execution_drift_bps,
        "slippage_bps": slippage_bps,
        "calibration_ece": calibration_ece,
        "calibration_brier": calibration_brier,
        "drift_psi": feature_drift_psi,
        "label_drift_psi": label_drift_psi,
        "residual_drift_psi": residual_drift_psi,
        "order_pacing_cap_hit_rate_pct": pacing_cap_hit_rate_pct,
        "pending_oldest_age_sec": pending_oldest_age_sec,
        "reject_samples": reject_samples,
        "drift_samples": drift_samples,
        "slippage_samples": slippage_samples,
        "calibration_ece_samples": calibration_ece_samples,
        "calibration_brier_samples": calibration_brier_samples,
        "feature_drift_samples": feature_drift_samples,
        "label_drift_samples": label_drift_samples,
        "residual_drift_samples": residual_drift_samples,
        "pacing_samples": pacing_samples,
        "pending_samples": pending_samples,
        "max_reject_rate_pct": max_reject_rate_pct,
        "max_execution_drift_bps": max_execution_drift_bps,
        "max_slippage_bps": max_slippage_bps,
        "max_calibration_ece": max_calibration_ece,
        "max_calibration_brier": max_calibration_brier,
        "max_feature_drift_psi": max_feature_drift_psi,
        "max_label_drift_psi": max_label_drift_psi,
        "max_residual_drift_psi": max_residual_drift_psi,
        "max_pacing_cap_hit_rate_pct": max_pacing_cap_hit_rate_pct,
        "max_pending_oldest_age_sec": max_pending_oldest_age_sec,
        "pending_min_samples": pending_min_samples,
        "drift_calibration_guard_enabled": drift_calibration_guard_enabled,
        "drift_calibration_stale_detected": stale_detected,
        "drift_calibration_stale_cycles": int(stale_cycles),
        "drift_calibration_stale_required_cycles": int(stale_required_cycles),
        "drift_calibration_stale_derisk_scale": float(stale_derisk_scale),
        "drift_calibration_stale_triggered": bool(drift_calibration_stale_triggered),
        "reject_breached": reject_breached,
        "drift_breached": drift_breached,
        "slippage_breached": slippage_breached,
        "calibration_ece_breached": calibration_ece_breached,
        "calibration_brier_breached": calibration_brier_breached,
        "feature_drift_breached": feature_drift_breached,
        "label_drift_breached": label_drift_breached,
        "residual_drift_breached": residual_drift_breached,
        "pacing_breached": pacing_breached,
        "pending_breached": pending_breached,
        "breached": breached,
    }
    if drift_calibration_stale_triggered:
        slo_derisk_scale = min(slo_derisk_scale, float(stale_derisk_scale))
        slo_derisk_details["stale_scale_applied"] = float(stale_derisk_scale)
        logger.warning(
            "DRIFT_CALIBRATION_STALE_DERISK",
            extra={
                "stale_cycles": int(stale_cycles),
                "required_cycles": int(stale_required_cycles),
                "scale": float(stale_derisk_scale),
            },
        )
    if breached:
        effective_mode, effective_scale_mult, mode_details = resolve_slo_derisk_effective_mode_func(
            configured_mode=mode,
            reject_breached=reject_breached,
            drift_breached=drift_breached,
            slippage_breached=slippage_breached,
            calibration_ece_breached=calibration_ece_breached,
            calibration_brier_breached=calibration_brier_breached,
            feature_drift_breached=feature_drift_breached,
            label_drift_breached=label_drift_breached,
            residual_drift_breached=residual_drift_breached,
            pacing_breached=pacing_breached,
            pending_breached=pending_breached,
            pacing_hit_rate_pct=pacing_cap_hit_rate_pct,
            pending_oldest_age_sec=pending_oldest_age_sec,
        )
        slo_derisk_details.update(mode_details)
        slo_derisk_details["effective_mode"] = effective_mode
        if effective_mode == "block":
            state.halt_trading = True
            state.halt_reason = "DERISK_SLO_BREACH_BLOCK"
        else:
            effective_scale_mult = max(0.05, min(float(effective_scale_mult), 1.0))
            slo_derisk_scale = min(slo_derisk_scale, effective_scale_mult)
            slo_derisk_details["scale_mult"] = effective_scale_mult
        logger.warning("DERISK_SLO_BREACH", extra=slo_derisk_details)
    return (slo_derisk_scale, slo_derisk_details)


def _build_gate_auto_disable_state(
    *,
    state: Any,
    now: datetime,
    logger: Any,
    policy_disabled_gate_roots: set[str],
    read_jsonl_records_func: Callable[..., list[dict[str, Any]]],
    gate_effectiveness_log_path_func: Callable[[], Any],
    apply_gate_auto_disable_hysteresis_func: Callable[..., tuple[set[str], dict[str, dict[str, float]], dict[str, Any]]],
) -> tuple[set[str], dict[str, dict[str, float]], dict[str, Any]]:
    ineffective_gate_blocklist: set[str] = set()
    ineffective_gate_diagnostics: dict[str, dict[str, float]] = {}
    gate_auto_disable_hysteresis_context: dict[str, Any] = {
        "min_on_dwell_sec": 0.0,
        "min_off_dwell_sec": 0.0,
        "min_disabled_hold_sec": 0.0,
        "max_transitions_per_hour": 0,
        "transitions_used_in_window": 0,
        "transitions": [],
        "holds": [],
        "candidate_count": 0,
        "effective_count": 0,
    }
    gate_auto_disable_enabled = bool(
        get_env("AI_TRADING_GATE_AUTO_DISABLE_NON_POSITIVE_ENABLED", True, cast=bool)
    )
    gate_auto_disable_lookback_cycles = max(
        10,
        int(get_env("AI_TRADING_GATE_AUTO_DISABLE_LOOKBACK_CYCLES", 240, cast=int)),
    )
    gate_auto_disable_min_blocked = max(
        10,
        int(get_env("AI_TRADING_GATE_AUTO_DISABLE_MIN_BLOCKED", 100, cast=int)),
    )
    gate_auto_disable_min_contribution_bps = float(
        get_env("AI_TRADING_GATE_AUTO_DISABLE_MIN_CONTRIBUTION_BPS", 0.0, cast=float)
    )
    critical_gates = {
        "KILL_SWITCH_BLOCK",
        "AUTH_BROKER_HALT_FORBIDDEN_COOLDOWN",
        "DERISK_PRIMARY_FEED_BLOCK",
        "PAPER_BURN_IN_BLOCK",
        "BURN_IN_POLICY_HASH_MISMATCH",
        "BURN_IN_CONFIG_HASH_MISMATCH",
        "RECON_MISMATCH_HALT",
        "PRE_SUBMIT_INSUFFICIENT_POSITION_AVAILABLE",
    }
    if gate_auto_disable_enabled:
        try:
            gate_rows = read_jsonl_records_func(
                str(gate_effectiveness_log_path_func()),
                max_records=max(gate_auto_disable_lookback_cycles * 2, 200),
            )
            gate_agg: dict[str, dict[str, float]] = {}
            for row in gate_rows[-gate_auto_disable_lookback_cycles:]:
                raw_attr = row.get("gate_attribution")
                if not isinstance(raw_attr, MappingABC):
                    continue
                for gate_name_raw, payload in raw_attr.items():
                    if not isinstance(payload, MappingABC):
                        continue
                    gate_name = str(gate_name_raw or "").strip().upper()
                    if not gate_name:
                        continue
                    try:
                        blocked = float(payload.get("blocked_records") or 0.0)
                    except (TypeError, ValueError):
                        blocked = 0.0
                    try:
                        edge_sum = float(payload.get("edge_proxy_bps_sum") or 0.0)
                    except (TypeError, ValueError):
                        edge_sum = 0.0
                    stats = gate_agg.setdefault(gate_name, {"blocked": 0.0, "edge_sum": 0.0})
                    stats["blocked"] = float(stats["blocked"]) + float(blocked)
                    stats["edge_sum"] = float(stats["edge_sum"]) + float(edge_sum)
            for gate_name, stats in gate_agg.items():
                blocked_count = int(max(stats.get("blocked", 0.0), 0.0))
                if blocked_count < gate_auto_disable_min_blocked or gate_name in critical_gates:
                    continue
                marginal_contribution_bps = float(
                    -float(stats.get("edge_sum", 0.0)) / float(max(blocked_count, 1))
                )
                ineffective_gate_diagnostics[gate_name] = {
                    "blocked_records": float(blocked_count),
                    "marginal_contribution_bps": float(marginal_contribution_bps),
                }
                if marginal_contribution_bps < float(gate_auto_disable_min_contribution_bps):
                    ineffective_gate_blocklist.add(str(gate_name))
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            logger.debug("GATE_AUTO_DISABLE_EVALUATION_FAILED", exc_info=True)
        (
            ineffective_gate_blocklist,
            ineffective_gate_diagnostics,
            gate_auto_disable_hysteresis_context,
        ) = apply_gate_auto_disable_hysteresis_func(
            state=state,
            candidate_disabled_gates=ineffective_gate_blocklist,
            candidate_diagnostics=ineffective_gate_diagnostics,
            now=now,
        )
        if ineffective_gate_blocklist:
            logger.info(
                "GATE_AUTO_DISABLE_APPLIED",
                extra={
                    "disabled_gates": sorted(ineffective_gate_blocklist),
                    "diagnostics": {
                        gate_name: ineffective_gate_diagnostics.get(gate_name, {})
                        for gate_name in sorted(ineffective_gate_blocklist)[:20]
                    },
                    "min_blocked": int(gate_auto_disable_min_blocked),
                    "min_contribution_bps": float(gate_auto_disable_min_contribution_bps),
                    "lookback_cycles": int(gate_auto_disable_lookback_cycles),
                    "hysteresis": {
                        "min_on_dwell_sec": float(
                            gate_auto_disable_hysteresis_context.get("min_on_dwell_sec", 0.0) or 0.0
                        ),
                        "min_off_dwell_sec": float(
                            gate_auto_disable_hysteresis_context.get("min_off_dwell_sec", 0.0) or 0.0
                        ),
                        "min_disabled_hold_sec": float(
                            gate_auto_disable_hysteresis_context.get("min_disabled_hold_sec", 0.0) or 0.0
                        ),
                        "max_transitions_per_hour": int(
                            gate_auto_disable_hysteresis_context.get("max_transitions_per_hour", 0) or 0
                        ),
                        "transitions_used_in_window": int(
                            gate_auto_disable_hysteresis_context.get("transitions_used_in_window", 0) or 0
                        ),
                    },
                },
            )
        elif bool(gate_auto_disable_hysteresis_context.get("transitions")):
            logger.info(
                "GATE_AUTO_DISABLE_HYSTERESIS_TRANSITION",
                extra={
                    "transitions": list(gate_auto_disable_hysteresis_context.get("transitions", []))[:20],
                    "holds": list(gate_auto_disable_hysteresis_context.get("holds", []))[:20],
                    "transitions_used_in_window": int(
                        gate_auto_disable_hysteresis_context.get("transitions_used_in_window", 0) or 0
                    ),
                    "max_transitions_per_hour": int(
                        gate_auto_disable_hysteresis_context.get("max_transitions_per_hour", 0) or 0
                    ),
                },
            )

    if policy_disabled_gate_roots:
        for gate_root in sorted(policy_disabled_gate_roots):
            ineffective_gate_blocklist.add(str(gate_root))
            ineffective_gate_diagnostics.setdefault(
                str(gate_root),
                {"blocked_records": 0.0, "marginal_contribution_bps": 0.0},
            )
        logger.warning(
            "POLICY_ROLLBACK_GATES_APPLIED",
            extra={"disabled_gate_roots": sorted(policy_disabled_gate_roots)},
        )
    return (
        ineffective_gate_blocklist,
        ineffective_gate_diagnostics,
        gate_auto_disable_hysteresis_context,
    )


def build_netting_execution_context(
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
    logger: Any,
    policy_disabled_gate_roots: set[str],
    decision_record_config_snapshot_func: Callable[..., dict[str, Any]],
    execution_model_lineage_func: Callable[[], dict[str, Any]],
    pretrade_rate_limiter_func: Callable[[Any], Any],
    tca_stale_block_reason_func: Callable[[datetime], str | None],
    resolve_slo_derisk_effective_mode_func: Callable[..., tuple[str, float, dict[str, Any]]],
    resolve_operational_safety_tier_func: Callable[..., tuple[SafetyTier, tuple[str, ...]]],
    apply_operational_safety_hysteresis_func: Callable[..., tuple[SafetyTier, tuple[str, ...]]],
    update_rollout_governance_state_func: Callable[..., dict[str, Any]],
    resolve_capacity_throttle_adaptive_params_func: Callable[..., tuple[float, float, float, float, float, dict[str, Any]]],
    resolve_primary_feed_derisk_state_func: Callable[[Any], dict[str, Any]],
    resolve_runtime_info_log_ttl_seconds_func: Callable[[str, float], float],
    should_emit_runtime_info_log_func: Callable[..., bool],
    read_jsonl_records_func: Callable[..., list[dict[str, Any]]],
    gate_effectiveness_log_path_func: Callable[[], Any],
    apply_gate_auto_disable_hysteresis_func: Callable[..., tuple[set[str], dict[str, dict[str, float]], dict[str, Any]]],
    symbol_adaptive_sizing_profiles_func: Callable[..., dict[str, dict[str, Any]]],
    get_sector_func: Callable[[str], str | None],
    load_uncertainty_capital_state_func: Callable[[], dict[str, Any]],
) -> NettingExecutionContext:
    decision_snapshot_template = decision_record_config_snapshot_func(
        cfg=cfg,
        state=state,
        allocation_weights=allocation_weights,
        learned_overrides=learned_overrides,
        sleeve_configs=sleeve_snapshot,
        liquidity_regime=None,
    )
    execution_model_lineage = execution_model_lineage_func()
    rate_limiter = pretrade_rate_limiter_func(state)
    tca_stale_reason = tca_stale_block_reason_func(now)

    slo_derisk_scale, slo_derisk_details = _build_slo_derisk_state(
        state=state,
        logger=logger,
        resolve_slo_derisk_effective_mode_func=resolve_slo_derisk_effective_mode_func,
    )

    expected_net_edge_cycle = 0.0
    if candidate_expected_net_edge:
        try:
            expected_net_edge_cycle = float(max(candidate_expected_net_edge.values()))
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            expected_net_edge_cycle = 0.0
    pending_symbol_count = len(blocked_symbols)
    previous_tier_raw = str(
        getattr(state, "operational_safety_tier", SafetyTier.NORMAL.value) or SafetyTier.NORMAL.value
    )
    try:
        previous_tier = SafetyTier(previous_tier_raw)
    except ValueError:
        previous_tier = SafetyTier.NORMAL
    operational_tier, tier_reasons = resolve_operational_safety_tier_func(
        effective_policy,
        {
            "pending_oldest_age_sec": float(slo_derisk_details.get("pending_oldest_age_sec", 0.0) or 0.0),
            "order_pacing_cap_hit_rate_pct": float(
                slo_derisk_details.get("order_pacing_cap_hit_rate_pct", 0.0) or 0.0
            ),
            "live_calibration_ece": float(slo_derisk_details.get("calibration_ece", 0.0) or 0.0),
            "live_calibration_brier": float(slo_derisk_details.get("calibration_brier", 0.0) or 0.0),
            "pending_orders_count": int(max(pending_symbol_count, 0)),
            "expected_net_edge_bps": float(expected_net_edge_cycle),
            "kill_switch": bool(kill_switch),
        },
        previous=previous_tier,
    )
    operational_tier, tier_reasons = apply_operational_safety_hysteresis_func(
        state=state,
        previous_tier=previous_tier,
        candidate_tier=operational_tier,
        candidate_reasons=tier_reasons,
        now=now,
    )
    state.operational_safety_tier = operational_tier.value
    if operational_tier != previous_tier:
        log_level = logging.WARNING if operational_tier is SafetyTier.SAFE else logging.INFO
        logger.log(
            log_level,
            "OPERATIONAL_SAFETY_TIER_CHANGED",
            extra={
                "previous": previous_tier.value,
                "current": operational_tier.value,
                "reasons": list(tier_reasons),
            },
        )

    rollout_summary = update_rollout_governance_state_func(
        state=state,
        cfg=cfg,
        effective_policy=effective_policy,
        slo_derisk_details=slo_derisk_details,
        now=now,
    )
    ramp_summary_raw = rollout_summary.get("capital_ramp", {})
    ramp_summary = dict(ramp_summary_raw) if isinstance(ramp_summary_raw, MappingABC) else {}
    ramp_live_multiplier = float(getattr(state, "capital_ramp_multiplier", 1.0) or 1.0)
    burn_in_live_ready = bool(getattr(state, "burn_in_ready", True))
    burn_in_live_reason = str(getattr(state, "burn_in_block_reason", "") or "")
    execution_mode = str(getattr(cfg, "execution_mode", "sim") or "sim").strip().lower()
    live_execution_mode = execution_mode == "live"

    liq_regime_enabled = bool(get_env("AI_TRADING_LIQ_REGIME_ENABLED", True, cast=bool))
    event_blackout_enabled = bool(get_env("AI_TRADING_EVENT_RISK_BLACKOUT_ENABLED", True, cast=bool))
    event_blackout_days = max(0, int(get_env("AI_TRADING_EVENT_BLACKOUT_DAYS", 3, cast=int)))
    event_blackout_cache: dict[str, bool] = {}
    alpha_decay_deweight_enabled = bool(
        get_env("AI_TRADING_ALPHA_DECAY_DEWEIGHT_ENABLED", True, cast=bool)
    )
    alpha_decay_qty_step = float(get_env("AI_TRADING_ALPHA_DECAY_QTY_STEP", 0.15, cast=float))
    alpha_decay_qty_max_deweight = float(
        get_env("AI_TRADING_ALPHA_DECAY_QTY_MAX_DEWEIGHT", 0.75, cast=float)
    )
    alpha_decay_qty_step = max(0.0, min(alpha_decay_qty_step, 1.0))
    alpha_decay_qty_max_deweight = max(0.0, min(alpha_decay_qty_max_deweight, 0.95))

    capacity_throttle_enabled = bool(
        get_env("AI_TRADING_CAPACITY_AWARE_THROTTLE_ENABLED", True, cast=bool)
    )
    capacity_spread_soft_bps = float(get_env("AI_TRADING_CAPACITY_SPREAD_SOFT_BPS", 12.0, cast=float))
    capacity_spread_hard_bps = float(get_env("AI_TRADING_CAPACITY_SPREAD_HARD_BPS", 30.0, cast=float))
    capacity_volume_soft_participation = float(
        get_env("AI_TRADING_CAPACITY_SOFT_PARTICIPATION", 0.05, cast=float)
    )
    capacity_volume_hard_participation = float(
        get_env("AI_TRADING_CAPACITY_HARD_PARTICIPATION", 0.20, cast=float)
    )
    capacity_min_scale = float(get_env("AI_TRADING_CAPACITY_MIN_SCALE", 0.25, cast=float))
    capacity_min_scale = max(0.05, min(capacity_min_scale, 1.0))
    (
        capacity_spread_soft_bps,
        capacity_spread_hard_bps,
        capacity_volume_soft_participation,
        capacity_volume_hard_participation,
        capacity_min_scale,
        capacity_adaptive_details,
    ) = resolve_capacity_throttle_adaptive_params_func(
        spread_soft_bps=float(capacity_spread_soft_bps),
        spread_hard_bps=float(capacity_spread_hard_bps),
        volume_soft_participation=float(capacity_volume_soft_participation),
        volume_hard_participation=float(capacity_volume_hard_participation),
        min_scale=float(capacity_min_scale),
        slo_derisk_details=slo_derisk_details,
    )
    capacity_adaptive_signature = (
        str(capacity_adaptive_details.get("mode") or "steady"),
        round(float(capacity_adaptive_details.get("spread_soft_bps", capacity_spread_soft_bps) or capacity_spread_soft_bps), 3),
        round(float(capacity_adaptive_details.get("spread_hard_bps", capacity_spread_hard_bps) or capacity_spread_hard_bps), 3),
        round(float(capacity_adaptive_details.get("volume_soft_participation", capacity_volume_soft_participation) or capacity_volume_soft_participation), 5),
        round(float(capacity_adaptive_details.get("volume_hard_participation", capacity_volume_hard_participation) or capacity_volume_hard_participation), 5),
        round(float(capacity_adaptive_details.get("min_scale", capacity_min_scale) or capacity_min_scale), 3),
    )
    if capacity_adaptive_signature != getattr(state, "_last_capacity_throttle_adaptive_signature", None):
        if bool(capacity_adaptive_details.get("enabled", False)):
            logger.info("CAPACITY_THROTTLE_ADAPTIVE_PARAMS", extra=dict(capacity_adaptive_details))
        state._last_capacity_throttle_adaptive_signature = capacity_adaptive_signature
    thin_spread_bps = float(get_env("AI_TRADING_LIQ_THIN_SPREAD_BPS", 25, cast=float))
    thin_vol_mult = float(get_env("AI_TRADING_LIQ_THIN_VOL_MULT", 1.8, cast=float))

    primary_feed_derisk = resolve_primary_feed_derisk_state_func(runtime)
    if primary_feed_derisk.get("triggered"):
        derisk_log_ttl_s = resolve_runtime_info_log_ttl_seconds_func(
            "AI_TRADING_PRIMARY_FEED_DERISK_LOG_TTL_SEC",
            120.0,
        )
        derisk_signature = (
            f"{primary_feed_derisk.get('mode')}:{int(primary_feed_derisk.get('duration_s', 0.0) // 30)}:"
            f"{int(bool(primary_feed_derisk.get('fallback_active')))}:"
            f"{int(bool(primary_feed_derisk.get('quote_quality_failed')))}"
        )
        if should_emit_runtime_info_log_func(
            runtime,
            f"PRIMARY_FEED_DERISK_ACTIVE:{derisk_signature}",
            ttl_seconds=derisk_log_ttl_s,
        ):
            logger.warning("PRIMARY_FEED_DERISK_ACTIVE", extra=dict(primary_feed_derisk))

    portfolio_current_gross = 0.0
    sector_gross: dict[str, float] = {}
    symbols_for_exposure = set(positions.keys()) | set(latest_price.keys())
    for exposure_symbol in symbols_for_exposure:
        px = float(latest_price.get(exposure_symbol, 0.0) or 0.0)
        qty = float(positions.get(exposure_symbol, 0.0) or 0.0)
        notional = abs(qty * px)
        if notional <= 0.0:
            continue
        portfolio_current_gross += notional
        sector = str(get_sector_func(str(exposure_symbol)) or "UNKNOWN").upper()
        sector_gross[sector] = sector_gross.get(sector, 0.0) + notional

    symbol_adaptive_profiles = symbol_adaptive_sizing_profiles_func(
        state,
        symbols=list(targets.keys()),
    )
    (
        ineffective_gate_blocklist,
        ineffective_gate_diagnostics,
        gate_auto_disable_hysteresis_context,
    ) = _build_gate_auto_disable_state(
        state=state,
        now=now,
        logger=logger,
        policy_disabled_gate_roots=policy_disabled_gate_roots,
        read_jsonl_records_func=read_jsonl_records_func,
        gate_effectiveness_log_path_func=gate_effectiveness_log_path_func,
        apply_gate_auto_disable_hysteresis_func=apply_gate_auto_disable_hysteresis_func,
    )

    uncertainty_capital_state = load_uncertainty_capital_state_func()
    uncertainty_cycle_events: list[dict[str, Any]] = []
    penalty_overlap_coordination_enabled = bool(
        get_env("AI_TRADING_MULTI_PENALTY_COORDINATION_ENABLED", True, cast=bool)
    )
    penalty_overlap_weight_dampen = max(
        0.0,
        min(
            1.0,
            float(get_env("AI_TRADING_MULTI_PENALTY_COORDINATION_WEIGHT_DAMPEN", 0.50, cast=float)),
        ),
    )
    penalty_overlap_min_scale_floor = max(
        0.05,
        min(
            1.0,
            float(get_env("AI_TRADING_MULTI_PENALTY_COORDINATION_MIN_SCALE_FLOOR", 0.55, cast=float)),
        ),
    )

    return NettingExecutionContext(
        decision_snapshot_template=decision_snapshot_template,
        execution_model_lineage=execution_model_lineage,
        rate_limiter=rate_limiter,
        tca_stale_reason=tca_stale_reason,
        slo_derisk_scale=float(slo_derisk_scale),
        slo_derisk_details=slo_derisk_details,
        operational_tier=operational_tier,
        tier_reasons=tuple(tier_reasons),
        rollout_summary=rollout_summary,
        ramp_summary=ramp_summary,
        ramp_live_multiplier=float(ramp_live_multiplier),
        burn_in_live_ready=bool(burn_in_live_ready),
        burn_in_live_reason=str(burn_in_live_reason),
        live_execution_mode=bool(live_execution_mode),
        liq_regime_enabled=bool(liq_regime_enabled),
        event_blackout_enabled=bool(event_blackout_enabled),
        event_blackout_days=int(event_blackout_days),
        event_blackout_cache=event_blackout_cache,
        alpha_decay_deweight_enabled=bool(alpha_decay_deweight_enabled),
        alpha_decay_qty_step=float(alpha_decay_qty_step),
        alpha_decay_qty_max_deweight=float(alpha_decay_qty_max_deweight),
        capacity_throttle_enabled=bool(capacity_throttle_enabled),
        capacity_spread_soft_bps=float(capacity_spread_soft_bps),
        capacity_spread_hard_bps=float(capacity_spread_hard_bps),
        capacity_volume_soft_participation=float(capacity_volume_soft_participation),
        capacity_volume_hard_participation=float(capacity_volume_hard_participation),
        capacity_min_scale=float(capacity_min_scale),
        capacity_adaptive_details=dict(capacity_adaptive_details),
        thin_spread_bps=float(thin_spread_bps),
        thin_vol_mult=float(thin_vol_mult),
        primary_feed_derisk=dict(primary_feed_derisk),
        portfolio_current_gross=float(portfolio_current_gross),
        sector_gross=sector_gross,
        symbol_adaptive_profiles=symbol_adaptive_profiles,
        ineffective_gate_blocklist=ineffective_gate_blocklist,
        ineffective_gate_diagnostics=ineffective_gate_diagnostics,
        gate_auto_disable_hysteresis_context=gate_auto_disable_hysteresis_context,
        uncertainty_capital_state=uncertainty_capital_state,
        uncertainty_cycle_events=uncertainty_cycle_events,
        penalty_overlap_coordination_enabled=bool(penalty_overlap_coordination_enabled),
        penalty_overlap_weight_dampen=float(penalty_overlap_weight_dampen),
        penalty_overlap_min_scale_floor=float(penalty_overlap_min_scale_floor),
    )
