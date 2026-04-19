"""Decision analytics and cycle-summary epilogue for the live netting cycle."""
from __future__ import annotations

from collections import Counter
import logging
from typing import Any, Callable, Mapping, Sequence


def _compute_reject_gate_counts(
    *,
    decision_gate_counts: Mapping[str, int],
    decision_observations: Sequence[Mapping[str, Any]],
    exclude_global_halts: bool,
    gate_name_is_halt_noise_func: Callable[[str], bool],
) -> dict[str, int]:
    if exclude_global_halts and decision_observations:
        filtered_reject_counts: Counter[str] = Counter()
        for observation in decision_observations:
            symbol_value = str(observation.get("symbol", "") or "").strip().upper()
            gates_raw = observation.get("gates")
            if isinstance(gates_raw, (str, bytes, bytearray)):
                gates_iterable: list[Any] = [gates_raw]
            elif isinstance(gates_raw, Sequence):
                gates_iterable = list(gates_raw)
            else:
                gates_iterable = []
            gates = [str(gate).strip() for gate in gates_iterable if str(gate).strip()]
            if symbol_value == "ALL" and any(
                gate_name_is_halt_noise_func(gate) for gate in gates
            ):
                continue
            if "OK_TRADE" in gates or bool(observation.get("accepted")):
                continue
            for gate in gates:
                if gate != "OK_TRADE":
                    filtered_reject_counts[gate] += 1
        return {
            gate: int(count)
            for gate, count in filtered_reject_counts.items()
            if int(count) > 0
        }

    return {
        gate: int(count)
        for gate, count in decision_gate_counts.items()
        if gate != "OK_TRADE" and int(count) > 0
    }


def finalize_netting_cycle_summary(
    *,
    runtime: Any,
    state: Any,
    now: Any,
    loop_start: float,
    symbols: Sequence[str],
    targets: Sequence[Any] | Mapping[str, Any],
    proposals_total: int,
    proposals_blocked: int,
    orders_attempted: int,
    orders_submitted: int,
    decision_recorder: Any,
    uncertainty_cycle_events: Sequence[Mapping[str, Any]],
    quarantine_enabled: bool,
    update_acceptance_rate_governor_state_func: Callable[..., Any],
    gate_effectiveness_exclude_global_halts_func: Callable[[], bool],
    gate_name_is_halt_noise_func: Callable[[str], bool],
    update_gate_effectiveness_analytics_func: Callable[..., Any],
    update_counterfactual_learning_analytics_func: Callable[..., Any],
    update_policy_ablation_analytics_func: Callable[..., Any],
    update_uncertainty_capital_analytics_func: Callable[..., Any],
    resolve_runtime_info_log_ttl_seconds_func: Callable[[str, float], float],
    should_emit_runtime_info_log_func: Callable[..., bool],
    log_throttled_event_func: Callable[..., Any],
    persist_quarantine_manager_func: Callable[[Any], None],
    monotonic_time_func: Callable[[], float],
    logger: Any,
    netting_cycle_slo_log_ttl_default: float,
) -> None:
    decision_gate_counts_raw = getattr(decision_recorder, "decision_gate_counts", {})
    decision_gate_counts = (
        dict(decision_gate_counts_raw)
        if isinstance(decision_gate_counts_raw, Mapping)
        else {}
    )
    decision_records_total = int(getattr(decision_recorder, "decision_records_total", 0) or 0)
    observations_raw = getattr(decision_recorder, "decision_observations", [])
    decision_observations = [
        dict(item) for item in observations_raw if isinstance(item, Mapping)
    ]
    accepted_decisions = int(decision_gate_counts.get("OK_TRADE", 0))

    update_acceptance_rate_governor_state_func(
        state,
        decision_records_total=decision_records_total,
        accepted_decisions=accepted_decisions,
    )

    reject_gate_counts = _compute_reject_gate_counts(
        decision_gate_counts=decision_gate_counts,
        decision_observations=decision_observations,
        exclude_global_halts=gate_effectiveness_exclude_global_halts_func(),
        gate_name_is_halt_noise_func=gate_name_is_halt_noise_func,
    )

    update_gate_effectiveness_analytics_func(
        decision_gate_counts={gate: int(count) for gate, count in decision_gate_counts.items()},
        decision_records_total=decision_records_total,
        accepted_decisions=accepted_decisions,
        decision_observations=decision_observations,
    )
    update_counterfactual_learning_analytics_func(
        observations=decision_observations,
        now=now,
    )
    update_policy_ablation_analytics_func(
        observations=decision_observations,
        now=now,
    )
    update_uncertainty_capital_analytics_func(
        events=uncertainty_cycle_events,
        now=now,
    )

    if reject_gate_counts:
        top_reasons = sorted(
            reject_gate_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:5]
        summary_payload = {
            "records_total": decision_records_total,
            "accepted_records": accepted_decisions,
            "rejected_records": max(decision_records_total - accepted_decisions, 0),
            "top_reasons": [{"reason": reason, "count": count} for reason, count in top_reasons],
            "unique_reasons": len(reject_gate_counts),
        }
        reject_ttl_s = resolve_runtime_info_log_ttl_seconds_func(
            "AI_TRADING_DECISION_REJECT_SUMMARY_LOG_TTL_SEC",
            60.0,
        )
        reject_signature = (
            "|".join(f"{reason}:{count}" for reason, count in top_reasons) or "none"
        )
        if should_emit_runtime_info_log_func(
            runtime,
            f"DECISION_REJECT_REASON_SUMMARY:{reject_signature}",
            ttl_seconds=reject_ttl_s,
        ):
            logger.info("DECISION_REJECT_REASON_SUMMARY", extra=summary_payload)

    cycle_elapsed_ms = int(max((monotonic_time_func() - float(loop_start)) * 1000.0, 0.0))
    slo_payload = {
        "symbols_requested": len(symbols),
        "targets": len(targets),
        "proposals_total": proposals_total,
        "proposals_blocked": proposals_blocked,
        "orders_attempted": orders_attempted,
        "orders_submitted": orders_submitted,
        "orders_skipped": max(orders_attempted - orders_submitted, 0),
        "compute_ms": cycle_elapsed_ms,
    }
    slo_ttl_s = resolve_runtime_info_log_ttl_seconds_func(
        "AI_TRADING_NETTING_CYCLE_SLO_LOG_TTL_SEC",
        netting_cycle_slo_log_ttl_default,
    )
    slo_signature = (
        f"{int(slo_payload['orders_submitted'])}:"
        f"{int(slo_payload['orders_skipped'])}:"
        f"{int(slo_payload['proposals_blocked'])}:"
        f"{int(slo_payload['targets'])}"
    )
    if should_emit_runtime_info_log_func(
        runtime,
        f"NETTING_CYCLE_SLO:{slo_signature}",
        ttl_seconds=slo_ttl_s,
    ):
        log_throttled_event_func(
            logger,
            f"NETTING_CYCLE_SLO_{slo_signature}",
            level=logging.INFO,
            extra=slo_payload,
            message="NETTING_CYCLE_SLO",
        )

    if quarantine_enabled:
        persist_quarantine_manager_func(state)


__all__ = ["finalize_netting_cycle_summary"]
