"""Build the operator-facing daily research report bundle."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.request import urlopen

from ai_trading.config.launch_profiles import launch_profile_payload, resolve_launch_profile
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.tools.symbol_promotion_comparison import symbol_promotion_digest


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _health_from_endpoint(url: str) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=5.0) as response:  # nosec B310 - local operator health endpoint
            parsed = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _nested(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def _default_path(relative: str) -> Path:
    return resolve_runtime_artifact_path(relative, default_relative=relative)


def _default_output(report_date: str) -> tuple[Path, Path, Path]:
    root = resolve_runtime_artifact_path(
        "runtime/research_reports",
        default_relative="runtime/research_reports",
        for_write=True,
    )
    return (
        root / f"daily_research_{report_date.replace('-', '')}.json",
        root / "daily_research_latest.json",
        root / f"daily_research_{report_date.replace('-', '')}.md",
    )


def _status(payload: Mapping[str, Any], default: str = "missing") -> str:
    raw = payload.get("status")
    if isinstance(raw, Mapping):
        return str(raw.get("status") or default)
    return str(raw or default)


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _promotion_ok(payload: Mapping[str, Any]) -> bool:
    return bool(payload.get("promotion_ready") or payload.get("status") == "ready_for_approval")


def _summary_status(payload: Mapping[str, Any], default: str = "missing") -> str:
    status = payload.get("status")
    if isinstance(status, Mapping):
        return str(status.get("status") or default)
    return str(status or default)


def _hf_summary(
    discovery: Mapping[str, Any],
    intake: Mapping[str, Any],
    cache: Mapping[str, Any],
) -> dict[str, Any]:
    discovery_summary = _nested(discovery, "summary")
    intake_summary = _nested(intake, "summary")
    cache_summary = _nested(cache, "summary")
    blocker_reasons: list[str] = []
    for payload in (discovery, intake, cache):
        raw = payload.get("blocked_reasons") if isinstance(payload, Mapping) else []
        if isinstance(raw, list):
            blocker_reasons.extend(str(item) for item in raw if item not in (None, ""))
    return {
        "available": bool(discovery or intake or cache),
        "status": {
            "discovery": _summary_status(discovery),
            "intake": _summary_status(intake),
            "cache": _summary_status(cache),
        },
        "research_only": True,
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
        "provider_authority": False,
        "manual_approval_required": True,
        "summary": {
            "candidates_scanned": discovery_summary.get("candidate_count", 0),
            "accepted_for_offline_experiment": (
                intake_summary.get("accepted_for_offline_experiment")
                or discovery_summary.get("accepted_for_offline_experiment")
                or 0
            ),
            "blocked_candidates": intake_summary.get("blocked", 0),
            "downloaded_to_cache": cache_summary.get("materialized", 0),
            "top_blocker_reasons": sorted(set(blocker_reasons))[:10],
        },
        "operator_action": (
            intake.get("operator_action")
            or discovery.get("operator_action")
            or cache.get("operator_action")
            or "no_hf_action_required"
        ),
    }


def _replay_status(payload: Mapping[str, Any]) -> dict[str, Any]:
    gate_raw = payload.get("replay_live_parity_gate")
    gate = dict(gate_raw) if isinstance(gate_raw, Mapping) else {}
    status = str(payload.get("status") or "").strip().lower()
    if not status:
        return gate
    ok_statuses = {"ok", "pass", "passed", "ready", "complete", "completed", "success"}
    blocked_statuses = {"blocked", "failed", "fail", "error", "not_ok", "non_ok"}
    top_level_ok = status in ok_statuses
    if status in blocked_statuses or not top_level_ok:
        gate["ok"] = False
        gate["reason"] = gate.get("reason") or payload.get("reason") or "replay_governance_failed"
    elif "ok" not in gate:
        gate["ok"] = True
    gate["status"] = str(gate.get("status") or status)
    gate["tool_status"] = status
    return gate


def _trade_allowed(report: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    runtime_health = _nested(report, "runtime_health")
    if not bool(runtime_health.get("ok")):
        reasons.append("runtime_health_not_ok")
    broker = _nested(runtime_health, "broker")
    if broker and broker.get("connected") is False:
        reasons.append("broker_not_connected")
    database = _nested(runtime_health, "database")
    if database and database.get("ok") is False:
        reasons.append("database_not_ok")
    for gate_name in ("oms_invariants", "oms_lifecycle_parity"):
        gate = _nested(runtime_health, gate_name)
        if gate and gate.get("ok") is False:
            reasons.append(f"{gate_name}_not_ok")
    attention_flags = report.get("runtime_attention_flags")
    if isinstance(attention_flags, list) and "market_closed_non_flat_positions" in {
        str(flag) for flag in attention_flags
    }:
        reasons.append("market_closed_non_flat_positions")
    provider_authority = _nested(report, "provider_authority")
    if provider_authority and provider_authority.get("ok") is False:
        reasons.append("provider_authority_not_ok")
    if str(_nested(report, "data_provider_health").get("status") or "").lower() not in {
        "healthy",
        "ready",
        "warming_up",
    }:
        reasons.append("data_provider_not_healthy")
    live_cost_status = _nested(report, "live_cost_status")
    if not bool(live_cost_status.get("available")):
        reasons.append("live_cost_unavailable")
    elif str(live_cost_status.get("status") or "").lower() not in {"ready", "ok"}:
        reasons.append("live_cost_not_ready")
    if int(live_cost_status.get("breach_count") or 0) > 0:
        reasons.append("live_cost_breach")
    replay_status = _nested(report, "replay_status")
    if replay_status and replay_status.get("ok") is False:
        reasons.append("replay_governance_failed")
    elif replay_status and str(replay_status.get("status") or "").lower() in {
        "blocked",
        "failed",
        "fail",
        "error",
    }:
        reasons.append("replay_governance_failed")
    runtime_gonogo = _nested(report, "runtime_gonogo")
    if bool(runtime_gonogo.get("available")) and runtime_gonogo.get("gate_passed") is False:
        reasons.append("runtime_gonogo_failed")
    launch_profile = _nested(report, "launch_profile")
    if str(launch_profile.get("name") or "").startswith("live_"):
        promotion_status = _nested(report, "promotion_status")
        if not bool(promotion_status.get("promotion_ready")):
            reasons.append("promotion_not_ready")
    if str(_nested(report, "memory_status").get("status") or "").lower() == "critical":
        reasons.append("memory_status_critical")
    portfolio_output = str(
        _nested(report, "portfolio_edge_control").get("output")
        or _nested(report, "portfolio_edge_control").get("status")
        or ""
    ).lower()
    if portfolio_output in {"no_new_entries"}:
        reasons.append("portfolio_edge_no_new_entries")
    if str(_nested(report, "execution_capture").get("status") or "").lower() == "degraded":
        reasons.append("execution_capture_degraded")
    risk_status = str(_nested(report, "pretrade_risk_control_verifier").get("status") or "").lower()
    if risk_status in {"failed", "blocked", "missing"}:
        reasons.append("pretrade_risk_controls_not_verified")
    surveillance_status = str(_nested(report, "post_trade_surveillance").get("status") or "").lower()
    if surveillance_status in {"critical", "blocked"}:
        reasons.append("post_trade_surveillance_critical")
    drift_status = str(_nested(report, "model_data_drift_monitor").get("status") or "").lower()
    if drift_status == "blocked":
        reasons.append("model_data_drift_monitor_blocked")
    if bool(_nested(report, "order_type_optimizer").get("live_enabled", False)):
        reasons.append("order_type_optimizer_live_enabled_unexpectedly")
    if bool(_nested(report, "walk_forward_capital_simulation").get("live_enabled", False)):
        reasons.append("walk_forward_capital_simulation_live_enabled_unexpectedly")
    return not reasons, reasons


def build_daily_research_report(
    *,
    report_date: str,
    health: Mapping[str, Any] | None = None,
    live_cost_model: Mapping[str, Any] | None = None,
    shadow_report: Mapping[str, Any] | None = None,
    replay_governance: Mapping[str, Any] | None = None,
    symbol_scorecard: Mapping[str, Any] | None = None,
    promotion_report: Mapping[str, Any] | None = None,
    symbol_promotion_comparison: Mapping[str, Any] | None = None,
    symbol_lifecycle: Mapping[str, Any] | None = None,
    replay_live_cost_alignment: Mapping[str, Any] | None = None,
    regime_entry_throttle: Mapping[str, Any] | None = None,
    execution_capture: Mapping[str, Any] | None = None,
    execution_capture_improvement: Mapping[str, Any] | None = None,
    counterfactual_execution: Mapping[str, Any] | None = None,
    portfolio_edge: Mapping[str, Any] | None = None,
    decision_receipts: Mapping[str, Any] | None = None,
    training_accelerator: Mapping[str, Any] | None = None,
    opportunity_markouts: Mapping[str, Any] | None = None,
    historical_backfill: Mapping[str, Any] | None = None,
    historical_training: Mapping[str, Any] | None = None,
    expected_edge_calibration: Mapping[str, Any] | None = None,
    evidence_starvation: Mapping[str, Any] | None = None,
    paper_sampling_state: Mapping[str, Any] | None = None,
    runtime_gonogo: Mapping[str, Any] | None = None,
    memory_audit: Mapping[str, Any] | None = None,
    artifact_retention: Mapping[str, Any] | None = None,
    model_registry: Mapping[str, Any] | None = None,
    pretrade_risk_verifier: Mapping[str, Any] | None = None,
    post_trade_surveillance: Mapping[str, Any] | None = None,
    experiment_ledger: Mapping[str, Any] | None = None,
    walk_forward_capital: Mapping[str, Any] | None = None,
    order_type_optimizer: Mapping[str, Any] | None = None,
    regime_champions: Mapping[str, Any] | None = None,
    adversarial_failure: Mapping[str, Any] | None = None,
    drift_monitor: Mapping[str, Any] | None = None,
    operator_control_plane: Mapping[str, Any] | None = None,
    weekend_research: Mapping[str, Any] | None = None,
    upward_trajectory: Mapping[str, Any] | None = None,
    metrics_improvement: Mapping[str, Any] | None = None,
    huggingface_discovery: Mapping[str, Any] | None = None,
    huggingface_candidate_intake: Mapping[str, Any] | None = None,
    huggingface_cache_materialization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    health = health or {}
    live_cost_model = live_cost_model or {}
    shadow_report = shadow_report or {}
    replay_governance = replay_governance or {}
    symbol_scorecard = symbol_scorecard or {}
    promotion_report = promotion_report or {}
    symbol_promotion_comparison = symbol_promotion_comparison or {}
    symbol_lifecycle = symbol_lifecycle or {}
    replay_live_cost_alignment = replay_live_cost_alignment or {}
    regime_entry_throttle = regime_entry_throttle or {}
    execution_capture = execution_capture or {}
    execution_capture_improvement = execution_capture_improvement or {}
    counterfactual_execution = counterfactual_execution or {}
    portfolio_edge = portfolio_edge or {}
    decision_receipts = decision_receipts or {}
    training_accelerator = training_accelerator or {}
    opportunity_markouts = opportunity_markouts or {}
    historical_backfill = historical_backfill or {}
    historical_training = historical_training or {}
    expected_edge_calibration = expected_edge_calibration or {}
    evidence_starvation = evidence_starvation or {}
    paper_sampling_state = paper_sampling_state or {}
    runtime_gonogo = runtime_gonogo or {}
    runtime_gonogo_payload = _nested(runtime_gonogo, "go_no_go") or runtime_gonogo
    memory_audit = memory_audit or {}
    artifact_retention = artifact_retention or {}
    model_registry = model_registry or {}
    pretrade_risk_verifier = pretrade_risk_verifier or {}
    post_trade_surveillance = post_trade_surveillance or {}
    experiment_ledger = experiment_ledger or {}
    walk_forward_capital = walk_forward_capital or {}
    order_type_optimizer = order_type_optimizer or {}
    regime_champions = regime_champions or {}
    adversarial_failure = adversarial_failure or {}
    drift_monitor = drift_monitor or {}
    operator_control_plane = operator_control_plane or {}
    weekend_research = weekend_research or {}
    upward_trajectory = upward_trajectory or {}
    metrics_improvement = metrics_improvement or {}
    huggingface_discovery = huggingface_discovery or {}
    huggingface_candidate_intake = huggingface_candidate_intake or {}
    huggingface_cache_materialization = huggingface_cache_materialization or {}
    historical_symbol_rows = historical_backfill.get("symbols")
    historical_rows = sum(
        _nonnegative_int(row.get("row_count"))
        for row in historical_symbol_rows
        if isinstance(row, Mapping)
    ) if isinstance(historical_symbol_rows, list) else 0
    model_shadow_rows = _nonnegative_int(
        shadow_report.get("filtered_rows", shadow_report.get("raw_rows", 0))
    )
    opportunity_markout_rows = _nonnegative_int(
        opportunity_markouts.get("outcomes_emitted")
    )
    shadow_rows = (
        opportunity_markout_rows if opportunity_markouts else model_shadow_rows
    )
    paper_fill_rows = _nonnegative_int(
        _nested(execution_capture, "sample_gate").get("samples")
    )
    huggingface_research = _hf_summary(
        huggingface_discovery,
        huggingface_candidate_intake,
        huggingface_cache_materialization,
    )
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "artifact_type": "daily_research_report",
        "report_date": report_date,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "runtime_health": {
            "ok": bool(health.get("ok")),
            "status": health.get("status"),
            "reason": health.get("reason"),
            "broker": _nested(health, "broker"),
            "database": _nested(health, "database"),
            "oms_invariants": _nested(health, "oms_invariants"),
            "oms_lifecycle_parity": _nested(health, "oms_lifecycle_parity"),
        },
        "runtime_attention_flags": list(health.get("attention_flags", []))
        if isinstance(health.get("attention_flags"), list)
        else [],
        "data_provider_health": _nested(health, "data_provider"),
        "provider_authority": _nested(health, "provider_authority"),
        "launch_profile": launch_profile_payload(resolve_launch_profile()),
        "live_cost_status": _nested(live_cost_model, "status"),
        "shadow_status": {
            "status": _status(_nested(shadow_report, "sample_gate")),
            "decision_summary": _nested(shadow_report, "decision_summary"),
            "sample_gate": _nested(shadow_report, "sample_gate"),
        },
        "opportunity_markouts": {
            "available": bool(opportunity_markouts),
            "eligible_opportunities": _nonnegative_int(
                opportunity_markouts.get("eligible_opportunities")
            ),
            "outcomes_emitted": opportunity_markout_rows,
            "horizons_bars": opportunity_markouts.get("horizons_bars", []),
            "label_status_counts": _nested(
                opportunity_markouts,
                "label_status_counts",
            ),
            "label_reason_counts": _nested(
                opportunity_markouts,
                "label_reason_counts",
            ),
            "bars_provenance": _nested(
                opportunity_markouts,
                "bars_provenance",
            ),
            "model_shadow_telemetry_samples": model_shadow_rows,
            "evidence_type": "shadow_counterfactual",
            "promotion_eligible": False,
            "promotion_authority": False,
            "runtime_authority": False,
            "live_money_authority": False,
        },
        "replay_status": _replay_status(replay_governance),
        "promotion_status": {
            "status": promotion_report.get("status", "missing"),
            "promotion_ready": _promotion_ok(promotion_report),
            "gates": _nested(promotion_report, "gates"),
        },
        "runtime_gonogo": {
            "available": bool(runtime_gonogo_payload),
            "gate_passed": bool(runtime_gonogo_payload.get("gate_passed"))
            if runtime_gonogo_payload
            else None,
            "failed_checks": list(runtime_gonogo_payload.get("failed_checks", []))
            if isinstance(runtime_gonogo_payload.get("failed_checks"), list)
            else [],
        },
        "memory_status": {
            "status": memory_audit.get("status", "missing"),
            "service_memory": _nested(memory_audit, "service_memory"),
            "recent_memory_samples": _nested(memory_audit, "recent_memory_samples"),
            "observations": list(memory_audit.get("observations", []))
            if isinstance(memory_audit.get("observations"), list)
            else [],
        },
        "artifact_retention": {
            "status": artifact_retention.get("status", "missing"),
            "apply": bool(artifact_retention.get("apply", False)),
            "total_reclaimable_mb": artifact_retention.get("total_reclaimable_mb"),
        },
        "symbol_actions": {
            "summary": _nested(symbol_scorecard, "summary"),
            "policy": _nested(symbol_scorecard, "policy"),
            "shadow_promotion": _nested(symbol_scorecard, "shadow_promotion"),
            "symbols": symbol_scorecard.get("symbols", []),
        },
        "symbol_promotion": {
            "available": bool(symbol_promotion_comparison.get("symbols")),
            "promotion_authority": bool(symbol_promotion_comparison.get("promotion_authority", False)),
            "runtime_symbol_gating_changed": bool(
                symbol_promotion_comparison.get("runtime_symbol_gating_changed", False)
            ),
            "status": symbol_promotion_comparison.get("status", "missing"),
            "summary": _nested(symbol_promotion_comparison, "summary"),
            "digest": symbol_promotion_digest(symbol_promotion_comparison),
            "symbols": symbol_promotion_comparison.get("symbols", []),
        },
        "symbol_lifecycle": {
            "available": bool(symbol_lifecycle.get("symbols")),
            "status": symbol_lifecycle.get("status", "missing"),
            "summary": _nested(symbol_lifecycle, "summary"),
            "manual_approval_required_for_authority_increase": bool(
                symbol_lifecycle.get("manual_approval_required_for_authority_increase", True)
            ),
            "runtime_symbol_gating_changed": bool(
                symbol_lifecycle.get("runtime_symbol_gating_changed", False)
            ),
            "symbols": symbol_lifecycle.get("symbols", []),
        },
        "replay_live_cost_alignment": {
            "available": bool(replay_live_cost_alignment.get("summary")),
            "status": replay_live_cost_alignment.get("status", "missing"),
            "cost_realism": _nested(replay_live_cost_alignment, "cost_realism"),
            "summary": _nested(replay_live_cost_alignment, "summary"),
        },
        "regime_entry_throttle": {
            "available": bool(regime_entry_throttle.get("actions") or regime_entry_throttle.get("latest")),
            "status": regime_entry_throttle.get("status", "ready" if regime_entry_throttle else "missing"),
            "mode": regime_entry_throttle.get("mode"),
            "actions": regime_entry_throttle.get("actions", {}),
            "policy_actions": _nested(regime_entry_throttle, "policy_actions"),
            "latest": _nested(regime_entry_throttle, "latest"),
        },
        "execution_capture": {
            "available": bool(execution_capture),
            "status": execution_capture.get("status", "missing"),
            "sample_gate": _nested(execution_capture, "sample_gate"),
            "summary": _nested(execution_capture, "summary"),
            "by_symbol": _nested(execution_capture, "by_symbol"),
            "promotion_authority": bool(execution_capture.get("promotion_authority", False)),
        },
        "execution_capture_improvement": {
            "available": bool(execution_capture_improvement),
            "status": execution_capture_improvement.get("status", "missing"),
            "recommended_next_action": execution_capture_improvement.get("recommended_next_action"),
            "summary": _nested(execution_capture_improvement, "summary"),
            "bad_buckets": _nested(execution_capture_improvement, "bad_buckets"),
            "edge_haircuts": _nested(execution_capture_improvement, "edge_haircuts"),
            "training_labels": _nested(execution_capture_improvement, "training_labels"),
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
        },
        "counterfactual_execution": {
            "available": bool(counterfactual_execution),
            "status": counterfactual_execution.get("status", "missing"),
            "summary": _nested(counterfactual_execution, "summary"),
            "promotion_authority": bool(counterfactual_execution.get("promotion_authority", False)),
        },
        "portfolio_edge_control": {
            "available": bool(portfolio_edge),
            "status": portfolio_edge.get("status", "missing"),
            "output": portfolio_edge.get("output") or portfolio_edge.get("status"),
            "flags": list(portfolio_edge.get("flags", []))
            if isinstance(portfolio_edge.get("flags"), list)
            else list(_nested(portfolio_edge, "controls").get("breaches", []))
            if isinstance(_nested(portfolio_edge, "controls").get("breaches"), list)
            else [],
            "controls": _nested(portfolio_edge, "controls"),
            "summary": _nested(portfolio_edge, "summary"),
        },
        "decision_receipts": {
            "available": bool(decision_receipts),
            "status": decision_receipts.get("status", "missing"),
            "summary": _nested(decision_receipts, "summary"),
            "live_money_authority": bool(decision_receipts.get("live_money_authority", False)),
        },
        "training_accelerator": {
            "available": bool(training_accelerator),
            "status": training_accelerator.get("status", "missing"),
            "promotion_authority": bool(training_accelerator.get("promotion_authority", False)),
            "cache": _nested(training_accelerator, "cache"),
            "input_signature": training_accelerator.get("input_signature"),
            "timing": _nested(training_accelerator, "timing"),
            "ranked_candidate_count": training_accelerator.get("ranked_candidate_count"),
            "lead_candidate_count": training_accelerator.get("lead_candidate_count"),
        },
        "evidence_collection": {
            "promotion_policy": "executed_paper_or_live_fill_evidence_only",
            "promotion_eligible_sample_count": paper_fill_rows,
            "sources": {
                "historical_research": {
                    "samples": historical_rows,
                    "available": bool(historical_backfill),
                    "quality_passed": bool(
                        historical_backfill.get("quality_passed", False)
                    ),
                    "evidence_type": "historical_research",
                    "promotion_eligible": False,
                    "promotion_exclusion_reason": (
                        "historical_research_is_not_execution_evidence"
                    ),
                },
                "shadow_counterfactual": {
                    "samples": shadow_rows,
                    "available": bool(opportunity_markouts or shadow_report),
                    "source": (
                        "universal_opportunity_markouts"
                        if opportunity_markouts
                        else "model_shadow_telemetry"
                    ),
                    "evidence_type": "shadow_counterfactual",
                    "promotion_eligible": False,
                    "promotion_exclusion_reason": (
                        "shadow_counterfactual_is_not_fill_evidence"
                    ),
                },
                "paper_fill": {
                    "samples": paper_fill_rows,
                    "available": bool(execution_capture),
                    "evidence_type": "paper_fill",
                    "promotion_eligible": True,
                    "eligibility_scope": "sample_threshold_evidence_only",
                },
            },
            "excluded_from_promotion": {
                "historical_research": historical_rows,
                "shadow_counterfactual": shadow_rows,
            },
        },
        "historical_training": {
            "available": bool(historical_training),
            "status": historical_training.get(
                "status",
                "ready" if historical_training else "missing",
            ),
            "dataset_hash": (
                _nested(historical_training, "acquisition").get("dataset_hash")
                or _nested(historical_training, "dataset").get("dataset_hash")
            ),
            "walk_forward": _nested(historical_training, "walk_forward"),
            "authority": _nested(historical_training, "authority"),
            "promotion_eligible": False,
            "promotion_authority": False,
            "live_money_authority": False,
        },
        "expected_edge_calibration": {
            "available": bool(expected_edge_calibration),
            "status": expected_edge_calibration.get("status", "missing"),
            "recommended_next_action": expected_edge_calibration.get("recommended_next_action"),
            "sample_gate": _nested(expected_edge_calibration, "sample_gate"),
            "summary": _nested(expected_edge_calibration, "summary"),
            "execution_capture_diagnosis": _nested(
                expected_edge_calibration,
                "execution_capture_diagnosis",
            ),
            "promotion_authority": bool(
                expected_edge_calibration.get("promotion_authority", False)
            ),
        },
        "execution_capture_diagnosis": _nested(
            expected_edge_calibration,
            "execution_capture_diagnosis",
        ),
        "evidence_starvation": {
            "available": bool(evidence_starvation),
            "status": evidence_starvation.get("status", "missing"),
            "recommendation": evidence_starvation.get("recommendation"),
            "counts": _nested(evidence_starvation, "counts"),
            "estimated_days_until_sample_sufficiency": evidence_starvation.get(
                "estimated_days_until_sample_sufficiency"
            ),
            "hard_safety_blockers": list(evidence_starvation.get("hard_safety_blockers", []))
            if isinstance(evidence_starvation.get("hard_safety_blockers"), list)
            else [],
        },
        "diagnostic_sampling": {
            "activity": evidence_starvation.get("recommendation"),
            "status": evidence_starvation.get("status", "missing"),
            "state": dict(paper_sampling_state),
            "paper_only": True,
            "live_money_authority": False,
        },
        "long_only_side_semantics": {
            "status": "reported_by_trading_day_and_gate_events",
            "open_short_blockers_visible": True,
            "sell_to_close_allowed": True,
        },
        "trade_quality_labels": {
            "available": bool(training_accelerator),
            "status": training_accelerator.get("status", "missing"),
            "objectives": _nested(training_accelerator, "config").get("label_objectives"),
            "shadow_only": True,
            "promotion_authority": False,
        },
        "model_registry": {
            "available": bool(model_registry),
            "status": model_registry.get("status", "missing"),
            "artifact_type": model_registry.get("artifact_type"),
            "summary": _nested(model_registry, "summary"),
            "active_champion": _nested(model_registry, "active_champion"),
            "active_challenger": _nested(model_registry, "active_challenger"),
            "identity_discovery": _nested(model_registry, "identity_discovery"),
            "blocked_reasons": list(model_registry.get("blocked_reasons", []))
            if isinstance(model_registry.get("blocked_reasons"), list)
            else [],
            "promotion_authority": False,
            "live_money_authority": False,
            "manual_approval_required": True,
        },
        "pretrade_risk_control_verifier": {
            "available": bool(pretrade_risk_verifier),
            "status": pretrade_risk_verifier.get("status", "missing"),
            "fail_closed": bool(pretrade_risk_verifier.get("fail_closed", True)),
            "summary": _nested(pretrade_risk_verifier, "summary"),
            "violations": pretrade_risk_verifier.get("violations", []),
            "live_money_authority": False,
        },
        "post_trade_surveillance": {
            "available": bool(post_trade_surveillance),
            "status": post_trade_surveillance.get("status", "missing"),
            "summary": _nested(post_trade_surveillance, "summary"),
            "findings": post_trade_surveillance.get("findings", []),
            "live_money_authority": False,
        },
        "experiment_ledger": {
            "available": bool(experiment_ledger),
            "status": experiment_ledger.get("status", "missing"),
            "latest_run": _nested(experiment_ledger, "latest_run"),
            "completion_guard": _nested(experiment_ledger, "completion_guard"),
            "reported_complete": experiment_ledger.get("reported_complete"),
        },
        "walk_forward_capital_simulation": {
            "available": bool(walk_forward_capital),
            "status": walk_forward_capital.get("status", "missing"),
            "summary": _nested(walk_forward_capital, "summary"),
            "live_enabled": bool(walk_forward_capital.get("live_enabled", False)),
        },
        "order_type_optimizer": {
            "available": bool(order_type_optimizer),
            "status": order_type_optimizer.get("status", "missing"),
            "mode": order_type_optimizer.get("mode"),
            "summary": _nested(order_type_optimizer, "summary"),
            "recommendations_enabled": bool(order_type_optimizer.get("recommendations_enabled", False)),
            "live_enabled": bool(order_type_optimizer.get("live_enabled", False)),
        },
        "regime_champion_models": {
            "available": bool(regime_champions),
            "status": regime_champions.get("status", "missing"),
            "summary": _nested(regime_champions, "summary"),
            "blocked_regimes": regime_champions.get("blocked_regimes", []),
            "manual_approval_required": True,
        },
        "adversarial_failure_simulation": {
            "available": bool(adversarial_failure),
            "status": adversarial_failure.get("status", "missing"),
            "summary": _nested(adversarial_failure, "summary"),
            "live_money_authority": bool(adversarial_failure.get("live_money_authority", False)),
        },
        "model_data_drift_monitor": {
            "available": bool(drift_monitor),
            "status": drift_monitor.get("status", "missing"),
            "summary": _nested(drift_monitor, "summary"),
            "reasons": drift_monitor.get("reasons", []),
        },
        "operator_control_plane": {
            "available": bool(operator_control_plane),
            "status": operator_control_plane.get("status", "missing"),
            "summary": _nested(operator_control_plane, "summary"),
            "read_only": bool(operator_control_plane.get("read_only", True)),
        },
        "weekend_research": {
            "available": bool(weekend_research),
            "status": weekend_research.get("status", "missing"),
            "cadence": weekend_research.get("cadence"),
            "monday_preparation": weekend_research.get("monday_preparation"),
            "research_only": True,
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
            "manual_approval_required": True,
        },
        "upward_trajectory": {
            "available": bool(upward_trajectory),
            "status": upward_trajectory.get("status", "missing"),
            "summary": _nested(upward_trajectory, "summary"),
            "authority": _nested(upward_trajectory, "authority"),
            "evidence_acceleration_engine": _nested(
                upward_trajectory,
                "evidence_acceleration_engine",
            ),
            "validation_to_replay_gap_analyzer": _nested(
                upward_trajectory,
                "validation_to_replay_gap_analyzer",
            ),
            "candidate_tournament_system": _nested(
                upward_trajectory,
                "candidate_tournament_system",
            ),
            "active_learning_paper_trades": _nested(
                upward_trajectory,
                "active_learning_paper_trades",
            ),
            "runtime_authority": False,
            "promotion_authority": bool(
                _nested(upward_trajectory, "authority").get("promotion_authority", False)
            ),
            "live_money_authority": bool(
                _nested(upward_trajectory, "authority").get("live_money_authority", False)
            ),
        },
        "metrics_improvement_control": {
            "available": bool(metrics_improvement),
            "status": metrics_improvement.get("status", "missing"),
            "summary": _nested(metrics_improvement, "summary"),
            "runtime_safety_control": bool(
                metrics_improvement.get("runtime_safety_control", False)
            ),
            "authority_increase_allowed": bool(
                metrics_improvement.get("authority_increase_allowed", False)
            ),
            "promotion_authority": bool(metrics_improvement.get("promotion_authority", False)),
            "live_money_authority": bool(metrics_improvement.get("live_money_authority", False)),
        },
        "huggingface_research": huggingface_research,
    }
    allowed, reasons = _trade_allowed(report)
    profile_name = str(_nested(report, "launch_profile").get("name") or "paper_observe")
    report["status"] = "ready" if allowed else "blocked"
    report["trade_allowed"] = bool(allowed)
    report["blocked_reasons"] = reasons
    report["recommended_next_session_mode"] = (
        profile_name if allowed else ("paper_only" if profile_name.startswith("live_") else "observe")
    )
    report["next_session_limits"] = {
        "profile": profile_name,
        "allowed_symbols": _nested(report, "launch_profile").get("allowed_symbols", []),
        "max_order_count": _nested(report, "launch_profile").get("max_order_count"),
        "max_notional_per_order": _nested(report, "launch_profile").get("max_notional_per_order"),
        "max_daily_loss": _nested(report, "launch_profile").get("max_daily_loss"),
        "max_quote_age_ms": _nested(report, "launch_profile").get("max_quote_age_ms"),
        "max_spread_bps": _nested(report, "launch_profile").get("max_spread_bps"),
        "shorts_allowed": _nested(report, "launch_profile").get("shorts_allowed"),
        "execution_quote_authority": _nested(report, "launch_profile").get("execution_quote_authority"),
        "backup_provider_live_policy": _nested(report, "launch_profile").get("backup_provider_live_policy"),
        "provider_authority_ok": _nested(report, "provider_authority").get("ok"),
    }
    report["health_report_summary"] = {
        "runtime_status": _summary_status(_nested(report, "runtime_health")),
        "runtime_ok": bool(_nested(report, "runtime_health").get("ok")),
        "data_provider_status": _summary_status(_nested(report, "data_provider_health")),
        "provider_authority_ok": _nested(report, "provider_authority").get("ok"),
        "live_cost_status": _summary_status(_nested(report, "live_cost_status")),
        "replay_status": _summary_status(_nested(report, "replay_status")),
        "runtime_gonogo_passed": _nested(report, "runtime_gonogo").get("gate_passed"),
        "memory_status": _summary_status(_nested(report, "memory_status")),
        "trade_allowed": bool(allowed),
        "blocked_reasons": reasons,
    }
    report["next_level_artifacts"] = {
        "trading_day_report": {
            "status": "available_in_trading_day_latest",
            "latest_path": "runtime/research_reports/latest/trading_day_latest.json",
        },
        "live_capital_readiness": {
            "status": "generated_after_daily_research",
            "latest_path": "runtime/research_reports/latest/live_capital_readiness_latest.json",
            "live_money_authority": False,
            "manual_approval_required": True,
        },
        "expected_edge_calibration": {
            "status": _summary_status(_nested(report, "expected_edge_calibration")),
            "recommended_next_action": _nested(report, "expected_edge_calibration").get(
                "recommended_next_action"
            ),
            "promotion_authority": bool(
                _nested(report, "expected_edge_calibration").get("promotion_authority", False)
            ),
        },
        "evidence_starvation": {
            "status": _summary_status(_nested(report, "evidence_starvation")),
            "recommendation": _nested(report, "evidence_starvation").get("recommendation"),
            "live_money_authority": False,
        },
        "diagnostic_sampling": {
            "status": _summary_status(_nested(report, "diagnostic_sampling")),
            "activity": _nested(report, "diagnostic_sampling").get("activity"),
            "paper_only": True,
        },
        "trade_quality_labels": {
            "status": _summary_status(_nested(report, "trade_quality_labels")),
            "shadow_only": True,
            "promotion_authority": False,
        },
        "model_registry": {
            "status": _summary_status(_nested(report, "model_registry")),
            "promotion_authority": False,
            "live_money_authority": False,
            "active_challenger": _nested(report, "model_registry", "active_challenger"),
            "blocked_reasons": _nested(report, "model_registry").get("blocked_reasons", []),
            "manual_approval_required": True,
        },
        "pretrade_risk_control_verifier": {
            "status": _summary_status(_nested(report, "pretrade_risk_control_verifier")),
            "fail_closed": bool(_nested(report, "pretrade_risk_control_verifier").get("fail_closed", True)),
        },
        "post_trade_surveillance": {
            "status": _summary_status(_nested(report, "post_trade_surveillance")),
            "summary": _nested(report, "post_trade_surveillance").get("summary"),
        },
        "experiment_ledger": {
            "status": _summary_status(_nested(report, "experiment_ledger")),
            "latest_run": _nested(report, "experiment_ledger").get("latest_run"),
        },
        "walk_forward_capital_simulation": {
            "status": _summary_status(_nested(report, "walk_forward_capital_simulation")),
            "live_enabled": bool(_nested(report, "walk_forward_capital_simulation").get("live_enabled", False)),
        },
        "order_type_optimizer": {
            "status": _summary_status(_nested(report, "order_type_optimizer")),
            "live_enabled": bool(_nested(report, "order_type_optimizer").get("live_enabled", False)),
        },
        "regime_champion_models": {
            "status": _summary_status(_nested(report, "regime_champion_models")),
            "manual_approval_required": True,
        },
        "adversarial_failure_simulation": {
            "status": _summary_status(_nested(report, "adversarial_failure_simulation")),
            "live_money_authority": bool(
                _nested(report, "adversarial_failure_simulation").get("live_money_authority", False)
            ),
        },
        "model_data_drift_monitor": {
            "status": _summary_status(_nested(report, "model_data_drift_monitor")),
            "summary": _nested(report, "model_data_drift_monitor").get("summary"),
        },
        "operator_control_plane": {
            "status": _summary_status(_nested(report, "operator_control_plane")),
            "read_only": bool(_nested(report, "operator_control_plane").get("read_only", True)),
        },
        "weekend_research": {
            "status": _summary_status(_nested(report, "weekend_research")),
            "monday_preparation": _nested(report, "weekend_research").get("monday_preparation"),
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
            "manual_approval_required": True,
        },
        "upward_trajectory": {
            "status": _summary_status(_nested(report, "upward_trajectory")),
            "recommended_next_action": _nested(
                report,
                "upward_trajectory",
                "summary",
            ).get("recommended_next_action"),
            "candidate_count": _nested(report, "upward_trajectory", "summary").get(
                "candidate_count"
            ),
            "runtime_authority": False,
            "promotion_authority": bool(
                _nested(report, "upward_trajectory").get("promotion_authority", False)
            ),
            "live_money_authority": bool(
                _nested(report, "upward_trajectory").get("live_money_authority", False)
            ),
        },
        "metrics_improvement_control": {
            "status": _summary_status(_nested(report, "metrics_improvement_control")),
            "summary": _nested(report, "metrics_improvement_control").get("summary"),
            "runtime_safety_control": bool(
                _nested(report, "metrics_improvement_control").get(
                    "runtime_safety_control",
                    False,
                )
            ),
            "authority_increase_allowed": bool(
                _nested(report, "metrics_improvement_control").get(
                    "authority_increase_allowed",
                    False,
                )
            ),
        },
        "huggingface_research": {
            "status": _nested(report, "huggingface_research").get("status"),
            "summary": _nested(report, "huggingface_research").get("summary"),
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
        },
    }
    report["openclaw_summary"] = {
        "service": "ai-trading-research",
        "severity": "info" if allowed else "warning",
        "summary": (
            f"daily_research trade_allowed={str(bool(allowed)).lower()} "
            f"mode={report['recommended_next_session_mode']}"
        ),
        "suggested_action": (
            "review live_capital_readiness before live cutover"
            if allowed
            else "resolve daily research blockers before next session"
        ),
        "blocked_reasons": reasons,
        "details": {
            "health_report_summary": report["health_report_summary"],
            "next_level_artifacts": report["next_level_artifacts"],
        },
    }
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    reasons = report.get("blocked_reasons") if isinstance(report.get("blocked_reasons"), list) else []
    shadow_suggestions = _nested(report, "symbol_actions", "shadow_promotion").get(
        "suggestions",
        [],
    )
    suggestion_text = "none"
    if isinstance(shadow_suggestions, list) and shadow_suggestions:
        suggestion_text = ", ".join(
            str(item.get("symbol"))
            for item in shadow_suggestions
            if isinstance(item, Mapping) and item.get("symbol")
        ) or "none"
    return "\n".join(
        [
            f"# Daily Research {report.get('report_date')}",
            "",
            f"- Trade allowed: `{str(report.get('trade_allowed')).lower()}`",
            f"- Recommended next session mode: `{report.get('recommended_next_session_mode')}`",
            f"- Blockers: {', '.join(str(item) for item in reasons) if reasons else 'none'}",
            f"- Limits: `{_nested(report, 'next_session_limits').get('profile')}` "
            f"orders={_nested(report, 'next_session_limits').get('max_order_count')} "
            f"notional={_nested(report, 'next_session_limits').get('max_notional_per_order')} "
            f"symbols={_nested(report, 'next_session_limits').get('allowed_symbols')}",
            f"- Runtime health: `{_nested(report, 'runtime_health').get('status')}`",
            f"- Data provider: `{_nested(report, 'data_provider_health').get('status')}`",
            f"- Live cost: `{_nested(report, 'live_cost_status').get('status', 'missing')}`",
            f"- Memory: `{_nested(report, 'memory_status').get('status', 'missing')}`",
            f"- Promotion: `{_nested(report, 'promotion_status').get('status', 'missing')}`",
            f"- Shadow promotion candidates: {suggestion_text}",
            f"- Symbol promotion advisory: {_nested(report, 'symbol_promotion').get('digest', 'none')}",
            f"- Symbol lifecycle: `{_nested(report, 'symbol_lifecycle').get('status', 'missing')}`",
            f"- Replay/live cost realism: `{_nested(report, 'replay_live_cost_alignment', 'cost_realism').get('status', 'missing')}`",
            f"- Regime throttle: `{_nested(report, 'regime_entry_throttle').get('actions', {})}`",
            f"- Execution capture: `{_nested(report, 'execution_capture').get('status', 'missing')}`",
            f"- Execution capture improvement: `{_nested(report, 'execution_capture_improvement').get('status', 'missing')}` "
            f"action=`{_nested(report, 'execution_capture_improvement').get('recommended_next_action', 'missing')}`",
            f"- Counterfactual execution: `{_nested(report, 'counterfactual_execution').get('status', 'missing')}`",
            f"- Portfolio edge: `{_nested(report, 'portfolio_edge_control').get('output', 'missing')}`",
            f"- Decision receipts: `{_nested(report, 'decision_receipts').get('status', 'missing')}`",
            f"- Training accelerator: `{_nested(report, 'training_accelerator').get('status', 'missing')}`",
            f"- Expected-edge calibration: `{_nested(report, 'expected_edge_calibration').get('status', 'missing')}`",
            f"- Evidence starvation: `{_nested(report, 'evidence_starvation').get('status', 'missing')}`",
            f"- Model registry: `{_nested(report, 'model_registry').get('status', 'missing')}`",
            f"- Active shadow challenger: `{_nested(report, 'model_registry', 'active_challenger').get('model_id', 'missing')}`",
            f"- Pre-trade risk verifier: `{_nested(report, 'pretrade_risk_control_verifier').get('status', 'missing')}`",
            f"- Post-trade surveillance: `{_nested(report, 'post_trade_surveillance').get('status', 'missing')}`",
            f"- Experiment ledger: `{_nested(report, 'experiment_ledger').get('status', 'missing')}`",
            f"- Walk-forward capital: `{_nested(report, 'walk_forward_capital_simulation').get('status', 'missing')}`",
            f"- Order-type optimizer: `{_nested(report, 'order_type_optimizer').get('status', 'missing')}`",
            f"- Regime champions: `{_nested(report, 'regime_champion_models').get('status', 'missing')}`",
            f"- Upward trajectory: `{_nested(report, 'upward_trajectory').get('status', 'missing')}` "
            f"action=`{_nested(report, 'upward_trajectory', 'summary').get('recommended_next_action', 'missing')}`",
            f"- Metrics improvement control: `{_nested(report, 'metrics_improvement_control').get('status', 'missing')}`",
            f"- Adversarial simulation: `{_nested(report, 'adversarial_failure_simulation').get('status', 'missing')}`",
            f"- Drift monitor: `{_nested(report, 'model_data_drift_monitor').get('status', 'missing')}`",
            f"- Operator control plane: `{_nested(report, 'operator_control_plane').get('status', 'missing')}`",
            f"- Hugging Face research: `{_nested(report, 'huggingface_research', 'status').get('discovery', 'missing')}` "
            f"accepted={_nested(report, 'huggingface_research', 'summary').get('accepted_for_offline_experiment', 0)} "
            f"runtime_authority=false",
            f"- Health/report summary: `{_nested(report, 'health_report_summary').get('runtime_status', 'missing')}` "
            f"trade_allowed={str(report.get('trade_allowed')).lower()}",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--health-json", type=Path, default=None)
    parser.add_argument("--health-url", default="http://127.0.0.1:9001/healthz")
    parser.add_argument("--live-cost-model-json", type=Path, default=None)
    parser.add_argument("--shadow-report-json", type=Path, default=None)
    parser.add_argument("--replay-governance-json", type=Path, default=None)
    parser.add_argument("--symbol-scorecard-json", type=Path, default=None)
    parser.add_argument("--promotion-report-json", type=Path, default=None)
    parser.add_argument("--symbol-promotion-json", type=Path, default=None)
    parser.add_argument("--symbol-lifecycle-json", type=Path, default=None)
    parser.add_argument("--replay-live-cost-alignment-json", type=Path, default=None)
    parser.add_argument("--regime-entry-throttle-json", type=Path, default=None)
    parser.add_argument("--execution-capture-json", type=Path, default=None)
    parser.add_argument("--execution-capture-improvement-json", type=Path, default=None)
    parser.add_argument("--counterfactual-execution-json", type=Path, default=None)
    parser.add_argument("--portfolio-edge-json", type=Path, default=None)
    parser.add_argument("--decision-receipts-json", type=Path, default=None)
    parser.add_argument("--training-accelerator-json", type=Path, default=None)
    parser.add_argument("--opportunity-markouts-json", type=Path, default=None)
    parser.add_argument("--historical-backfill-json", type=Path, default=None)
    parser.add_argument("--historical-training-json", type=Path, default=None)
    parser.add_argument("--expected-edge-calibration-json", type=Path, default=None)
    parser.add_argument("--evidence-starvation-json", type=Path, default=None)
    parser.add_argument("--paper-sampling-state-json", type=Path, default=None)
    parser.add_argument("--runtime-gonogo-json", type=Path, default=None)
    parser.add_argument("--memory-audit-json", type=Path, default=None)
    parser.add_argument("--artifact-retention-json", type=Path, default=None)
    parser.add_argument("--model-registry-json", type=Path, default=None)
    parser.add_argument("--pretrade-risk-json", type=Path, default=None)
    parser.add_argument("--post-trade-surveillance-json", type=Path, default=None)
    parser.add_argument("--experiment-ledger-json", type=Path, default=None)
    parser.add_argument("--walk-forward-capital-json", type=Path, default=None)
    parser.add_argument("--order-type-optimizer-json", type=Path, default=None)
    parser.add_argument("--regime-champions-json", type=Path, default=None)
    parser.add_argument("--adversarial-failure-json", type=Path, default=None)
    parser.add_argument("--drift-monitor-json", type=Path, default=None)
    parser.add_argument("--operator-control-plane-json", type=Path, default=None)
    parser.add_argument("--weekend-research-json", type=Path, default=None)
    parser.add_argument("--upward-trajectory-json", type=Path, default=None)
    parser.add_argument("--metrics-improvement-json", type=Path, default=None)
    parser.add_argument("--huggingface-discovery-json", type=Path, default=None)
    parser.add_argument("--huggingface-candidate-intake-json", type=Path, default=None)
    parser.add_argument("--huggingface-cache-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args(argv)
    default_output, default_latest, default_md = _default_output(args.report_date)
    output_json = args.output_json or default_output
    latest_json = args.latest_json or default_latest
    output_md = args.output_md or default_md
    report = build_daily_research_report(
        report_date=str(args.report_date),
        health=(
            _read_json(args.health_json)
            if args.health_json is not None
            else _health_from_endpoint(str(args.health_url))
            or _read_json(_default_path("runtime/health_latest.json"))
        ),
        live_cost_model=_read_json(
            args.live_cost_model_json or _default_path("runtime/live_cost_model_latest.json")
        ),
        shadow_report=_read_json(
            args.shadow_report_json
            or _default_path("runtime/ml_shadow_report_latest.json")
        ),
        replay_governance=_read_json(
            args.replay_governance_json
            or _default_path("runtime/replay_governance_refresh_latest.json")
        ),
        symbol_scorecard=_read_json(
            args.symbol_scorecard_json or _default_path("runtime/symbol_universe_scorecard_latest.json")
        ),
        promotion_report=_read_json(
            args.promotion_report_json or _default_path("runtime/promotion_report_latest.json")
        ),
        symbol_promotion_comparison=_read_json(
            args.symbol_promotion_json
            or _default_path("runtime/research_reports/latest/symbol_promotion_latest.json")
        ),
        symbol_lifecycle=_read_json(
            args.symbol_lifecycle_json
            or _default_path("runtime/research_reports/latest/symbol_lifecycle_latest.json")
        ),
        replay_live_cost_alignment=_read_json(
            args.replay_live_cost_alignment_json
            or _default_path("runtime/replay_live_cost_alignment_latest.json")
        ),
        regime_entry_throttle=_read_json(
            args.regime_entry_throttle_json
            or _default_path("runtime/regime_entry_throttle_latest.json")
        ),
        execution_capture=_read_json(
            args.execution_capture_json
            or _default_path("runtime/reports/execution_capture_latest.json")
        ),
        execution_capture_improvement=_read_json(
            args.execution_capture_improvement_json
            or _default_path("runtime/reports/execution_capture_improvement_latest.json")
        ),
        counterfactual_execution=_read_json(
            args.counterfactual_execution_json
            or _default_path("runtime/reports/counterfactual_execution_latest.json")
        ),
        portfolio_edge=_read_json(
            args.portfolio_edge_json
            or _default_path("runtime/reports/portfolio_edge_control_latest.json")
        ),
        decision_receipts=_read_json(
            args.decision_receipts_json
            or _default_path("runtime/reports/decision_receipts_latest.json")
        ),
        training_accelerator=_read_json(
            args.training_accelerator_json
            or _default_path("runtime/training_accelerator_daily_latest.json")
        ),
        opportunity_markouts=_read_json(
            args.opportunity_markouts_json
            or _default_path(
                "runtime/research_reports/latest/opportunity_markouts_latest.json"
            )
        ),
        historical_backfill=_read_json(
            args.historical_backfill_json
            or _default_path(
                "runtime/research_reports/latest/"
                "historical_training_backfill_latest.json"
            )
        ),
        historical_training=_read_json(
            args.historical_training_json
            or _default_path(
                "runtime/research_reports/latest/"
                "historical_replay_aligned_training_latest.json"
            )
        ),
        expected_edge_calibration=_read_json(
            args.expected_edge_calibration_json
            or _default_path("runtime/reports/expected_edge_calibration_latest.json")
        ),
        evidence_starvation=_read_json(
            args.evidence_starvation_json
            or _default_path("runtime/reports/evidence_starvation_latest.json")
        ),
        paper_sampling_state=_read_json(
            args.paper_sampling_state_json
            or _default_path("runtime/paper_sampling_state_latest.json")
        ),
        runtime_gonogo=_read_json(args.runtime_gonogo_json),
        memory_audit=_read_json(
            args.memory_audit_json or _default_path("runtime/memory_hotspot_audit_latest.json")
        ),
        artifact_retention=_read_json(
            args.artifact_retention_json
            or _default_path("runtime/runtime_artifact_retention_latest.json")
        ),
        model_registry=_read_json(
            args.model_registry_json
            or _default_path("runtime/research_reports/latest/model_registry_latest.json")
        ),
        pretrade_risk_verifier=_read_json(
            args.pretrade_risk_json
            or _default_path("runtime/reports/pretrade_risk_control_verification_latest.json")
        ),
        post_trade_surveillance=_read_json(
            args.post_trade_surveillance_json
            or _default_path("runtime/reports/post_trade_surveillance_latest.json")
        ),
        experiment_ledger=_read_json(
            args.experiment_ledger_json
            or _default_path("runtime/research_reports/latest/experiment_ledger_latest.json")
        ),
        walk_forward_capital=_read_json(
            args.walk_forward_capital_json
            or _default_path("runtime/walk_forward_capital_simulation_latest.json")
        ),
        order_type_optimizer=_read_json(
            args.order_type_optimizer_json
            or _default_path("runtime/order_type_optimizer_latest.json")
        ),
        regime_champions=_read_json(
            args.regime_champions_json
            or _default_path("runtime/regime_champion_models_latest.json")
        ),
        adversarial_failure=_read_json(
            args.adversarial_failure_json
            or _default_path("runtime/adversarial_failure_simulation_latest.json")
        ),
        drift_monitor=_read_json(
            args.drift_monitor_json
            or _default_path("runtime/model_data_drift_monitor_latest.json")
        ),
        operator_control_plane=_read_json(
            args.operator_control_plane_json
            or _default_path("runtime/operator_control_plane_latest.json")
        ),
        weekend_research=_read_json(
            args.weekend_research_json
            or _default_path("runtime/research_reports/latest/weekend_research_latest.json")
        ),
        upward_trajectory=_read_json(
            args.upward_trajectory_json
            or _default_path("runtime/reports/upward_trajectory_latest.json")
        ),
        metrics_improvement=_read_json(
            args.metrics_improvement_json
            or _default_path("runtime/reports/metrics_improvement_control_latest.json")
        ),
        huggingface_discovery=_read_json(
            args.huggingface_discovery_json
            or _default_path("runtime/research_reports/latest/hf_discovery_latest.json")
        ),
        huggingface_candidate_intake=_read_json(
            args.huggingface_candidate_intake_json
            or _default_path("runtime/research_reports/latest/hf_candidate_intake_latest.json")
        ),
        huggingface_cache_materialization=_read_json(
            args.huggingface_cache_json
            or _default_path("runtime/research_reports/latest/hf_cache_materialization_latest.json")
        ),
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest_json.parent.mkdir(parents=True, exist_ok=True)
    latest_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(report), encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_json), "trade_allowed": report["trade_allowed"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
