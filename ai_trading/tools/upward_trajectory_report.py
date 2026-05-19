"""Build research-only upward-trajectory diagnostics from existing evidence.

The report deliberately reads generated artifacts and emits recommendations
only. It never places orders, edits runtime configuration, promotes models, or
changes symbol authority.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _nested(payload: Mapping[str, Any] | None, *keys: str) -> dict[str, Any]:
    current: Any = payload or {}
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key, {})
    return dict(current) if isinstance(current, Mapping) else {}


def _status(payload: Mapping[str, Any] | None, default: str = "missing") -> str:
    if not payload:
        return default
    raw = payload.get("status")
    if isinstance(raw, Mapping):
        return str(raw.get("status") or raw.get("state") or default)
    return str(raw or default)


def _read_json_from_payload_path(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    raw = str(payload.get(key) or "").strip()
    if not raw:
        return {}
    return _read_json(Path(raw).expanduser())


def _candidate_rows(
    *,
    training_accelerator: Mapping[str, Any],
    multi_horizon_report: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    multi = dict(multi_horizon_report or {})
    if not multi:
        multi = _read_json_from_payload_path(training_accelerator, "multi_horizon_report")
    ranked = multi.get("ranked_candidates")
    if isinstance(ranked, list):
        rows = [dict(row) for row in ranked if isinstance(row, Mapping)]
    else:
        rows = []
    return rows, multi


def _best_validation_edge(candidate: Mapping[str, Any]) -> float | None:
    values: list[float] = []
    for row in candidate.get("threshold_sweep", []) if isinstance(candidate.get("threshold_sweep"), list) else []:
        if not isinstance(row, Mapping):
            continue
        for key in ("mean_net_markout_bps", "net_edge_bps", "expectancy_bps"):
            value = _safe_float(row.get(key))
            if value is not None:
                values.append(value)
    return max(values) if values else None


def _validation_to_replay_gap(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    classifications: Counter[str] = Counter()
    for candidate in candidates:
        replay = _nested(candidate, "replay")
        validation = _nested(candidate, "validation")
        validation_edge = _best_validation_edge(candidate)
        replay_expectancy = _safe_float(replay.get("expectancy_bps"))
        auc = _safe_float(validation.get("roc_auc"))
        gap = (
            float(validation_edge - replay_expectancy)
            if validation_edge is not None and replay_expectancy is not None
            else None
        )
        if validation_edge is None or replay_expectancy is None:
            classification = "insufficient_comparable_evidence"
        elif validation_edge > 0.0 and replay_expectancy < 0.0:
            classification = "validation_positive_replay_negative"
        elif replay_expectancy > 0.0:
            classification = "replay_confirmed_edge"
        elif auc is not None and auc < 0.55:
            classification = "weak_signal_quality"
        else:
            classification = "execution_or_cost_gap"
        classifications[classification] += 1
        rows.append(
            {
                "model_name": candidate.get("model_name"),
                "horizon_bars": candidate.get("horizon_bars"),
                "label_objective": candidate.get("label_objective"),
                "validation_roc_auc": auc,
                "best_validation_edge_bps": validation_edge,
                "replay_expectancy_bps": replay_expectancy,
                "validation_replay_gap_bps": gap,
                "replay_profit_factor": _safe_float(replay.get("profit_factor")),
                "replay_win_rate": _safe_float(replay.get("win_rate")),
                "classification": classification,
            }
        )
    return {
        "status": "ready" if rows else "insufficient_candidates",
        "classification_counts": dict(sorted(classifications.items())),
        "largest_gaps": sorted(
            rows,
            key=lambda row: abs(_safe_float(row.get("validation_replay_gap_bps")) or 0.0),
            reverse=True,
        )[:10],
        "diagnosis": (
            "replay_gap_detected"
            if classifications.get("validation_positive_replay_negative", 0) > 0
            else "no_confirmed_validation_replay_gap"
        ),
    }


def _candidate_tournament(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tournament_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        replay = _nested(candidate, "replay")
        validation = _nested(candidate, "validation")
        expectancy = _safe_float(replay.get("expectancy_bps"))
        auc = _safe_float(validation.get("roc_auc"))
        profit_factor = _safe_float(replay.get("profit_factor"))
        total_trades = _safe_int(replay.get("total_trades"))
        score = (
            (expectancy or -100.0)
            + ((profit_factor or 0.0) * 5.0)
            + (((auc or 0.5) - 0.5) * 20.0)
            + min(total_trades, 200) / 200.0
        )
        if expectancy is None:
            decision = "needs_replay"
        elif expectancy <= 0.0:
            decision = "reject_for_now"
        elif total_trades < 25:
            decision = "collect_more_shadow_evidence"
        else:
            decision = "shadow_challenger"
        tournament_rows.append(
            {
                "model_name": candidate.get("model_name"),
                "horizon_bars": candidate.get("horizon_bars"),
                "label_objective": candidate.get("label_objective"),
                "score": round(float(score), 6),
                "decision": decision,
                "replay_expectancy_bps": expectancy,
                "replay_profit_factor": profit_factor,
                "validation_roc_auc": auc,
                "total_trades": total_trades,
                "promotion_authority": False,
                "runtime_authority": False,
            }
        )
    ranked = sorted(tournament_rows, key=lambda row: float(row["score"]), reverse=True)
    return {
        "status": "ready" if ranked else "insufficient_candidates",
        "champion_candidate": ranked[0] if ranked else None,
        "ranked_candidates": ranked,
        "recommendation": (
            "evaluate_top_candidate_in_shadow_only" if ranked else "train_more_candidates"
        ),
        "manual_promotion_required": True,
    }


def _symbols_from_rows(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            output.append(symbol)
    return output


def _symbol_expansion_pruning(
    *,
    symbol_lifecycle: Mapping[str, Any],
    symbol_scorecard: Mapping[str, Any],
) -> dict[str, Any]:
    lifecycle_rows = [
        dict(row) for row in symbol_lifecycle.get("symbols", []) if isinstance(row, Mapping)
    ]
    scorecard_rows = [
        dict(row) for row in symbol_scorecard.get("symbols", []) if isinstance(row, Mapping)
    ]
    actions: list[dict[str, Any]] = []
    for row in lifecycle_rows:
        recommendation = str(row.get("recommendation") or "collect_more_evidence")
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        if recommendation in {"disable", "move_to_shadow_only", "restrict"}:
            action = "prune_or_restrict"
        elif recommendation in {"consider_canary", "consider_allow"}:
            action = "manual_review_only"
        else:
            action = "collect_more_evidence"
        actions.append(
            {
                "symbol": symbol,
                "action": action,
                "current_mode": row.get("current_mode"),
                "recommended_mode": row.get("recommended_mode"),
                "authority_increase": bool(row.get("authority_increase", False)),
                "manual_approval_required": True,
                "reason": recommendation,
            }
        )
    lifecycle_symbols = set(_symbols_from_rows(lifecycle_rows))
    for row in scorecard_rows:
        symbol = str(row.get("symbol") or "").upper()
        if symbol and symbol not in lifecycle_symbols:
            actions.append(
                {
                    "symbol": symbol,
                    "action": "shadow_only_candidate",
                    "current_mode": "unknown",
                    "recommended_mode": "shadow_only",
                    "authority_increase": False,
                    "manual_approval_required": False,
                    "reason": "scorecard_symbol_not_in_lifecycle",
                }
            )
    counts = Counter(str(row["action"]) for row in actions)
    return {
        "status": "ready" if actions else "insufficient_symbol_evidence",
        "summary": dict(sorted(counts.items())),
        "symbols": actions,
        "runtime_symbol_gating_changed": False,
    }


def _evidence_acceleration(
    *,
    expected_edge_calibration: Mapping[str, Any],
    execution_capture: Mapping[str, Any],
    symbol_lifecycle: Mapping[str, Any],
    paper_sampling_state: Mapping[str, Any],
    min_bucket_samples: int,
) -> dict[str, Any]:
    calibration_status = _status(expected_edge_calibration)
    execution_status = _status(execution_capture)
    priority_buckets: list[dict[str, Any]] = []
    for symbol in _symbols_from_rows(
        row for row in symbol_lifecycle.get("symbols", []) if isinstance(row, Mapping)
    ):
        priority_buckets.append(
            {
                "symbol": symbol,
                "side": "buy",
                "session_bucket": "opening_or_midday",
                "spread_bucket": "tight_or_normal",
                "quote_age_bucket": "fresh",
                "regime": "normal_volatility",
                "reason": "calibration_or_execution_samples_needed",
                "target_min_samples": int(min_bucket_samples),
            }
        )
    if not priority_buckets:
        priority_buckets.append(
            {
                "symbol": "UNIVERSE",
                "side": "buy",
                "session_bucket": "opening_or_midday",
                "spread_bucket": "tight_or_normal",
                "quote_age_bucket": "fresh",
                "regime": "normal_volatility",
                "reason": "no_symbol_lifecycle_rows_available",
                "target_min_samples": int(min_bucket_samples),
            }
        )
    status = (
        "accelerate_collection"
        if "insufficient" in calibration_status or "insufficient" in execution_status
        else "monitor"
    )
    return {
        "status": status,
        "calibration_status": calibration_status,
        "execution_capture_status": execution_status,
        "priority_buckets": priority_buckets[:25],
        "diagnostic_sampling_limits": {
            "paper_only": True,
            "suggested_max_trades_per_day": min(4, max(1, len(priority_buckets))),
            "suggested_max_trades_per_symbol_per_day": 1,
            "suggested_max_notional_per_order": paper_sampling_state.get(
                "max_notional_per_order",
                250,
            ),
            "manual_operator_review_required": True,
        },
        "runtime_authority": False,
        "live_money_authority": False,
    }


def _active_learning(
    evidence_acceleration: Mapping[str, Any],
    execution_capture: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _nested(execution_capture, "summary")
    capture = _safe_float(summary.get("execution_capture_ratio") or summary.get("capture_ratio"))
    proposals: list[dict[str, Any]] = []
    for index, bucket in enumerate(evidence_acceleration.get("priority_buckets", [])):
        if not isinstance(bucket, Mapping):
            continue
        uncertainty = 1.0 if capture is None else min(2.0, abs(1.0 - capture))
        score = round(float(uncertainty / (1 + index)), 6)
        proposals.append(
            {
                "rank": index + 1,
                "information_gain_score": score,
                "paper_only": True,
                "action": "diagnostic_sample_if_all_hard_gates_pass",
                "bucket": dict(bucket),
            }
        )
    return {
        "status": "ready" if proposals else "insufficient_buckets",
        "proposal_count": len(proposals),
        "proposals": proposals[:10],
        "live_money_authority": False,
    }


def _execution_aware_labels(
    *,
    training_accelerator: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    live_cost_model: Mapping[str, Any],
    execution_capture: Mapping[str, Any],
) -> dict[str, Any]:
    config = _nested(training_accelerator, "config")
    objectives = str(config.get("label_objectives") or "").split(",")
    objectives = [item.strip() for item in objectives if item.strip()]
    cost_status = _status(_nested(live_cost_model, "status") or live_cost_model)
    capture_status = _status(execution_capture)
    live_cost_requested = bool(config.get("use_live_cost_model"))
    live_cost_candidate_count = 0
    live_cost_unusable_reasons: Counter[str] = Counter()
    for candidate in candidates:
        metadata = candidate.get("live_cost_model")
        if not isinstance(metadata, Mapping):
            continue
        if bool(metadata.get("usable")) or bool(metadata.get("enabled")):
            live_cost_candidate_count += 1
        elif bool(metadata.get("requested")):
            live_cost_unusable_reasons[str(metadata.get("reason") or "not_loaded")] += 1
    required = {
        "spread_adjusted_return": "spread_adjusted" in objectives,
        "live_cost_adjusted_net_edge": live_cost_candidate_count > 0,
        "risk_adjusted_label": "risk_adjusted" in objectives,
        "mae_mfe_label": "mae_mfe" in objectives,
        "execution_capture_feedback": capture_status not in {"missing"},
    }
    missing = [name for name, present in required.items() if not present]
    return {
        "status": "ready" if not missing else "partial",
        "objectives": objectives,
        "cost_model_status": cost_status,
        "live_cost_model_requested": live_cost_requested,
        "live_cost_model_applied_candidate_count": live_cost_candidate_count,
        "live_cost_model_unusable_reasons": dict(sorted(live_cost_unusable_reasons.items())),
        "execution_capture_status": capture_status,
        "coverage": required,
        "missing_label_capabilities": missing,
        "candidate_evaluation_priority": "cost_adjusted_expectancy_over_raw_accuracy",
        "promotion_authority": False,
    }


def _regime_escalation(regime_champions: Mapping[str, Any]) -> dict[str, Any]:
    blocked = regime_champions.get("blocked_regimes", [])
    summary = _nested(regime_champions, "summary")
    status = _status(regime_champions)
    return {
        "status": "ready" if regime_champions else "missing",
        "regime_champion_status": status,
        "summary": summary,
        "blocked_regimes": blocked if isinstance(blocked, list) else [],
        "escalation_policy": {
            "manual_promotion_required": True,
            "missing_or_stale_evidence_action": "fallback_to_global_or_shadow_only",
            "authority_increase_allowed": False,
        },
    }


def _feature_autopsy(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        dataset = _nested(candidate, "dataset")
        validation = _nested(candidate, "validation")
        replay = _nested(candidate, "replay")
        validation_edge = _best_validation_edge(candidate)
        replay_expectancy = _safe_float(replay.get("expectancy_bps"))
        rows.append(
            {
                "model_name": candidate.get("model_name"),
                "feature_importance_available": bool(candidate.get("feature_importance")),
                "dataset_rows": dataset.get("rows"),
                "timestamp_authoritative": dataset.get("timestamp_authoritative"),
                "validation_roc_auc": _safe_float(validation.get("roc_auc")),
                "best_validation_edge_bps": validation_edge,
                "replay_expectancy_bps": replay_expectancy,
                "autopsy": (
                    "validation_edge_not_execution_robust"
                    if validation_edge is not None
                    and replay_expectancy is not None
                    and validation_edge > 0
                    and replay_expectancy < 0
                    else "monitor"
                ),
            }
        )
    counts = Counter(str(row["autopsy"]) for row in rows)
    return {
        "status": "ready" if rows else "insufficient_candidates",
        "summary": dict(sorted(counts.items())),
        "signals": rows[:25],
        "next_action": "add_feature_importance_to_training_manifests" if rows else "train_candidates",
    }


def build_upward_trajectory_report(
    *,
    expected_edge_calibration: Mapping[str, Any] | None = None,
    execution_capture: Mapping[str, Any] | None = None,
    training_accelerator: Mapping[str, Any] | None = None,
    multi_horizon_report: Mapping[str, Any] | None = None,
    symbol_lifecycle: Mapping[str, Any] | None = None,
    symbol_scorecard: Mapping[str, Any] | None = None,
    paper_sampling_state: Mapping[str, Any] | None = None,
    regime_champions: Mapping[str, Any] | None = None,
    live_cost_model: Mapping[str, Any] | None = None,
    report_date: str | None = None,
    min_bucket_samples: int = 25,
) -> dict[str, Any]:
    expected_edge_calibration = expected_edge_calibration or {}
    execution_capture = execution_capture or {}
    training_accelerator = training_accelerator or {}
    symbol_lifecycle = symbol_lifecycle or {}
    symbol_scorecard = symbol_scorecard or {}
    paper_sampling_state = paper_sampling_state or {}
    regime_champions = regime_champions or {}
    live_cost_model = live_cost_model or {}
    candidates, multi = _candidate_rows(
        training_accelerator=training_accelerator,
        multi_horizon_report=multi_horizon_report,
    )
    evidence = _evidence_acceleration(
        expected_edge_calibration=expected_edge_calibration,
        execution_capture=execution_capture,
        symbol_lifecycle=symbol_lifecycle,
        paper_sampling_state=paper_sampling_state,
        min_bucket_samples=min_bucket_samples,
    )
    validation_gap = _validation_to_replay_gap(candidates)
    tournament = _candidate_tournament(candidates)
    active_learning = _active_learning(evidence, execution_capture)
    sections = {
        "evidence_acceleration_engine": evidence,
        "validation_to_replay_gap_analyzer": validation_gap,
        "candidate_tournament_system": tournament,
        "symbol_expansion_pruning_loop": _symbol_expansion_pruning(
            symbol_lifecycle=symbol_lifecycle,
            symbol_scorecard=symbol_scorecard,
        ),
        "execution_aware_model_labels": _execution_aware_labels(
            training_accelerator=training_accelerator,
            candidates=candidates,
            live_cost_model=live_cost_model,
            execution_capture=execution_capture,
        ),
        "active_learning_paper_trades": active_learning,
        "regime_champion_escalation": _regime_escalation(regime_champions),
        "feature_attribution_signal_autopsy": _feature_autopsy(candidates),
    }
    attention = [
        name
        for name, section in sections.items()
        if str(section.get("status") or "").lower()
        in {"partial", "insufficient_candidates", "insufficient_symbol_evidence", "accelerate_collection"}
    ]
    recommendation = "collect_paper_evidence_and_shadow_candidates"
    if validation_gap.get("diagnosis") == "replay_gap_detected":
        recommendation = "debug_validation_replay_gap_before_promotion"
    return {
        "schema_version": "1.0.0",
        "artifact_type": "upward_trajectory_report",
        "report_date": report_date or datetime.now(UTC).strftime("%Y-%m-%d"),
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "ready",
        "authority": {
            "research_only": True,
            "paper_only_diagnostics": True,
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
            "manual_approval_required_for_authority_increase": True,
        },
        "source_artifacts": {
            "training_accelerator_status": _status(training_accelerator),
            "multi_horizon_status": _status(multi),
            "expected_edge_calibration_status": _status(expected_edge_calibration),
            "execution_capture_status": _status(execution_capture),
            "symbol_lifecycle_status": _status(symbol_lifecycle),
            "regime_champion_status": _status(regime_champions),
        },
        "summary": {
            "feature_count": len(sections),
            "attention_features": attention,
            "candidate_count": len(candidates),
            "recommended_next_action": recommendation,
        },
        **sections,
    }


def _default_path(relative: str) -> Path:
    return resolve_runtime_artifact_path(relative, default_relative=relative)


def _default_outputs(report_date: str) -> tuple[Path, Path, Path]:
    compact = report_date.replace("-", "")
    dated = resolve_runtime_artifact_path(
        f"runtime/reports/upward_trajectory_{compact}.json",
        default_relative=f"runtime/reports/upward_trajectory_{compact}.json",
        for_write=True,
    )
    latest = resolve_runtime_artifact_path(
        "runtime/reports/upward_trajectory_latest.json",
        default_relative="runtime/reports/upward_trajectory_latest.json",
        for_write=True,
    )
    research_latest = resolve_runtime_artifact_path(
        "runtime/research_reports/latest/upward_trajectory_latest.json",
        default_relative="runtime/research_reports/latest/upward_trajectory_latest.json",
        for_write=True,
    )
    return dated, latest, research_latest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--expected-edge-calibration-json", type=Path, default=None)
    parser.add_argument("--execution-capture-json", type=Path, default=None)
    parser.add_argument("--training-accelerator-json", type=Path, default=None)
    parser.add_argument("--multi-horizon-json", type=Path, default=None)
    parser.add_argument("--symbol-lifecycle-json", type=Path, default=None)
    parser.add_argument("--symbol-scorecard-json", type=Path, default=None)
    parser.add_argument("--paper-sampling-json", type=Path, default=None)
    parser.add_argument("--regime-champions-json", type=Path, default=None)
    parser.add_argument("--live-cost-model-json", type=Path, default=None)
    parser.add_argument("--min-bucket-samples", type=int, default=25)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    parser.add_argument("--research-latest-json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = build_upward_trajectory_report(
        report_date=str(args.report_date),
        expected_edge_calibration=_read_json(
            args.expected_edge_calibration_json
            or _default_path("runtime/reports/expected_edge_calibration_latest.json")
        ),
        execution_capture=_read_json(
            args.execution_capture_json
            or _default_path("runtime/reports/execution_capture_latest.json")
        ),
        training_accelerator=_read_json(
            args.training_accelerator_json
            or _default_path("runtime/training_accelerator_daily_latest.json")
        ),
        multi_horizon_report=_read_json(args.multi_horizon_json),
        symbol_lifecycle=_read_json(
            args.symbol_lifecycle_json
            or _default_path("runtime/research_reports/latest/symbol_lifecycle_latest.json")
        ),
        symbol_scorecard=_read_json(
            args.symbol_scorecard_json
            or _default_path("runtime/symbol_universe_scorecard_latest.json")
        ),
        paper_sampling_state=_read_json(
            args.paper_sampling_json
            or _default_path("runtime/paper_sampling_state_latest.json")
        ),
        regime_champions=_read_json(
            args.regime_champions_json
            or _default_path("runtime/regime_champion_models_latest.json")
        ),
        live_cost_model=_read_json(
            args.live_cost_model_json
            or _default_path("runtime/live_cost_model_latest.json")
        ),
        min_bucket_samples=max(1, int(args.min_bucket_samples)),
    )
    dated, latest, research_latest = _default_outputs(str(args.report_date))
    for path in (
        args.output_json or dated,
        args.latest_json or latest,
        args.research_latest_json or research_latest,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"status": report["status"], "path": str(args.output_json or dated)}) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
