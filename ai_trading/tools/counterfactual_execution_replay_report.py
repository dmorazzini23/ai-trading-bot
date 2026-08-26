"""Replay accepted and rejected decisions into counterfactual execution evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _read_jsonl(path: Path | None, *, report_date: str | None = None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            if report_date and not _date_match(parsed, report_date):
                continue
            rows.append(parsed)
    return rows


def _read_outcomes(path: Path | None, *, report_date: str | None = None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _read_jsonl(path, report_date=report_date)
    if isinstance(payload, Mapping):
        raw_rows = payload.get("outcomes", payload.get("resolved_outcomes", []))
    else:
        raw_rows = payload
    if not isinstance(raw_rows, list):
        return []
    rows = [dict(row) for row in raw_rows if isinstance(row, Mapping)]
    if report_date:
        rows = [row for row in rows if _date_match(row, report_date)]
    return rows


def _date_match(row: Mapping[str, Any], report_date: str) -> bool:
    journal = row.get("decision_journal")
    journal = journal if isinstance(journal, Mapping) else {}
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    ts = str(
        row.get("ts")
        or row.get("prediction_ts")
        or row.get("timestamp")
        or row.get("decision_ts")
        or row.get("source_timestamp")
        or row.get("decision_timestamp")
        or row.get("bar_ts")
        or journal.get("decision_ts")
        or journal.get("source_timestamp")
        or metrics.get("decision_ts")
        or metrics.get("source_timestamp")
        or row.get("filled_at")
        or ""
    )
    return ts.startswith(report_date)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _first_float(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        parsed = _safe_float(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _decision_id(row: Mapping[str, Any]) -> str:
    journal = row.get("decision_journal")
    journal = journal if isinstance(journal, Mapping) else {}
    return str(
        row.get("decision_id")
        or row.get("intent_id")
        or row.get("client_order_id")
        or journal.get("decision_id")
        or journal.get("intent_id")
        or ""
    ).strip()


def _prediction_id(row: Mapping[str, Any]) -> str:
    return str(row.get("prediction_id") or _decision_id(row)).strip()


def _correlation_id(row: Mapping[str, Any]) -> str:
    journal = row.get("decision_journal")
    journal = journal if isinstance(journal, Mapping) else {}
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    return str(
        row.get("correlation_id")
        or row.get("decision_correlation_id")
        or row.get("opportunity_id")
        or journal.get("correlation_id")
        or journal.get("decision_correlation_id")
        or metrics.get("correlation_id")
        or _prediction_id(row)
    ).strip()


def _identity_candidates(row: Mapping[str, Any]) -> set[str]:
    journal = row.get("decision_journal")
    journal = journal if isinstance(journal, Mapping) else {}
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    values = {
        _decision_id(row),
        _prediction_id(row),
        _correlation_id(row),
        str(row.get("decision_trace_id") or "").strip(),
        str(row.get("root_client_order_id") or "").strip(),
        str(journal.get("decision_trace_id") or "").strip(),
        str(metrics.get("decision_trace_id") or "").strip(),
    }
    return {value for value in values if value}


def _decision_timestamp(row: Mapping[str, Any]) -> str:
    journal = row.get("decision_journal")
    journal = journal if isinstance(journal, Mapping) else {}
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    raw = (
        row.get("decision_timestamp")
        or row.get("decision_ts")
        or row.get("source_timestamp")
        or row.get("bar_ts")
        or journal.get("decision_ts")
        or journal.get("source_timestamp")
        or metrics.get("decision_ts")
        or metrics.get("source_timestamp")
    )
    if raw in (None, ""):
        return ""
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(
            UTC
        ).isoformat()
    except ValueError:
        return str(raw).strip()


def _rejection_reason(row: Mapping[str, Any]) -> str:
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    journal = row.get("decision_journal")
    journal = journal if isinstance(journal, Mapping) else {}
    for value in (
        row.get("reason"),
        row.get("gate"),
        metrics.get("terminal_reason"),
        metrics.get("reason"),
        journal.get("terminal_reason"),
        journal.get("reason"),
    ):
        if str(value or "").strip():
            return str(value).strip()
    gates = row.get("gates")
    if isinstance(gates, Sequence) and not isinstance(gates, (str, bytes)):
        reasons: list[str] = []
        for gate in gates:
            if isinstance(gate, Mapping):
                value = gate.get("reason") or gate.get("name") or gate.get("gate")
            else:
                value = gate
            if str(value or "").strip():
                reasons.append(str(value).strip())
        if reasons:
            return "+".join(sorted(set(reasons)))
    return "unknown"


def _symbol(row: Mapping[str, Any]) -> str:
    return str(row.get("symbol") or row.get("ticker") or "UNKNOWN").strip().upper() or "UNKNOWN"


def _status(row: Mapping[str, Any]) -> str:
    explicit = str(
        row.get("status") or row.get("action") or row.get("decision") or ""
    ).strip().lower()
    if explicit:
        return explicit
    journal = row.get("decision_journal")
    journal = journal if isinstance(journal, Mapping) else {}
    if journal.get("submitted") is True:
        return "submitted"
    if journal.get("accepted") is True:
        return "accepted"
    if journal.get("accepted") is False:
        return "rejected"
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    if metrics.get("terminal_reason") not in (None, ""):
        return "rejected"
    gates = row.get("gates")
    if isinstance(gates, Sequence) and not isinstance(gates, (str, bytes)) and gates:
        return "rejected"
    return ""


def _is_accepted(row: Mapping[str, Any]) -> bool:
    return _status(row) in {"accept", "accepted", "submit", "submitted", "filled", "new"}


def _is_rejected(row: Mapping[str, Any]) -> bool:
    return _status(row) in {"reject", "rejected", "blocked", "skip", "skipped"}


def _select_resolved_outcome(
    decision: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if not candidates:
        return None
    desired_role = str(decision.get("model_role") or "").strip().lower()
    desired_horizon = _safe_float(decision.get("horizon_bars"))
    eligible: list[Mapping[str, Any]] = []
    for row in candidates:
        evidence_type = str(
            row.get("evidence_type") or "hypothetical"
        ).strip().lower()
        if bool(row.get("executed")):
            continue
        if evidence_type == "hypothetical":
            eligible.append(row)
            continue
        if evidence_type != "shadow_counterfactual":
            continue
        if str(row.get("label_status") or "").strip().lower() != "resolved":
            continue
        if row.get("research_only") is not True:
            continue
        if str(row.get("evidence_partition") or "").strip().lower() != "shadow":
            continue
        if any(
            row.get(field) is not False
            for field in (
                "fill_based_evidence",
                "promotion_eligible",
                "runtime_authority",
                "promotion_authority",
                "live_money_authority",
            )
        ):
            continue
        eligible.append(row)
    if desired_role:
        role_matches = [
            row
            for row in eligible
            if str(row.get("model_role") or "").strip().lower() == desired_role
        ]
        if role_matches:
            eligible = role_matches
    elif any(
        str(row.get("model_role") or "").strip().lower() == "challenger"
        for row in eligible
    ):
        eligible = [
            row
            for row in eligible
            if str(row.get("model_role") or "").strip().lower() == "challenger"
        ]
    if desired_horizon is not None:
        horizon_matches = [
            row
            for row in eligible
            if _safe_float(row.get("horizon_bars")) == desired_horizon
        ]
        if horizon_matches:
            eligible = horizon_matches
    primary = [row for row in eligible if bool(row.get("primary_horizon"))]
    if primary:
        eligible = primary
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: _safe_float(row.get("horizon_bars")) or float("inf"),
    )


def build_counterfactual_execution_replay_report(
    *,
    report_date: str,
    decisions: Sequence[Mapping[str, Any]],
    fills: Sequence[Mapping[str, Any]] = (),
    outcomes: Sequence[Mapping[str, Any]] = (),
    min_counterfactual_samples: int = 10,
    max_missed_edge_bps: float = 25.0,
    min_filter_ablation_samples: int = 5,
) -> dict[str, Any]:
    fills_by_id: dict[str, Mapping[str, Any]] = {}
    for fill in fills:
        for identity in {_decision_id(fill), _correlation_id(fill)}:
            if identity:
                fills_by_id[identity] = fill
    outcomes_by_id: dict[str, list[Mapping[str, Any]]] = {}
    outcomes_by_symbol_timestamp: dict[
        tuple[str, str], list[Mapping[str, Any]]
    ] = {}
    for outcome in outcomes:
        for identity in _identity_candidates(outcome):
            outcomes_by_id.setdefault(identity, []).append(outcome)
        timestamp = _decision_timestamp(outcome)
        if timestamp:
            outcomes_by_symbol_timestamp.setdefault(
                (_symbol(outcome), timestamp), []
            ).append(outcome)
    accepted_edges: list[float] = []
    rejected_edges: list[float] = []
    missed_positive: list[dict[str, Any]] = []
    avoided_negative = 0
    rejection_reasons: Counter[str] = Counter()
    accepted = 0
    rejected = 0
    hypothetical_outcome_samples = 0
    executed_outcome_rows_ignored = 0
    rejected_decisions_without_linked_outcomes = 0
    outcome_join_methods: Counter[str] = Counter()
    edges_by_rejection_reason: dict[str, list[float]] = {}
    for row in decisions:
        if _is_accepted(row):
            accepted += 1
            realized = _first_float(
                row,
                "realized_net_edge_bps",
                "net_edge_bps",
                "markout_bps",
            )
            fill_identity = _correlation_id(row) or _decision_id(row)
            if realized is None and fill_identity in fills_by_id:
                realized = _first_float(
                    fills_by_id[fill_identity],
                    "realized_net_edge_bps",
                    "net_edge_bps",
                    "markout_bps",
                )
            if realized is not None:
                accepted_edges.append(float(realized))
        elif _is_rejected(row):
            rejected += 1
            reason = _rejection_reason(row)
            rejection_reasons[reason] += 1
            counterfactual = _first_float(
                row,
                "counterfactual_net_edge_bps",
                "hypothetical_net_edge_bps",
            )
            selected_outcome: Mapping[str, Any] | None = None
            if counterfactual is None:
                candidates: list[Mapping[str, Any]] = []
                seen_candidates: set[str] = set()
                for identity in _identity_candidates(row):
                    for candidate in outcomes_by_id.get(identity, []):
                        candidate_key = str(
                            candidate.get("outcome_id")
                            or candidate.get("replay_row_id")
                            or id(candidate)
                        )
                        if candidate_key not in seen_candidates:
                            seen_candidates.add(candidate_key)
                            candidates.append(candidate)
                join_method = "identity" if candidates else "unmatched"
                if not candidates and (timestamp := _decision_timestamp(row)):
                    candidates = list(
                        outcomes_by_symbol_timestamp.get((_symbol(row), timestamp), [])
                    )
                    if candidates:
                        join_method = "symbol_decision_timestamp"
                outcome_join_methods[join_method] += 1
                executed_outcome_rows_ignored += sum(
                    1
                    for candidate in candidates
                    if bool(candidate.get("executed"))
                    or str(candidate.get("evidence_type") or "").strip().lower()
                    == "executed"
                )
                selected_outcome = _select_resolved_outcome(row, candidates)
                if selected_outcome is not None:
                    counterfactual = _first_float(
                        selected_outcome,
                        "counterfactual_net_edge_bps",
                        "net_markout_bps",
                    )
                    if counterfactual is not None:
                        hypothetical_outcome_samples += 1
                if not candidates:
                    rejected_decisions_without_linked_outcomes += 1
            if counterfactual is None:
                continue
            rejected_edges.append(float(counterfactual))
            edges_by_rejection_reason.setdefault(reason, []).append(
                float(counterfactual)
            )
            if counterfactual > 0.0:
                evidence = selected_outcome or row
                missed_positive.append(
                    {
                        "decision_id": _decision_id(row) or None,
                        "prediction_id": _prediction_id(row) or None,
                        "correlation_id": (
                            _correlation_id(row)
                            or _correlation_id(evidence)
                            or None
                        ),
                        "symbol": _symbol(row),
                        "reason": reason,
                        "counterfactual_net_edge_bps": float(counterfactual),
                        "evidence_type": (
                            str(evidence.get("evidence_type") or "hypothetical")
                        ),
                        "model_role": evidence.get("model_role"),
                        "horizon_bars": evidence.get("horizon_bars"),
                        "model_id": evidence.get("model_id"),
                        "model_version": evidence.get("model_version"),
                        "model_artifact_hash": evidence.get("model_artifact_hash"),
                        "feature_version": evidence.get("feature_version"),
                        "required_bar_timeframe": evidence.get(
                            "required_bar_timeframe"
                        ),
                    }
                )
            else:
                avoided_negative += 1
    missed_edge = sum(item["counterfactual_net_edge_bps"] for item in missed_positive)
    sufficient = len(rejected_edges) >= int(min_counterfactual_samples)
    passed = bool(sufficient and missed_edge <= float(max_missed_edge_bps))
    if not sufficient:
        status = "insufficient_counterfactual_samples"
        action = "collect_more_rejected_decision_markouts"
    elif passed:
        status = "passed"
        action = "keep_current_execution_filters"
    else:
        status = "needs_review"
        action = "review_filters_blocking_positive_counterfactual_edge"
    filter_ablations: list[dict[str, Any]] = []
    for reason, values in sorted(edges_by_rejection_reason.items()):
        positive_values = [value for value in values if value > 0.0]
        negative_values = [value for value in values if value <= 0.0]
        positive_edge = float(sum(positive_values))
        avoided_negative_edge = float(abs(sum(negative_values)))
        mean_edge = float(sum(values) / len(values))
        enough = len(values) >= max(1, int(min_filter_ablation_samples))
        if not enough:
            disposition = "collect_more_shadow_evidence"
        elif (
            mean_edge > 0.0
            and positive_edge > avoided_negative_edge
            and len(positive_values) / len(values) >= 0.60
        ):
            disposition = "eligible_for_bounded_shadow_ablation"
        elif mean_edge <= 0.0:
            disposition = "keep_filter"
        else:
            disposition = "observe_filter"
        filter_ablations.append(
            {
                "rejection_reason": reason,
                "samples": len(values),
                "min_samples": max(1, int(min_filter_ablation_samples)),
                "sufficient_evidence": enough,
                "positive_count": len(positive_values),
                "avoided_negative_count": len(negative_values),
                "positive_rate": float(len(positive_values) / len(values)),
                "mean_counterfactual_net_edge_bps": mean_edge,
                "missed_positive_edge_bps": positive_edge,
                "avoided_negative_edge_bps": avoided_negative_edge,
                "disposition": disposition,
                "runtime_filter_change_authorized": False,
            }
        )
    return {
        "schema_version": "1.1.0",
        "artifact_type": "counterfactual_execution_replay_report",
        "report_date": report_date,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "recommended_next_action": action,
        "counterfactual": {
            "passed": passed,
            "min_samples": int(min_counterfactual_samples),
            "samples": len(rejected_edges),
            "max_missed_edge_bps": float(max_missed_edge_bps),
        },
        "summary": {
            "accepted_decisions": accepted,
            "rejected_decisions": rejected,
            "accepted_realized_samples": len(accepted_edges),
            "rejected_counterfactual_samples": len(rejected_edges),
            "hypothetical_outcome_samples": hypothetical_outcome_samples,
            "executed_outcome_rows_ignored": executed_outcome_rows_ignored,
            "rejected_decisions_without_linked_outcomes": (
                rejected_decisions_without_linked_outcomes
            ),
            "outcome_join_methods": dict(sorted(outcome_join_methods.items())),
            "mean_accepted_realized_edge_bps": (
                float(sum(accepted_edges) / len(accepted_edges)) if accepted_edges else None
            ),
            "mean_rejected_counterfactual_edge_bps": (
                float(sum(rejected_edges) / len(rejected_edges)) if rejected_edges else None
            ),
            "missed_positive_count": len(missed_positive),
            "avoided_negative_count": avoided_negative,
            "missed_positive_edge_bps": float(missed_edge),
        },
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "filter_ablations": {
            "status": (
                "shadow_candidates_available"
                if any(
                    row["disposition"]
                    == "eligible_for_bounded_shadow_ablation"
                    for row in filter_ablations
                )
                else "no_shadow_relaxation_candidate"
            ),
            "rows": filter_ablations,
            "research_only": True,
            "runtime_filter_change_authorized": False,
        },
        "missed_positive_decisions": sorted(
            missed_positive,
            key=lambda item: float(item["counterfactual_net_edge_bps"]),
            reverse=True,
        )[:25],
        "promotion_authority": False,
        "research_evidence": {
            "evidence_partition": "shadow",
            "fill_based_evidence": False,
            "promotion_eligible": False,
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
        },
        "live_money_authority": False,
    }


def _default_report_paths(report_date: str) -> tuple[Path, Path]:
    root = resolve_runtime_artifact_path(
        "runtime/reports",
        default_relative="runtime/reports",
        for_write=True,
    )
    compact = report_date.replace("-", "")
    return root / f"counterfactual_execution_replay_{compact}.json", root / "counterfactual_execution_replay_latest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--decisions-jsonl", type=Path, default=None)
    parser.add_argument("--fills-jsonl", type=Path, default=None)
    parser.add_argument(
        "--outcomes-json",
        type=Path,
        default=None,
        help="ML shadow report JSON, outcome list JSON, or resolved-outcome JSONL.",
    )
    parser.add_argument("--min-counterfactual-samples", type=int, default=10)
    parser.add_argument("--max-missed-edge-bps", type=float, default=25.0)
    parser.add_argument("--min-filter-ablation-samples", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)
    output_json, latest_json = _default_report_paths(str(args.report_date))
    output_json = args.output_json or output_json
    latest_json = args.latest_json or latest_json
    report = build_counterfactual_execution_replay_report(
        report_date=str(args.report_date),
        decisions=_read_jsonl(args.decisions_jsonl, report_date=str(args.report_date)),
        fills=_read_jsonl(args.fills_jsonl, report_date=str(args.report_date)),
        outcomes=_read_outcomes(args.outcomes_json, report_date=str(args.report_date)),
        min_counterfactual_samples=int(args.min_counterfactual_samples),
        max_missed_edge_bps=float(args.max_missed_edge_bps),
        min_filter_ablation_samples=int(args.min_filter_ablation_samples),
    )
    for path in (output_json, latest_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_json), "status": report["status"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
