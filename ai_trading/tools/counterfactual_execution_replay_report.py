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


def _date_match(row: Mapping[str, Any], report_date: str) -> bool:
    ts = str(row.get("ts") or row.get("timestamp") or row.get("decision_ts") or row.get("filled_at") or "")
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
    return str(row.get("decision_id") or row.get("intent_id") or row.get("client_order_id") or "").strip()


def _symbol(row: Mapping[str, Any]) -> str:
    return str(row.get("symbol") or row.get("ticker") or "UNKNOWN").strip().upper() or "UNKNOWN"


def _status(row: Mapping[str, Any]) -> str:
    return str(row.get("status") or row.get("action") or row.get("decision") or "").strip().lower()


def _is_accepted(row: Mapping[str, Any]) -> bool:
    return _status(row) in {"accept", "accepted", "submit", "submitted", "filled", "new"}


def _is_rejected(row: Mapping[str, Any]) -> bool:
    return _status(row) in {"reject", "rejected", "blocked", "skip", "skipped"}


def build_counterfactual_execution_replay_report(
    *,
    report_date: str,
    decisions: Sequence[Mapping[str, Any]],
    fills: Sequence[Mapping[str, Any]] = (),
    min_counterfactual_samples: int = 10,
    max_missed_edge_bps: float = 25.0,
) -> dict[str, Any]:
    fills_by_id = {_decision_id(row): row for row in fills if _decision_id(row)}
    accepted_edges: list[float] = []
    rejected_edges: list[float] = []
    missed_positive: list[dict[str, Any]] = []
    avoided_negative = 0
    rejection_reasons: Counter[str] = Counter()
    accepted = 0
    rejected = 0
    for row in decisions:
        realized = _first_float(row, "counterfactual_net_edge_bps", "realized_net_edge_bps", "markout_bps", "net_edge_bps")
        if realized is None and _decision_id(row) in fills_by_id:
            realized = _first_float(fills_by_id[_decision_id(row)], "realized_net_edge_bps", "markout_bps", "net_edge_bps")
        if _is_accepted(row):
            accepted += 1
            if realized is not None:
                accepted_edges.append(float(realized))
        elif _is_rejected(row):
            rejected += 1
            rejection_reasons[str(row.get("reason") or row.get("gate") or "unknown")] += 1
            if realized is None:
                continue
            rejected_edges.append(float(realized))
            if realized > 0.0:
                missed_positive.append(
                    {
                        "decision_id": _decision_id(row) or None,
                        "symbol": _symbol(row),
                        "reason": str(row.get("reason") or row.get("gate") or "unknown"),
                        "counterfactual_net_edge_bps": float(realized),
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
    return {
        "schema_version": "1.0.0",
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
        "missed_positive_decisions": sorted(
            missed_positive,
            key=lambda item: float(item["counterfactual_net_edge_bps"]),
            reverse=True,
        )[:25],
        "promotion_authority": False,
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
    parser.add_argument("--min-counterfactual-samples", type=int, default=10)
    parser.add_argument("--max-missed-edge-bps", type=float, default=25.0)
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
        min_counterfactual_samples=int(args.min_counterfactual_samples),
        max_missed_edge_bps=float(args.max_missed_edge_bps),
    )
    for path in (output_json, latest_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_json), "status": report["status"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
