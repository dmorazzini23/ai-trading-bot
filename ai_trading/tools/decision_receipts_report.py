"""Build audit receipts that tie decisions to gates, orders, and fills."""

from __future__ import annotations

import argparse
import json
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
            if isinstance(parsed, dict) and (not report_date or _date_match(parsed, report_date)):
                rows.append(parsed)
    return rows


def _date_match(row: Mapping[str, Any], report_date: str) -> bool:
    return _timestamp(row).startswith(report_date)


def _timestamp(row: Mapping[str, Any]) -> str:
    return str(row.get("ts") or row.get("timestamp") or row.get("decision_ts") or row.get("submitted_at") or row.get("filled_at") or "")


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _index_by_decision_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {_decision_id(row): row for row in rows if _decision_id(row)}


def build_decision_receipts_report(
    *,
    report_date: str,
    decisions: Sequence[Mapping[str, Any]],
    order_intents: Sequence[Mapping[str, Any]] = (),
    fills: Sequence[Mapping[str, Any]] = (),
    gate_rows: Sequence[Mapping[str, Any]] = (),
    max_receipts: int = 500,
) -> dict[str, Any]:
    orders_by_id = _index_by_decision_id(order_intents)
    fills_by_id = _index_by_decision_id(fills)
    gates_by_id = _index_by_decision_id(gate_rows)
    receipts: list[dict[str, Any]] = []
    completeness: Counter[str] = Counter()
    for row in decisions[: max(1, int(max_receipts))]:
        decision_id = _decision_id(row)
        order = orders_by_id.get(decision_id, {})
        fill = fills_by_id.get(decision_id, {})
        gate = gates_by_id.get(decision_id, {})
        status = _status(row)
        accepted = status in {"accept", "accepted", "submit", "submitted", "filled", "new"}
        rejected = status in {"reject", "rejected", "blocked", "skip", "skipped"}
        has_terminal_evidence = bool(fill) or bool(gate) or rejected
        receipt_complete = bool(decision_id and (not accepted or order) and has_terminal_evidence)
        completeness["complete" if receipt_complete else "incomplete"] += 1
        receipts.append(
            {
                "decision_id": decision_id or None,
                "symbol": _symbol(row),
                "decision_ts": _timestamp(row) or None,
                "decision_status": status or "unknown",
                "reason": str(row.get("reason") or gate.get("reason") or gate.get("gate") or "unknown"),
                "expected_net_edge_bps": _first_float(row, "expected_net_edge_bps", "expected_edge_bps", "predicted_net_edge_bps"),
                "order_present": bool(order),
                "fill_present": bool(fill),
                "gate_present": bool(gate),
                "realized_net_edge_bps": _first_float(fill, "realized_net_edge_bps", "net_edge_bps", "markout_bps") if fill else None,
                "receipt_complete": receipt_complete,
            }
        )
    status = "complete" if completeness.get("incomplete", 0) == 0 else "gaps"
    return {
        "schema_version": "1.0.0",
        "artifact_type": "decision_receipts_report",
        "report_date": report_date,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "recommended_next_action": "continue_sampling" if status == "complete" else "backfill_missing_decision_receipt_links",
        "summary": {
            "receipts": len(receipts),
            "complete": completeness.get("complete", 0),
            "incomplete": completeness.get("incomplete", 0),
            "truncated": len(decisions) > max(1, int(max_receipts)),
        },
        "receipts": receipts,
        "promotion_authority": False,
        "live_money_authority": False,
    }


def _default_report_paths(report_date: str) -> tuple[Path, Path]:
    root = resolve_runtime_artifact_path("runtime/reports", default_relative="runtime/reports", for_write=True)
    compact = report_date.replace("-", "")
    return root / f"decision_receipts_{compact}.json", root / "decision_receipts_latest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--decisions-jsonl", type=Path, default=None)
    parser.add_argument("--order-intents-jsonl", type=Path, default=None)
    parser.add_argument("--fills-jsonl", type=Path, default=None)
    parser.add_argument("--gate-jsonl", type=Path, default=None)
    parser.add_argument("--max-receipts", type=int, default=500)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)
    output_json, latest_json = _default_report_paths(str(args.report_date))
    output_json = args.output_json or output_json
    latest_json = args.latest_json or latest_json
    report = build_decision_receipts_report(
        report_date=str(args.report_date),
        decisions=_read_jsonl(args.decisions_jsonl, report_date=str(args.report_date)),
        order_intents=_read_jsonl(args.order_intents_jsonl, report_date=str(args.report_date)),
        fills=_read_jsonl(args.fills_jsonl, report_date=str(args.report_date)),
        gate_rows=_read_jsonl(args.gate_jsonl, report_date=str(args.report_date)),
        max_receipts=int(args.max_receipts),
    )
    for path in (output_json, latest_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_json), "status": report["status"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
