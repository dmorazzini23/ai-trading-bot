"""Build audit receipts that tie decisions to gates, orders, and fills."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.tools.execution_capture_improvement_report import (
    is_fill_based_execution_evidence,
)


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


def _nested(row: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: Any = row
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def _value(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        current: Any = row
        found = True
        for part in key.split("."):
            if isinstance(current, Mapping) and part in current:
                current = current.get(part)
                continue
            if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
                try:
                    index = int(part)
                except ValueError:
                    found = False
                    break
                if index < 0 or index >= len(current):
                    found = False
                    break
                current = current[index]
                continue
            found = False
            break
        if found and current not in (None, ""):
            return current
    return None


def _timestamp(row: Mapping[str, Any]) -> str:
    return str(
        _value(
            row,
            "ts",
            "timestamp",
            "decision_ts",
            "submitted_at",
            "filled_at",
            "decision_journal.bar_ts",
            "decision_journal.ts",
            "decision_journal.order_intent.bar_ts",
            "bar_ts",
        )
        or ""
    )


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        parsed = _safe_float(_value(row, key))
        if parsed is not None:
            return parsed
    return None


def _decision_id(row: Mapping[str, Any]) -> str:
    return str(
        _value(
            row,
            "correlation_id",
            "decision_journal.correlation_id",
            "metrics.correlation_id",
            "order.correlation_id",
            "decision_journal.order_intent.correlation_id",
            "decision_id",
            "decision_trace_id",
            "intent_id",
            "client_order_id",
            "order_id",
            "decision_journal.decision_trace_id",
            "decision_journal.client_order_id",
            "decision_journal.order_intent.decision_trace_id",
            "decision_journal.order_intent.client_order_id",
        )
        or ""
    ).strip()


def _symbol(row: Mapping[str, Any]) -> str:
    return str(
        _value(
            row,
            "symbol",
            "ticker",
            "decision_journal.symbol",
            "decision_journal.risk_decision.symbol",
            "decision_journal.order_intent.symbol",
            "net_target.symbol",
        )
        or "UNKNOWN"
    ).strip().upper() or "UNKNOWN"


def _status(row: Mapping[str, Any]) -> str:
    status = str(
        _value(row, "status", "action", "decision", "decision_journal.status")
        or ""
    ).strip().lower()
    if status:
        return status
    journal = _nested(row, "decision_journal")
    if journal:
        if bool(journal.get("submitted")):
            return "submitted"
        if bool(journal.get("accepted")):
            return "accepted"
        reasons = journal.get("reasons")
        gates = journal.get("gates")
        if reasons or gates:
            return "rejected"
    if _safe_float(row.get("accepted_records")) and (_safe_float(row.get("accepted_records")) or 0.0) > 0.0:
        return "accepted"
    if _safe_float(row.get("rejected_records")) and (_safe_float(row.get("rejected_records")) or 0.0) > 0.0:
        return "rejected"
    return ""


def _correlation_id(row: Mapping[str, Any]) -> str | None:
    value = _value(
        row,
        "correlation_id",
        "decision_journal.correlation_id",
        "metrics.correlation_id",
        "order.correlation_id",
        "decision_journal.order_intent.correlation_id",
        "tca.correlation_id",
    )
    token = str(value or "").strip()
    return token or None


def _index_by_decision_id(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    indexed: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        for key in _link_keys(row):
            bucket = indexed.setdefault(key, [])
            if not any(candidate is row for candidate in bucket):
                bucket.append(row)
    return indexed


def _link_keys(row: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    correlation_id = _correlation_id(row)
    if correlation_id:
        keys.append(f"correlation:{correlation_id}")
    for key in (
        "decision_id",
        "decision_trace_id",
        "intent_id",
        "client_order_id",
        "order_id",
        "broker_order_id",
        "fill_id",
        "execution_id",
        "order.decision_trace_id",
        "order.client_order_id",
        "order.order_id",
        "order.broker_order_id",
        "decision_journal.decision_trace_id",
        "decision_journal.client_order_id",
        "decision_journal.order_intent.decision_trace_id",
        "decision_journal.order_intent.client_order_id",
        "decision_journal.order_intent.order_id",
    ):
        value = str(_value(row, key) or "").strip()
        link_key = f"identity:{value}"
        if value and link_key not in keys:
            keys.append(link_key)
    return keys


def _hard_identity_keys(row: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    for key in (
        "client_order_id",
        "order_id",
        "broker_order_id",
        "fill_id",
        "execution_id",
        "order.client_order_id",
        "order.order_id",
        "order.broker_order_id",
        "decision_journal.client_order_id",
        "decision_journal.order_intent.client_order_id",
        "decision_journal.order_intent.order_id",
    ):
        value = str(_value(row, key) or "").strip()
        link_key = f"identity:{value}"
        if value and link_key not in keys:
            keys.append(link_key)
    return keys


def _first_linked(
    indexes: Mapping[str, Sequence[Mapping[str, Any]]],
    row: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    keys = _link_keys(row)
    correlation_key = next(
        (key for key in keys if key.startswith("correlation:")),
        None,
    )
    identity_keys = [key for key in keys if key.startswith("identity:")]
    if correlation_key is not None:
        correlated = list(indexes.get(correlation_key, ()))
        matching_identity_keys = _hard_identity_keys(row) or identity_keys
        if matching_identity_keys:
            exact = [
                candidate
                for candidate in correlated
                if any(
                    candidate in indexes.get(key, ())
                    for key in matching_identity_keys
                )
            ]
            unique_exact = list({id(candidate): candidate for candidate in exact}.values())
            if len(unique_exact) == 1:
                return unique_exact[0], "correlation_and_order_id"
            if len(unique_exact) > 1:
                return {}, "ambiguous_correlation_and_order_id"
            return {}, "correlation_order_id_unmatched"
        if len(correlated) == 1:
            return correlated[0], "correlation_id"
        if len(correlated) > 1:
            return {}, "ambiguous_correlation_id"
        return {}, "correlation_id_unmatched"

    legacy_candidates = [
        candidate
        for key in identity_keys
        for candidate in indexes.get(key, ())
    ]
    unique_legacy = list(
        {id(candidate): candidate for candidate in legacy_candidates}.values()
    )
    if len(unique_legacy) == 1:
        return unique_legacy[0], "legacy_exact_id"
    if len(unique_legacy) > 1:
        return {}, "ambiguous_legacy_exact_id"
    return {}, "unmatched"


def _reasons(row: Mapping[str, Any], gate: Mapping[str, Any]) -> list[str]:
    raw_values = [
        _value(row, "reasons"),
        _value(row, "gates"),
        _value(row, "decision_journal.reasons"),
        _value(row, "decision_journal.gates"),
        _value(row, "decision_journal.risk_decision.reasons"),
        _value(row, "decision_journal.risk_decision.gates"),
        _value(gate, "reason"),
        _value(gate, "gate"),
    ]
    out: list[str] = []
    for raw in raw_values:
        values = raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else [raw]
        for value in values:
            text = str(value or "").strip()
            if text and text not in out:
                out.append(text)
    return out


def _synthetic_decision_id(row: Mapping[str, Any], *, index: int) -> str:
    ts = _timestamp(row)
    symbol = _symbol(row)
    reasons = _reasons(row, {})
    reason = reasons[0] if reasons else "no_reason"
    base = "|".join(
        [
            ts or f"row-{index}",
            symbol,
            reason,
            str(index),
        ]
    )
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"decision-summary:{digest}"


def _aggregate_gate_receipt(row: Mapping[str, Any], *, index: int) -> dict[str, Any] | None:
    total = int(_safe_float(row.get("records_total")) or 0)
    accepted = int(_safe_float(row.get("accepted_records")) or 0)
    rejected = int(_safe_float(row.get("rejected_records")) or 0)
    if total <= 0 and accepted <= 0 and rejected <= 0:
        return None
    ts = _timestamp(row)
    gate_counts = row.get("gate_counts")
    symbol_attribution = row.get("symbol_attribution")
    top_gates: list[dict[str, Any]] = []
    if isinstance(gate_counts, Mapping):
        for gate, count in gate_counts.items():
            top_gates.append({"gate": str(gate), "count": int(_safe_float(count) or 0)})
        top_gates.sort(key=lambda item: (-int(_safe_float(item.get("count")) or 0), str(item.get("gate"))))
    symbols = (
        sorted(str(symbol).upper() for symbol in symbol_attribution)
        if isinstance(symbol_attribution, Mapping)
        else []
    )
    return {
        "decision_id": f"gate-summary:{ts or index}",
        "symbol": ",".join(symbols) if symbols else "MULTI",
        "decision_ts": ts or None,
        "decision_status": "aggregate",
        "reason": top_gates[0]["gate"] if top_gates else "gate_effectiveness_summary",
        "reasons": [item["gate"] for item in top_gates],
        "expected_net_edge_bps": _first_float(row, "total_expected_net_edge_bps", "total_edge_proxy_bps"),
        "order_present": False,
        "fill_present": False,
        "gate_present": True,
        "realized_net_edge_bps": None,
        "receipt_complete": True,
        "receipt_granularity": "aggregate_gate_summary",
        "records_total": total,
        "accepted_records": accepted,
        "rejected_records": rejected,
        "top_gates": top_gates[:5],
    }


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
    fill_evidence = [row for row in fills if is_fill_based_execution_evidence(row)]
    fills_by_id = _index_by_decision_id(fill_evidence)
    gates_by_id = _index_by_decision_id(gate_rows)
    receipts: list[dict[str, Any]] = []
    completeness: Counter[str] = Counter()
    for index, row in enumerate(decisions[: max(1, int(max_receipts))]):
        aggregate_receipt = _aggregate_gate_receipt(row, index=index)
        if aggregate_receipt is not None:
            completeness["complete"] += 1
            receipts.append(aggregate_receipt)
            continue
        decision_id = _decision_id(row)
        order, order_link_method = _first_linked(orders_by_id, row)
        fill, fill_link_method = _first_linked(fills_by_id, row)
        gate, gate_link_method = _first_linked(gates_by_id, row)
        status = _status(row)
        accepted = status in {"accept", "accepted", "submit", "submitted", "filled", "new"}
        rejected = status in {"reject", "rejected", "blocked", "skip", "skipped"}
        reasons = _reasons(row, gate)
        has_terminal_evidence = bool(fill) or bool(gate) or rejected or bool(reasons)
        if not decision_id and rejected and has_terminal_evidence:
            decision_id = _synthetic_decision_id(row, index=index)
        receipt_complete = bool(decision_id and (not accepted or order) and has_terminal_evidence)
        completeness["complete" if receipt_complete else "incomplete"] += 1
        receipts.append(
            {
                "decision_id": decision_id or None,
                "correlation_id": _correlation_id(row),
                "symbol": _symbol(row),
                "decision_ts": _timestamp(row) or None,
                "decision_status": status or "unknown",
                "reason": reasons[0] if reasons else "unknown",
                "reasons": reasons,
                "expected_net_edge_bps": _first_float(
                    row,
                    "expected_net_edge_bps",
                    "expected_edge_bps",
                    "predicted_net_edge_bps",
                    "decision_journal.risk_decision.expected_net_edge_bps",
                    "net_target.proposals.0.debug.expected_net_edge_bps",
                ),
                "order_present": bool(order),
                "fill_present": bool(fill),
                "gate_present": bool(gate),
                "realized_net_edge_bps": _first_float(fill, "realized_net_edge_bps", "net_edge_bps", "markout_bps") if fill else None,
                "receipt_complete": receipt_complete,
                "receipt_granularity": "decision",
                "link_methods": {
                    "order": order_link_method,
                    "fill": fill_link_method,
                    "gate": gate_link_method,
                },
                "fill_based_evidence": bool(fill),
                "promotion_eligible": bool(fill),
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
            "non_fill_rows_excluded": len(fills) - len(fill_evidence),
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
