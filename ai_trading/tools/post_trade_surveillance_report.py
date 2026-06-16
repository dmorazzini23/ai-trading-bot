"""Produce post-trade surveillance findings from decision, order, fill, and OMS artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


REJECT_STATUSES = {"reject", "rejected", "blocked", "skip", "skipped", "canceled", "cancelled"}
SELL_SHORT_TOKENS = {"sell_short", "sellshort", "short", "short_sell"}
PARTIAL_STATUSES = {"partial", "partially_filled", "partially-filled", "partial_fill"}
CONTROLLED_SKIP_CATEGORIES = {"metrics_improvement_control"}


def _read_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_positions(path: Path | None) -> list[dict[str, Any]]:
    parsed = _read_json(path)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, Mapping):
        raw_positions = parsed.get("positions")
        if isinstance(raw_positions, list):
            return [item for item in raw_positions if isinstance(item, dict)]
        return [dict(parsed)]
    return []


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
    ts = str(
        row.get("ts")
        or row.get("timestamp")
        or row.get("decision_ts")
        or row.get("submitted_at")
        or row.get("filled_at")
        or row.get("event_ts")
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


def _token(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip().lower()
        if value:
            return value
    return ""


def _controlled_skip_category(row: Mapping[str, Any]) -> str | None:
    fields = [
        str(row.get(key) or "").strip().lower()
        for key in (
            "controlled_skip",
            "skip_category",
            "detail",
            "reason",
            "gate",
            "last_error",
            "error",
            "rejection_reason",
            "reject_reason",
        )
    ]
    context = row.get("context")
    if isinstance(context, Mapping):
        fields.extend(
            str(context.get(key) or "").strip().lower()
            for key in ("reason", "detail", "gate", "action")
        )
    payload = _parse_payload(row)
    if payload:
        fields.extend(
            str(payload.get(key) or "").strip().lower()
            for key in ("reason", "detail", "gate", "last_error", "error")
        )
    combined = " ".join(field for field in fields if field)
    if "metrics_improvement_control" in combined:
        return "metrics_improvement_control"
    return None


def _symbol(row: Mapping[str, Any]) -> str:
    return str(row.get("symbol") or row.get("ticker") or "UNKNOWN").strip().upper() or "UNKNOWN"


def _row_id(row: Mapping[str, Any]) -> str:
    for key in ("intent_id", "decision_id", "client_order_id", "order_id", "id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _qty(row: Mapping[str, Any]) -> float | None:
    return _first_float(row, "qty", "quantity", "order_qty", "requested_qty")


def _filled_qty(row: Mapping[str, Any]) -> float | None:
    return _first_float(row, "filled_qty", "filled_quantity", "fill_qty", "executed_qty")


def _finding(category: str, row: Mapping[str, Any], detail: str, *, severity: str = "warning") -> dict[str, Any]:
    return {
        "category": category,
        "severity": severity,
        "id": _row_id(row) or None,
        "symbol": _symbol(row),
        "detail": detail,
    }


def _duplicate_intent_findings(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        intent_id = _row_id(row)
        if intent_id:
            grouped[intent_id].append(row)
    findings: list[dict[str, Any]] = []
    for intent_id, bucket in sorted(grouped.items()):
        if len(bucket) > 1:
            findings.append(
                {
                    "category": "duplicate_intent",
                    "severity": "critical",
                    "id": intent_id,
                    "symbol": _symbol(bucket[0]),
                    "detail": f"intent_id repeated {len(bucket)} times",
                }
            )
    return findings


def _parse_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = row.get("payload")
    if isinstance(raw, Mapping):
        return raw
    raw_json = row.get("payload_json")
    if isinstance(raw_json, str) and raw_json.strip():
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def _oms_identifiers(oms_events: Sequence[Mapping[str, Any]]) -> set[str]:
    identifiers: set[str] = set()
    for event in oms_events:
        for key in ("intent_id", "decision_id", "client_order_id", "order_id", "id"):
            value = str(event.get(key) or "").strip()
            if value:
                identifiers.add(value)
        payload = _parse_payload(event)
        for key in ("intent_id", "decision_id", "client_order_id", "order_id", "id"):
            value = str(payload.get(key) or "").strip()
            if value:
                identifiers.add(value)
    return identifiers


def _is_reject(row: Mapping[str, Any]) -> bool:
    if _controlled_skip_category(row) in CONTROLLED_SKIP_CATEGORIES:
        return False
    status = _token(row, "status", "state", "decision", "action")
    return status in REJECT_STATUSES or bool(row.get("reject_reason") or row.get("rejection_reason"))


def _is_sell_short(row: Mapping[str, Any]) -> bool:
    side = _token(row, "side", "order_side", "intended_side", "action")
    return side in SELL_SHORT_TOKENS


def _partial_fill_issue(row: Mapping[str, Any], *, min_fill_ratio: float) -> bool:
    status = _token(row, "status", "state", "fill_status")
    if status in PARTIAL_STATUSES:
        return True
    ratio = _first_float(row, "fill_ratio", "filled_ratio", "quantity_fill_ratio")
    if ratio is not None and ratio < min_fill_ratio:
        return True
    qty = _qty(row)
    filled = _filled_qty(row)
    return qty is not None and filled is not None and 0.0 < filled < qty


def _is_close(row: Mapping[str, Any]) -> bool:
    action = _token(row, "intent", "intent_type", "action", "order_intent")
    return action in {"close", "close_position", "liquidate", "flatten"} or bool(row.get("close_position"))


def _remaining_position(row: Mapping[str, Any]) -> float | None:
    return _first_float(row, "remaining_position", "position_after", "ending_position", "remaining_qty")


def build_post_trade_surveillance_report(
    *,
    report_date: str,
    decisions: Sequence[Mapping[str, Any]] = (),
    orders: Sequence[Mapping[str, Any]] = (),
    fills: Sequence[Mapping[str, Any]] = (),
    oms_events: Sequence[Mapping[str, Any]] = (),
    positions: Sequence[Mapping[str, Any]] = (),
    max_slippage_bps: float = 25.0,
    min_fill_ratio: float = 0.95,
    min_adverse_selection_fills: int = 5,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    controlled_skip_counts: Counter[str] = Counter()
    for row in (*decisions, *orders):
        controlled_skip = _controlled_skip_category(row)
        if controlled_skip:
            controlled_skip_counts[controlled_skip] += 1
        if _is_reject(row):
            findings.append(_finding("reject", row, "order or intent was rejected"))
        if _is_sell_short(row):
            findings.append(_finding("sell_short_attempt", row, "sell_short or short side observed", severity="critical"))

    findings.extend(_duplicate_intent_findings(decisions))

    for row in (*orders, *fills):
        if _partial_fill_issue(row, min_fill_ratio=float(min_fill_ratio)):
            findings.append(_finding("partial_fill_issue", row, "fill ratio or filled quantity indicates a partial fill"))

    for row in fills:
        expected = _first_float(row, "expected_net_edge_bps", "expected_edge_bps", "predicted_net_edge_bps")
        realized = _first_float(row, "realized_net_edge_bps", "net_edge_bps", "markout_bps")
        slippage = _first_float(row, "slippage_bps", "realized_slippage_bps", "slippage_drag_bps")
        if expected is not None and expected > 0.0 and realized is not None and realized < 0.0:
            severity = "critical" if len(fills) >= int(min_adverse_selection_fills) else "warning"
            findings.append(
                _finding(
                    "adverse_selection",
                    row,
                    "positive expected edge realized negative markout",
                    severity=severity,
                )
            )
        if slippage is not None and abs(slippage) > float(max_slippage_bps):
            findings.append(_finding("slippage_breach", row, "slippage exceeded surveillance threshold", severity="critical"))
        if _is_close(row) and (remaining := _remaining_position(row)) is not None and abs(remaining) > 0.0:
            findings.append(_finding("non_flat_close", row, "close intent left a non-flat residual position", severity="critical"))

    for row in positions:
        if _is_close(row) and (remaining := _remaining_position(row)) is not None and abs(remaining) > 0.0:
            findings.append(_finding("non_flat_close", row, "position snapshot remained non-flat after close", severity="critical"))

    oms_ids = _oms_identifiers(oms_events)
    if oms_events:
        for row in (*orders, *fills):
            row_id = _row_id(row)
            if row_id and row_id not in oms_ids:
                findings.append(_finding("oms_mismatch", row, "order or fill id was absent from OMS events", severity="critical"))
            oms_status = str(row.get("oms_status") or "").strip().lower()
            status = _token(row, "status", "state", "fill_status")
            if oms_status and status and oms_status != status:
                findings.append(_finding("oms_mismatch", row, "row status disagreed with OMS status", severity="critical"))

    counts = Counter(str(item["category"]) for item in findings)
    has_critical = any(str(item.get("severity")) == "critical" for item in findings)
    evidence_state = (
        "insufficient_fill_samples"
        if len(fills) < int(min_adverse_selection_fills)
        else "sample_sufficient"
    )
    if not findings:
        status = "clear"
        action = "continue_post_trade_sampling"
    elif has_critical:
        status = "control_breach"
        action = "review_surveillance_findings_before_increasing_risk"
    else:
        only_sample_limited_adverse = (
            evidence_state == "insufficient_fill_samples"
            and all(str(item.get("category")) == "adverse_selection" for item in findings)
        )
        status = "insufficient_samples" if only_sample_limited_adverse else "watchlist"
        action = "review_rejects_and_partial_fills"
    return {
        "schema_version": "1.0.0",
        "artifact_type": "post_trade_surveillance_report",
        "report_date": report_date,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "recommended_next_action": action,
        "thresholds": {
            "max_slippage_bps": float(max_slippage_bps),
            "min_fill_ratio": float(min_fill_ratio),
            "min_adverse_selection_fills": int(min_adverse_selection_fills),
        },
        "summary": {
            "decisions": len(decisions),
            "orders": len(orders),
            "fills": len(fills),
            "oms_events": len(oms_events),
            "positions": len(positions),
            "findings": len(findings),
            "category_counts": dict(sorted(counts.items())),
            "controlled_skips": int(sum(controlled_skip_counts.values())),
            "controlled_skip_counts": dict(sorted(controlled_skip_counts.items())),
            "evidence_state": evidence_state,
        },
        "findings": findings,
        "promotion_authority": False,
        "live_money_authority": False,
    }


def _default_report_paths(report_date: str) -> tuple[Path, Path]:
    root = resolve_runtime_artifact_path("runtime/reports", default_relative="runtime/reports", for_write=True)
    compact = report_date.replace("-", "")
    return root / f"post_trade_surveillance_{compact}.json", root / "post_trade_surveillance_latest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--decisions-jsonl", type=Path, default=None)
    parser.add_argument("--orders-jsonl", type=Path, default=None)
    parser.add_argument("--fills-jsonl", type=Path, default=None)
    parser.add_argument("--oms-jsonl", type=Path, default=None)
    parser.add_argument("--positions-json", type=Path, default=None)
    parser.add_argument("--max-slippage-bps", type=float, default=25.0)
    parser.add_argument("--min-fill-ratio", type=float, default=0.95)
    parser.add_argument("--min-adverse-selection-fills", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)
    output_json, latest_json = _default_report_paths(str(args.report_date))
    output_json = args.output_json or output_json
    latest_json = args.latest_json or latest_json
    report = build_post_trade_surveillance_report(
        report_date=str(args.report_date),
        decisions=_read_jsonl(args.decisions_jsonl, report_date=str(args.report_date)),
        orders=_read_jsonl(args.orders_jsonl, report_date=str(args.report_date)),
        fills=_read_jsonl(args.fills_jsonl, report_date=str(args.report_date)),
        oms_events=_read_jsonl(args.oms_jsonl, report_date=str(args.report_date)),
        positions=_read_positions(args.positions_json),
        max_slippage_bps=float(args.max_slippage_bps),
        min_fill_ratio=float(args.min_fill_ratio),
        min_adverse_selection_fills=int(args.min_adverse_selection_fills),
    )
    for path in (output_json, latest_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_json), "status": report["status"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
