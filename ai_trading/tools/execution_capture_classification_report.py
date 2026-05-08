"""Classify why expected execution edge was or was not captured."""

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


def _token(row: Mapping[str, Any], *keys: str, default: str = "unknown") -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value.lower()
    return default


def _symbol(row: Mapping[str, Any]) -> str:
    return str(row.get("symbol") or row.get("ticker") or "UNKNOWN").strip().upper() or "UNKNOWN"


def classify_execution_capture(row: Mapping[str, Any]) -> str:
    """Return a stable root-cause label for one realized execution row."""

    expected = _first_float(row, "expected_net_edge_bps", "expected_edge_bps", "predicted_net_edge_bps")
    realized = _first_float(row, "realized_net_edge_bps", "net_edge_bps", "markout_bps")
    if expected is None or realized is None:
        return "insufficient_edge_fields"
    quote_age = _first_float(row, "quote_age_ms", "quote_fresh_ms", "decision_quote_age_ms")
    spread = _first_float(row, "spread_bps", "decision_spread_bps")
    slippage = abs(_first_float(row, "slippage_bps", "slippage_drag_bps") or 0.0)
    latency = _first_float(row, "latency_ms", "fill_latency_ms", "submit_latency_ms")
    partial_ratio = _first_float(row, "fill_ratio", "filled_ratio", "quantity_fill_ratio")
    if quote_age is not None and quote_age > 5_000.0:
        return "stale_quote"
    if spread is not None and spread > 50.0:
        return "wide_spread"
    if partial_ratio is not None and partial_ratio < 0.5:
        return "partial_fill"
    if latency is not None and latency > 3_000.0:
        return "fill_timing_latency"
    if slippage > max(5.0, abs(expected) * 0.5):
        return "spread_slippage_drag"
    if expected > 0.0 and realized < 0.0:
        return "adverse_selection"
    if expected > 0.0 and realized < expected * 0.25:
        return "weak_execution_capture"
    return "captured_expected_edge"


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _bucket_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unknown")].append(row)
    output: dict[str, dict[str, Any]] = {}
    for bucket, bucket_rows in sorted(groups.items()):
        expected = [
            value
            for row in bucket_rows
            if (value := _safe_float(row.get("expected_net_edge_bps"))) is not None
        ]
        realized = [
            value
            for row in bucket_rows
            if (value := _safe_float(row.get("realized_net_edge_bps"))) is not None
        ]
        output[bucket] = {
            "count": len(bucket_rows),
            "mean_expected_net_edge_bps": _mean(expected),
            "mean_realized_net_edge_bps": _mean(realized),
            "capture_ratio": (
                float(sum(realized) / sum(value for value in expected if value > 0.0))
                if sum(value for value in expected if value > 0.0) > 0.0
                else None
            ),
        }
    return output


def build_execution_capture_classification_report(
    *,
    report_date: str,
    fills: Sequence[Mapping[str, Any]],
    min_samples: int = 10,
    min_capture_ratio: float = 0.25,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in fills:
        expected = _first_float(row, "expected_net_edge_bps", "expected_edge_bps", "predicted_net_edge_bps")
        realized = _first_float(row, "realized_net_edge_bps", "net_edge_bps", "markout_bps")
        if expected is None or realized is None:
            continue
        rows.append(
            {
                "symbol": _symbol(row),
                "side": _token(row, "side", "order_side"),
                "session_bucket": _token(row, "session_bucket", "session_regime", "session"),
                "order_type": _token(row, "order_type", "submitted_order_type"),
                "expected_net_edge_bps": float(expected),
                "realized_net_edge_bps": float(realized),
                "classification": classify_execution_capture(row),
            }
        )
    expected_values = [float(row["expected_net_edge_bps"]) for row in rows]
    realized_values = [float(row["realized_net_edge_bps"]) for row in rows]
    positive_expected = sum(value for value in expected_values if value > 0.0)
    capture_ratio = float(sum(realized_values) / positive_expected) if positive_expected > 0.0 else None
    counts = Counter(str(row["classification"]) for row in rows)
    if len(rows) < int(min_samples):
        status = "insufficient_samples"
        action = "collect_more_execution_capture_samples"
    elif capture_ratio is not None and capture_ratio < float(min_capture_ratio):
        status = "needs_review"
        action = "inspect_dominant_execution_capture_classification"
    else:
        status = "acceptable"
        action = "continue_sampling"
    return {
        "schema_version": "1.0.0",
        "artifact_type": "execution_capture_classification_report",
        "report_date": report_date,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "recommended_next_action": action,
        "sample_gate": {
            "min_samples": int(min_samples),
            "samples": len(rows),
            "sufficient": len(rows) >= int(min_samples),
        },
        "summary": {
            "mean_expected_net_edge_bps": _mean(expected_values),
            "mean_realized_net_edge_bps": _mean(realized_values),
            "execution_capture_ratio": capture_ratio,
            "classification_counts": dict(sorted(counts.items())),
        },
        "by_symbol": _bucket_summary(rows, "symbol"),
        "by_session": _bucket_summary(rows, "session_bucket"),
        "by_order_type": _bucket_summary(rows, "order_type"),
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
    return root / f"execution_capture_classification_{compact}.json", root / "execution_capture_classification_latest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--fills-jsonl", type=Path, default=None)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--min-capture-ratio", type=float, default=0.25)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)
    output_json, latest_json = _default_report_paths(str(args.report_date))
    output_json = args.output_json or output_json
    latest_json = args.latest_json or latest_json
    report = build_execution_capture_classification_report(
        report_date=str(args.report_date),
        fills=_read_jsonl(args.fills_jsonl, report_date=str(args.report_date)),
        min_samples=int(args.min_samples),
        min_capture_ratio=float(args.min_capture_ratio),
    )
    for path in (output_json, latest_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_json), "status": report["status"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
