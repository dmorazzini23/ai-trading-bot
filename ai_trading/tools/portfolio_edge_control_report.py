"""Summarize portfolio-level edge capture and concentration controls."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def _symbol(row: Mapping[str, Any]) -> str:
    return str(row.get("symbol") or row.get("ticker") or "UNKNOWN").strip().upper() or "UNKNOWN"


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _rows_from_calibration(calibration: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_symbol = calibration.get("bucketed_by_symbol")
    if not isinstance(by_symbol, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for symbol, payload in by_symbol.items():
        if not isinstance(payload, Mapping):
            continue
        count = int(_safe_float(payload.get("count")) or 0)
        expected = _safe_float(payload.get("mean_expected_net_edge_bps"))
        realized = _safe_float(payload.get("mean_realized_net_edge_bps"))
        for _ in range(max(0, count)):
            rows.append(
                {
                    "symbol": str(symbol).upper(),
                    "expected_net_edge_bps": expected,
                    "realized_net_edge_bps": realized,
                }
            )
    return rows


def build_portfolio_edge_control_report(
    *,
    report_date: str,
    fills: Sequence[Mapping[str, Any]] = (),
    expected_edge_calibration: Mapping[str, Any] | None = None,
    min_capture_ratio: float = 0.25,
    max_symbol_edge_share: float = 0.60,
    min_samples: int = 10,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source = "fills"
    for row in fills:
        expected = _first_float(row, "expected_net_edge_bps", "expected_edge_bps", "predicted_net_edge_bps")
        realized = _first_float(row, "realized_net_edge_bps", "net_edge_bps", "markout_bps")
        if expected is None or realized is None:
            continue
        rows.append(
            {
                "symbol": _symbol(row),
                "expected_net_edge_bps": float(expected),
                "realized_net_edge_bps": float(realized),
            }
        )
    if not rows and expected_edge_calibration:
        rows = _rows_from_calibration(expected_edge_calibration)
        source = "expected_edge_calibration"
    expected_values = [float(row["expected_net_edge_bps"]) for row in rows if row["expected_net_edge_bps"] is not None]
    realized_values = [float(row["realized_net_edge_bps"]) for row in rows if row["realized_net_edge_bps"] is not None]
    positive_expected = sum(value for value in expected_values if value > 0.0)
    realized_total = sum(realized_values)
    capture_ratio = float(realized_total / positive_expected) if positive_expected > 0.0 else None
    symbol_edge: dict[str, float] = defaultdict(float)
    symbol_realized: dict[str, float] = defaultdict(float)
    symbol_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        symbol = str(row["symbol"])
        expected = float(row["expected_net_edge_bps"] or 0.0)
        realized = float(row["realized_net_edge_bps"] or 0.0)
        symbol_edge[symbol] += max(0.0, expected)
        symbol_realized[symbol] += realized
        symbol_counts[symbol] += 1
    max_share = max((edge / positive_expected for edge in symbol_edge.values()), default=0.0) if positive_expected > 0.0 else 0.0
    dominant_symbol = max(symbol_edge, key=symbol_edge.get) if symbol_edge else None
    breaches: list[str] = []
    if len(rows) < int(min_samples):
        breaches.append("insufficient_samples")
    if capture_ratio is not None and capture_ratio < float(min_capture_ratio):
        breaches.append("portfolio_capture_ratio_low")
    if max_share > float(max_symbol_edge_share):
        breaches.append("symbol_edge_concentration")
    status = "ok" if not breaches else ("insufficient_samples" if breaches == ["insufficient_samples"] else "control_breach")
    action = "continue_sampling" if status == "ok" else "keep_tiny_sampling_and_review_edge_controls"
    return {
        "schema_version": "1.0.0",
        "artifact_type": "portfolio_edge_control_report",
        "report_date": report_date,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "recommended_next_action": action,
        "source": source,
        "controls": {
            "min_samples": int(min_samples),
            "min_capture_ratio": float(min_capture_ratio),
            "max_symbol_edge_share": float(max_symbol_edge_share),
            "breaches": breaches,
        },
        "summary": {
            "samples": len(rows),
            "mean_expected_net_edge_bps": _mean(expected_values),
            "mean_realized_net_edge_bps": _mean(realized_values),
            "portfolio_expected_edge_bps": float(positive_expected),
            "portfolio_realized_edge_bps": float(realized_total),
            "portfolio_capture_ratio": capture_ratio,
            "dominant_symbol": dominant_symbol,
            "dominant_symbol_edge_share": float(max_share),
        },
        "by_symbol": {
            symbol: {
                "count": symbol_counts[symbol],
                "positive_expected_edge_bps": symbol_edge[symbol],
                "realized_edge_bps": symbol_realized[symbol],
                "edge_share": (
                    float(symbol_edge[symbol] / positive_expected) if positive_expected > 0.0 else None
                ),
            }
            for symbol in sorted(symbol_counts)
        },
        "promotion_authority": False,
        "live_money_authority": False,
    }


def _default_report_paths(report_date: str) -> tuple[Path, Path]:
    root = resolve_runtime_artifact_path("runtime/reports", default_relative="runtime/reports", for_write=True)
    compact = report_date.replace("-", "")
    return root / f"portfolio_edge_control_{compact}.json", root / "portfolio_edge_control_latest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--fills-jsonl", type=Path, default=None)
    parser.add_argument("--expected-edge-calibration-json", type=Path, default=None)
    parser.add_argument("--min-capture-ratio", type=float, default=0.25)
    parser.add_argument("--max-symbol-edge-share", type=float, default=0.60)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)
    output_json, latest_json = _default_report_paths(str(args.report_date))
    output_json = args.output_json or output_json
    latest_json = args.latest_json or latest_json
    report = build_portfolio_edge_control_report(
        report_date=str(args.report_date),
        fills=_read_jsonl(args.fills_jsonl, report_date=str(args.report_date)),
        expected_edge_calibration=_read_json(args.expected_edge_calibration_json),
        min_capture_ratio=float(args.min_capture_ratio),
        max_symbol_edge_share=float(args.max_symbol_edge_share),
        min_samples=int(args.min_samples),
    )
    for path in (output_json, latest_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_json), "status": report["status"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
