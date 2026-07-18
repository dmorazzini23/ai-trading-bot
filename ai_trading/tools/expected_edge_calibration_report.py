"""Report whether expected edge converts into realized execution edge."""

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
from ai_trading.tools.execution_capture_improvement_report import (
    build_metadata_quality,
    enrich_fills_with_tca,
    normalize_execution_metadata,
)


_EDGE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("edge_le_0", -math.inf, 0.0),
    ("edge_0_5", 0.0, 5.0),
    ("edge_5_10", 5.0, 10.0),
    ("edge_10_25", 10.0, 25.0),
    ("edge_25_50", 25.0, 50.0),
    ("edge_gt_50", 50.0, math.inf),
)


def _read_jsonl(
    path: Path | None,
    *,
    report_date: str | None = None,
    max_rows: int = 250_000,
) -> list[dict[str, Any]]:
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
            if len(rows) > max(1, int(max_rows)):
                rows.pop(0)
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
    return parsed


def _first_float(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    return None


def _symbol(row: Mapping[str, Any]) -> str:
    return str(row.get("symbol") or row.get("ticker") or "UNKNOWN").strip().upper() or "UNKNOWN"


def _side(row: Mapping[str, Any]) -> str:
    raw = str(row.get("side") or row.get("order_side") or row.get("intended_side") or "unknown")
    token = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if token in {"short", "sellshort", "sell_short"}:
        return "sell_short"
    if token in {"buy", "sell"}:
        return token
    return "unknown"


def _session(row: Mapping[str, Any]) -> str:
    return str(
        row.get("session_bucket")
        or row.get("session_regime")
        or row.get("session")
        or row.get("time_of_day")
        or "unknown"
    ).strip().lower() or "unknown"


def _regime(row: Mapping[str, Any]) -> str:
    return str(
        row.get("regime")
        or row.get("market_regime")
        or row.get("volatility_regime")
        or "unknown"
    ).strip().lower() or "unknown"


def _spread_bucket(value: float | None) -> str:
    if value is None:
        return "spread_unknown"
    if value <= 10.0:
        return "spread_tight"
    if value <= 35.0:
        return "spread_normal"
    return "spread_wide"


def _quote_age_bucket(value: float | None) -> str:
    if value is None:
        return "quote_age_unknown"
    if value <= 1_000.0:
        return "quote_fresh"
    if value <= 5_000.0:
        return "quote_aging"
    return "quote_stale"


def _edge_bucket(value: float | None) -> str:
    if value is None:
        return "edge_unknown"
    for label, lower, upper in _EDGE_BUCKETS:
        if lower < value <= upper:
            return label
    return "edge_unknown"


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _profit_factor(values: Sequence[float]) -> float | None:
    wins = sum(value for value in values if value > 0.0)
    losses = abs(sum(value for value in values if value < 0.0))
    if losses <= 0.0:
        return None
    return float(wins / losses)


def _capture_ratio(expected: Sequence[float], realized: Sequence[float]) -> float | None:
    numerator = sum(realized)
    denominator = sum(value for value in expected if value > 0.0)
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _classify_attribution(row: Mapping[str, Any]) -> str:
    expected = _first_float(row, "expected_net_edge_bps", "expected_edge_bps", "predicted_net_edge_bps")
    realized = _first_float(row, "realized_net_edge_bps", "net_edge_bps", "markout_bps")
    slippage = abs(_first_float(row, "slippage_bps", "slippage_drag_bps") or 0.0)
    spread = _first_float(row, "spread_bps", "decision_spread_bps")
    quote_age = _first_float(row, "quote_age_ms", "quote_fresh_ms")
    latency = _first_float(row, "latency_ms", "fill_latency_ms", "submit_latency_ms")
    if expected is None or realized is None:
        return "insufficient_edge_fields"
    if quote_age is not None and quote_age > 5_000.0:
        return "stale_quote"
    if spread is not None and spread > 50.0:
        return "wide_spread"
    if slippage > max(5.0, abs(expected) * 0.5):
        return "spread_slippage_drag"
    if latency is not None and latency > 3_000.0:
        return "fill_timing_latency"
    if expected > 0.0 and realized < 0.0:
        return "bad_signal_direction_or_adverse_selection"
    if expected > 0.0 and realized < expected * 0.25:
        return "weak_execution_capture"
    return "captured_expected_edge"


def _bucket_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unknown")].append(row)
    output: dict[str, dict[str, Any]] = {}
    for bucket, bucket_rows in sorted(groups.items()):
        expected = [
            float(value)
            for row in bucket_rows
            if (value := _safe_float(row.get("expected_net_edge_bps"))) is not None
        ]
        realized = [
            float(value)
            for row in bucket_rows
            if (value := _safe_float(row.get("realized_net_edge_bps"))) is not None
        ]
        slippage = [
            float(value)
            for row in bucket_rows
            if (value := _safe_float(row.get("slippage_bps"))) is not None
        ]
        output[bucket] = {
            "count": len(bucket_rows),
            "mean_expected_net_edge_bps": _mean(expected),
            "mean_realized_net_edge_bps": _mean(realized),
            "mean_slippage_bps": _mean(slippage),
            "win_rate": (
                float(sum(1 for value in realized if value > 0.0) / len(realized))
                if realized
                else None
            ),
            "profit_factor": _profit_factor(realized),
            "capture_ratio": _capture_ratio(expected, realized),
        }
    return output


def _ordered_quartile_means(rows: Sequence[Mapping[str, Any]]) -> tuple[float | None, float | None]:
    eligible = [
        row
        for row in rows
        if _safe_float(row.get("expected_net_edge_bps")) is not None
        and _safe_float(row.get("realized_net_edge_bps")) is not None
    ]
    if len(eligible) < 4:
        return None, None
    ordered = sorted(eligible, key=lambda row: float(row["expected_net_edge_bps"]))
    width = max(1, len(ordered) // 4)
    low = [float(row["realized_net_edge_bps"]) for row in ordered[:width]]
    high = [float(row["realized_net_edge_bps"]) for row in ordered[-width:]]
    return _mean(low), _mean(high)


def _diagnostic_status(rows: Sequence[Mapping[str, Any]], min_samples: int) -> tuple[str, str]:
    if len(rows) < int(min_samples):
        return "insufficient_samples", "collect_more_diagnostic_paper_samples"
    expected = [float(row["expected_net_edge_bps"]) for row in rows]
    realized = [float(row["realized_net_edge_bps"]) for row in rows]
    capture = _capture_ratio(expected, realized)
    low_mean, high_mean = _ordered_quartile_means(rows)
    high_expected = [row for row in rows if float(row["expected_net_edge_bps"]) > 10.0]
    high_realized = _mean([float(row["realized_net_edge_bps"]) for row in high_expected])
    if low_mean is not None and high_mean is not None and high_mean < low_mean:
        return "inverted", "pause_scaling_and_retrain_expected_edge_labels"
    if high_expected and high_realized is not None and high_realized <= 0.0:
        return "overestimated", "keep_tiny_sampling_and_recalibrate_signal"
    if capture is not None and capture < 0.25:
        return "overestimated", "diagnose_execution_capture_and_cost_labels"
    return "calibrated", "keep_sampling_with_current_limits"


def _calibration_correction(
    *,
    expected_values: Sequence[float],
    realized_values: Sequence[float],
    rows: Sequence[Mapping[str, Any]],
    min_samples: int,
) -> dict[str, Any]:
    sample_count = min(len(expected_values), len(realized_values))
    capture = _capture_ratio(expected_values, realized_values)
    multiplier: float | None
    if sample_count < int(min_samples) or capture is None:
        multiplier = None
        action = "collect_more_samples"
        production_scaling_allowed = False
        reasons = ["insufficient_calibration_samples"]
    elif capture <= 0.0:
        multiplier = 0.0
        action = "shadow_or_retrain_expected_edge"
        production_scaling_allowed = False
        reasons = ["negative_or_zero_capture_ratio"]
    elif capture < 0.25:
        multiplier = max(0.0, min(float(capture), 1.0))
        action = "keep_tiny_sampling_and_recalibrate"
        production_scaling_allowed = False
        reasons = ["weak_capture_ratio"]
    else:
        multiplier = max(0.0, min(float(capture), 1.0))
        action = "apply_calibrated_edge_multiplier"
        production_scaling_allowed = capture >= 0.75
        reasons = ["calibration_multiplier_available"]
    side_multipliers: dict[str, dict[str, Any]] = {}
    for side, summary in _bucket_summary(rows, "side").items():
        side_capture = _safe_float(summary.get("capture_ratio"))
        side_count = int(summary.get("count") or 0)
        if side_count < max(1, int(min_samples // 2)):
            side_action = "observe"
            side_multiplier = None
            side_reasons = ["insufficient_side_samples"]
        elif side_capture is None:
            side_action = "observe"
            side_multiplier = None
            side_reasons = ["missing_side_capture_ratio"]
        elif side_capture <= 0.0:
            side_action = "shadow_or_retrain_side"
            side_multiplier = 0.0
            side_reasons = ["negative_or_zero_side_capture_ratio"]
        elif side_capture < 0.25:
            side_action = "downscale_side"
            side_multiplier = max(0.0, min(float(side_capture), 1.0))
            side_reasons = ["weak_side_capture_ratio"]
        else:
            side_action = "allow_with_multiplier"
            side_multiplier = max(0.0, min(float(side_capture), 1.0))
            side_reasons = ["side_calibration_multiplier_available"]
        side_multipliers[side] = {
            "samples": side_count,
            "capture_ratio": side_capture,
            "expected_edge_multiplier": side_multiplier,
            "recommended_action": side_action,
            "reasons": side_reasons,
        }
    return {
        "sample_count": sample_count,
        "global_capture_ratio": capture,
        "expected_edge_multiplier": multiplier,
        "recommended_action": action,
        "production_scaling_allowed": bool(production_scaling_allowed),
        "side_multipliers": side_multipliers,
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
        "reasons": reasons,
    }


def _exit_quality_diagnostics(rows: Sequence[Mapping[str, Any]], min_samples: int) -> dict[str, Any]:
    exit_rows = [row for row in rows if str(row.get("side") or "").lower() == "sell"]
    if not exit_rows:
        return {
            "samples": 0,
            "status": "missing",
            "recommended_action": "collect_sell_exit_samples",
            "runtime_authority": False,
        }
    summary = _bucket_summary(exit_rows, "side").get("sell", {})
    count = int(summary.get("count") or 0)
    capture = _safe_float(summary.get("capture_ratio"))
    mean_realized = _safe_float(summary.get("mean_realized_net_edge_bps"))
    if count < max(1, int(min_samples // 2)):
        status = "insufficient_samples"
        action = "collect_more_sell_exit_samples"
    elif capture is not None and capture <= 0.0:
        status = "inverted"
        action = "review_exit_timing_and_order_type_before_scaling"
    elif mean_realized is not None and mean_realized < 0.0:
        status = "negative_realized_edge"
        action = "tighten_exit_execution_quality_controls"
    else:
        status = "acceptable"
        action = "continue_exit_sampling"
    return {
        "samples": count,
        "status": status,
        "mean_realized_net_edge_bps": mean_realized,
        "capture_ratio": capture,
        "profit_factor": summary.get("profit_factor"),
        "win_rate": summary.get("win_rate"),
        "recommended_action": action,
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
    }


def build_expected_edge_calibration_report(
    *,
    report_date: str,
    fills: Sequence[Mapping[str, Any]],
    tca_rows: Sequence[Mapping[str, Any]] = (),
    candidates: Sequence[Mapping[str, Any]] = (),
    gate_rows: Sequence[Mapping[str, Any]] = (),
    min_samples: int = 25,
) -> dict[str, Any]:
    realized_rows: list[dict[str, Any]] = []
    for row in enrich_fills_with_tca(fills, tca_rows):
        expected = _first_float(row, "expected_net_edge_bps", "expected_edge_bps", "predicted_net_edge_bps")
        realized = _first_float(row, "realized_net_edge_bps", "net_edge_bps", "markout_bps")
        if expected is None or realized is None:
            continue
        slippage = _first_float(row, "slippage_bps", "slippage_drag_bps")
        execution_metadata = normalize_execution_metadata(row)
        normalized = {
            **execution_metadata,
            "symbol": _symbol(row),
            "side": _side(row),
            "session_bucket": execution_metadata["session"],
            "expected_net_edge_bps": float(expected),
            "realized_net_edge_bps": float(realized),
            "slippage_bps": float(slippage) if slippage is not None else None,
            "expected_edge_bucket": _edge_bucket(expected),
            "attribution": _classify_attribution({**dict(row), **execution_metadata}),
        }
        realized_rows.append(normalized)

    rejected_rows: list[dict[str, Any]] = []
    for row in list(candidates) + list(gate_rows):
        status = str(row.get("status") or row.get("action") or row.get("decision") or "").lower()
        if status not in {"reject", "rejected", "blocked", "skip", "skipped"}:
            continue
        expected = _first_float(row, "expected_net_edge_bps", "expected_edge_bps", "predicted_net_edge_bps")
        rejected_rows.append(
            {
                "symbol": _symbol(row),
                "side": _side(row),
                "session_bucket": _session(row),
                "expected_edge_bucket": _edge_bucket(expected),
                "reason": str(row.get("reason") or row.get("gate") or "unknown"),
            }
        )

    status, action = _diagnostic_status(realized_rows, int(min_samples))
    expected_values = [float(row["expected_net_edge_bps"]) for row in realized_rows]
    realized_values = [float(row["realized_net_edge_bps"]) for row in realized_rows]
    bucket_by_edge = _bucket_summary(realized_rows, "expected_edge_bucket")
    worst_buckets = sorted(
        (
            {
                "bucket": bucket,
                "count": summary["count"],
                "expected_minus_realized_bps": (
                    float(summary["mean_expected_net_edge_bps"])
                    - float(summary["mean_realized_net_edge_bps"])
                ),
                "mean_expected_net_edge_bps": summary["mean_expected_net_edge_bps"],
                "mean_realized_net_edge_bps": summary["mean_realized_net_edge_bps"],
            }
            for bucket, summary in bucket_by_edge.items()
            if summary["mean_expected_net_edge_bps"] is not None
            and summary["mean_realized_net_edge_bps"] is not None
        ),
        key=lambda item: float(item["expected_minus_realized_bps"]),
        reverse=True,
    )[:5]
    attribution_counts = Counter(str(row["attribution"]) for row in realized_rows)
    rejected_by_bucket = Counter(str(row["expected_edge_bucket"]) for row in rejected_rows)
    rejected_by_reason = Counter(str(row["reason"]) for row in rejected_rows)
    metadata_quality = build_metadata_quality(realized_rows)
    return {
        "schema_version": "1.0.0",
        "artifact_type": "expected_edge_calibration_report",
        "report_date": report_date,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "metadata_status": metadata_quality["status"],
        "warnings": metadata_quality["warnings"],
        "recommended_next_action": action,
        "sample_gate": {
            "min_samples": int(min_samples),
            "realized_samples": len(realized_rows),
            "sufficient": len(realized_rows) >= int(min_samples),
        },
        "summary": {
            "mean_expected_net_edge_bps": _mean(expected_values),
            "mean_realized_net_edge_bps": _mean(realized_values),
            "capture_ratio": _capture_ratio(expected_values, realized_values),
            "win_rate": (
                float(sum(1 for value in realized_values if value > 0.0) / len(realized_values))
                if realized_values
                else None
            ),
            "profit_factor": _profit_factor(realized_values),
            "rejected_records": len(rejected_rows),
        },
        "bucketed_by_expected_edge": bucket_by_edge,
        "bucketed_by_symbol": _bucket_summary(realized_rows, "symbol"),
        "bucketed_by_side": _bucket_summary(realized_rows, "side"),
        "bucketed_by_session": _bucket_summary(realized_rows, "session_bucket"),
        "bucketed_by_spread": _bucket_summary(realized_rows, "spread_bucket"),
        "bucketed_by_quote_age": _bucket_summary(realized_rows, "quote_age_bucket"),
        "bucketed_by_regime": _bucket_summary(realized_rows, "regime"),
        "bucketed_by_execution_profile": _bucket_summary(realized_rows, "execution_profile"),
        "bucketed_by_order_type": _bucket_summary(realized_rows, "order_type"),
        "metadata_quality": metadata_quality,
        "normalized_rows": realized_rows,
        "execution_capture_diagnosis": {
            "attribution_counts": dict(sorted(attribution_counts.items())),
            "worst_buckets": worst_buckets,
        },
        "calibration_correction": _calibration_correction(
            expected_values=expected_values,
            realized_values=realized_values,
            rows=realized_rows,
            min_samples=int(min_samples),
        ),
        "exit_quality_diagnostics": _exit_quality_diagnostics(
            realized_rows,
            int(min_samples),
        ),
        "rejected_decisions": {
            "count": len(rejected_rows),
            "by_expected_edge_bucket": dict(sorted(rejected_by_bucket.items())),
            "by_reason": dict(sorted(rejected_by_reason.items())),
        },
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
    return root / f"expected_edge_calibration_{compact}.json", root / "expected_edge_calibration_latest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--fills-jsonl", type=Path, default=None)
    parser.add_argument("--tca-jsonl", type=Path, default=None)
    parser.add_argument("--candidates-jsonl", type=Path, default=None)
    parser.add_argument("--gate-jsonl", type=Path, default=None)
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)
    output_json, latest_json = _default_report_paths(str(args.report_date))
    output_json = args.output_json or output_json
    latest_json = args.latest_json or latest_json
    report = build_expected_edge_calibration_report(
        report_date=str(args.report_date),
        fills=_read_jsonl(args.fills_jsonl, report_date=str(args.report_date)),
        tca_rows=_read_jsonl(args.tca_jsonl, report_date=str(args.report_date)),
        candidates=_read_jsonl(args.candidates_jsonl, report_date=str(args.report_date)),
        gate_rows=_read_jsonl(args.gate_jsonl, report_date=str(args.report_date)),
        min_samples=int(args.min_samples),
    )
    for path in (output_json, latest_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_json), "status": report["status"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
