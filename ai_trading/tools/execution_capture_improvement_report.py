"""Build paper-first execution capture improvement recommendations.

This report turns recent fill/TCA evidence into bad-bucket diagnostics,
conservative execution recommendations, edge haircuts, and training-label
metadata.  It is intentionally non-authoritative: it can only recommend
downscaling, stricter edge floors, wait/passive/skip behavior, and offline
training priorities.
"""

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
from ai_trading.tools.execution_capture_classification_report import (
    classify_execution_capture,
)


_EDGE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("edge_le_0", -math.inf, 0.0),
    ("edge_0_5", 0.0, 5.0),
    ("edge_5_10", 5.0, 10.0),
    ("edge_10_25", 10.0, 25.0),
    ("edge_25_50", 25.0, 50.0),
    ("edge_gt_50", 50.0, math.inf),
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
        or row.get("pending_resolved_ts")
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
    return float(parsed) if math.isfinite(parsed) else None


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
            return value.lower().replace(" ", "_").replace("-", "_")
    return default


def _symbol(row: Mapping[str, Any]) -> str:
    return str(row.get("symbol") or row.get("ticker") or "UNKNOWN").strip().upper() or "UNKNOWN"


def _side(row: Mapping[str, Any]) -> str:
    token = _token(row, "side", "order_side", "intended_side")
    if token in {"short", "sellshort", "sell_short"}:
        return "sell_short"
    if token in {"buy", "sell", "cover"}:
        return token
    return "unknown"


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
    gross_profit = sum(value for value in values if value > 0.0)
    gross_loss = abs(sum(value for value in values if value < 0.0))
    if gross_loss <= 0.0:
        return None
    return float(gross_profit / gross_loss)


def _capture_ratio(expected: Sequence[float], realized: Sequence[float]) -> float | None:
    expected_positive = sum(value for value in expected if value > 0.0)
    if expected_positive <= 0.0:
        return None
    return float(sum(realized) / expected_positive)


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    expected = _first_float(row, "expected_net_edge_bps", "expected_edge_bps", "predicted_net_edge_bps")
    if expected is None:
        return None
    realized = _first_float(row, "realized_net_edge_bps", "net_edge_bps", "markout_bps")
    resolved_qty = _first_float(row, "resolved_fill_qty", "qty", "filled_qty", "quantity")
    requested_qty = _first_float(row, "requested_qty", "quantity", "qty")
    terminal_nonfill = bool(row.get("pending_terminal_nonfill")) or (
        resolved_qty is not None and resolved_qty <= 0.0 and str(row.get("pending_resolved") or "").lower() == "true"
    )
    if realized is None and terminal_nonfill:
        realized = 0.0
    if realized is None:
        return None
    spread = _first_float(row, "spread_bps", "decision_spread_bps", "spread_paid_bps")
    quote_age = _first_float(row, "quote_age_ms", "quote_fresh_ms", "decision_quote_age_ms")
    fill_ratio = None
    if requested_qty is not None and requested_qty > 0.0 and resolved_qty is not None:
        fill_ratio = max(0.0, min(float(resolved_qty) / float(requested_qty), 1.0))
    normalized = {
        "symbol": _symbol(row),
        "side": _side(row),
        "session": _token(row, "session_bucket", "session_regime", "session", "venue_session"),
        "spread_bucket": _spread_bucket(spread),
        "quote_age_bucket": _quote_age_bucket(quote_age),
        "order_type": _token(row, "order_type", "submitted_order_type"),
        "regime": _token(row, "regime", "market_regime", "regime_profile", "volatility_regime"),
        "expected_edge_bucket": _edge_bucket(expected),
        "expected_net_edge_bps": float(expected),
        "realized_net_edge_bps": float(realized),
        "slippage_bps": _first_float(row, "slippage_bps", "slippage_drag_bps"),
        "fill_ratio": fill_ratio,
        "terminal_nonfill": terminal_nonfill,
    }
    normalized["classification"] = (
        "nonfill"
        if terminal_nonfill
        else classify_execution_capture({**dict(row), **normalized})
    )
    return normalized


def _bucket_key(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    return "|".join(str(row.get(field) or "unknown") for field in fields)


def _bucket_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = [float(row["expected_net_edge_bps"]) for row in rows]
    realized = [float(row["realized_net_edge_bps"]) for row in rows]
    classifications = Counter(str(row.get("classification") or "unknown") for row in rows)
    adverse = sum(classifications.get(key, 0) for key in ("adverse_selection", "weak_execution_capture"))
    nonfills = int(classifications.get("nonfill", 0))
    capture = _capture_ratio(expected, realized)
    return {
        "count": len(rows),
        "mean_expected_net_edge_bps": _mean(expected),
        "mean_realized_net_edge_bps": _mean(realized),
        "capture_ratio": capture,
        "win_rate": float(sum(1 for value in realized if value > 0.0) / len(realized)) if realized else None,
        "profit_factor": _profit_factor(realized),
        "classification_counts": dict(sorted(classifications.items())),
        "adverse_selection_count": int(adverse),
        "nonfill_count": int(nonfills),
    }


def _recommendation(summary: Mapping[str, Any], *, min_bucket_samples: int, min_capture_ratio: float) -> dict[str, Any]:
    count = int(summary.get("count") or 0)
    capture = _safe_float(summary.get("capture_ratio"))
    mean_realized = _safe_float(summary.get("mean_realized_net_edge_bps"))
    adverse_count = int(summary.get("adverse_selection_count") or 0)
    nonfill_count = int(summary.get("nonfill_count") or 0)
    reasons: list[str] = []
    action = "observe"
    order_behavior = "collect_more_samples"
    qty_scale = 1.0
    required_edge_add_bps = 0.0
    expected_edge_multiplier: float | None = None
    if count < int(min_bucket_samples):
        reasons.append("insufficient_bucket_samples")
        expected_edge_multiplier = None
    else:
        if capture is None:
            reasons.append("missing_capture_ratio")
            expected_edge_multiplier = None
        else:
            expected_edge_multiplier = max(0.0, min(float(capture), 1.0))
        if capture is not None and capture <= 0.0:
            action = "shadow"
            order_behavior = "skip_or_shadow_bucket"
            qty_scale = 0.0
            required_edge_add_bps = 5.0
            reasons.append("negative_or_zero_capture_ratio")
        elif capture is not None and capture < float(min_capture_ratio):
            action = "downscale"
            order_behavior = "wait_and_submit_or_passive_limit"
            qty_scale = max(0.05, min(float(capture), 0.5))
            required_edge_add_bps = max(1.0, (float(min_capture_ratio) - float(capture)) * 10.0)
            reasons.append("weak_capture_ratio")
        elif mean_realized is not None and mean_realized < 0.0:
            action = "downscale"
            order_behavior = "skip_deteriorating_quotes"
            qty_scale = 0.35
            required_edge_add_bps = 2.5
            reasons.append("negative_realized_edge")
        else:
            action = "allow"
            order_behavior = "continue_current_execution"
            reasons.append("bucket_capture_acceptable")
        if adverse_count >= max(2, int(count * 0.35)) and action == "allow":
            action = "downscale"
            order_behavior = "wait_and_submit_or_passive_limit"
            qty_scale = 0.5
            required_edge_add_bps = max(required_edge_add_bps, 1.5)
            reasons.append("adverse_selection_pressure")
        if nonfill_count >= max(2, int(count * 0.35)):
            order_behavior = "less_marketable_limit_or_skip"
            required_edge_add_bps = max(required_edge_add_bps, 1.0)
            reasons.append("nonfill_pressure")
    return {
        "action": action,
        "order_behavior": order_behavior,
        "qty_scale": float(qty_scale),
        "required_edge_add_bps": float(required_edge_add_bps),
        "expected_edge_multiplier": expected_edge_multiplier,
        "reasons": reasons,
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
    }


def _grouped(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], *, min_bucket_samples: int, min_capture_ratio: float) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_bucket_key(row, fields)].append(row)
    output: dict[str, dict[str, Any]] = {}
    for key, bucket_rows in sorted(groups.items()):
        summary = _bucket_summary(bucket_rows)
        output[key] = {
            "bucket_key": key,
            "dimensions": dict(zip(fields, key.split("|"), strict=False)),
            **summary,
            "recommendation": _recommendation(
                summary,
                min_bucket_samples=min_bucket_samples,
                min_capture_ratio=min_capture_ratio,
            ),
        }
    return output


def _worst_buckets(grouped: Mapping[str, Mapping[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    rows = []
    for key, summary in grouped.items():
        count = int(summary.get("count") or 0)
        expected = _safe_float(summary.get("mean_expected_net_edge_bps")) or 0.0
        realized = _safe_float(summary.get("mean_realized_net_edge_bps")) or 0.0
        capture = _safe_float(summary.get("capture_ratio"))
        score = (expected - realized) * max(1, count)
        if capture is not None and capture < 0.0:
            score += abs(capture) * 10.0
        rows.append(
            {
                "bucket_key": key,
                "count": count,
                "score": float(score),
                "mean_expected_net_edge_bps": summary.get("mean_expected_net_edge_bps"),
                "mean_realized_net_edge_bps": summary.get("mean_realized_net_edge_bps"),
                "capture_ratio": summary.get("capture_ratio"),
                "recommendation": summary.get("recommendation"),
            }
        )
    return sorted(rows, key=lambda row: float(row["score"]), reverse=True)[:limit]


def build_execution_capture_improvement_report(
    *,
    report_date: str,
    fills: Sequence[Mapping[str, Any]],
    tca_rows: Sequence[Mapping[str, Any]] = (),
    min_bucket_samples: int = 3,
    min_capture_ratio: float = 0.35,
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in [*fills, *tca_rows]:
        normalized_row = _normalize_row(row)
        if normalized_row is None:
            continue
        identity = (
            str(row.get("client_order_id") or ""),
            str(row.get("symbol") or ""),
            str(row.get("ts") or row.get("timestamp") or row.get("pending_resolved_ts") or ""),
        )
        # Only de-duplicate broker/order keyed rows. Fixture, legacy, or TCA rows can
        # legitimately share symbol/timestamp without representing the same event.
        if identity[0]:
            if identity in seen:
                continue
            seen.add(identity)
        normalized.append(normalized_row)

    expected = [float(row["expected_net_edge_bps"]) for row in normalized]
    realized = [float(row["realized_net_edge_bps"]) for row in normalized]
    overall = _bucket_summary(normalized) if normalized else {
        "count": 0,
        "mean_expected_net_edge_bps": None,
        "mean_realized_net_edge_bps": None,
        "capture_ratio": None,
        "win_rate": None,
        "profit_factor": None,
        "classification_counts": {},
        "adverse_selection_count": 0,
        "nonfill_count": 0,
    }
    by_symbol = _grouped(normalized, ("symbol",), min_bucket_samples=min_bucket_samples, min_capture_ratio=min_capture_ratio)
    by_symbol_side_session = _grouped(
        normalized,
        ("symbol", "side", "session"),
        min_bucket_samples=min_bucket_samples,
        min_capture_ratio=min_capture_ratio,
    )
    by_order_bucket = _grouped(
        normalized,
        ("symbol", "side", "session", "order_type", "spread_bucket", "quote_age_bucket", "regime"),
        min_bucket_samples=min_bucket_samples,
        min_capture_ratio=min_capture_ratio,
    )
    status = "insufficient_samples"
    if len(normalized) >= int(min_bucket_samples):
        capture = _safe_float(overall.get("capture_ratio"))
        status = "needs_review" if capture is None or capture < float(min_capture_ratio) else "ready"
    recommended = (
        "collect_more_execution_capture_samples"
        if status == "insufficient_samples"
        else "apply_conservative_execution_capture_haircuts"
        if status == "needs_review"
        else "continue_sampling_and_monitor_capture"
    )
    symbol_haircuts = {
        symbol: {
            "expected_edge_multiplier": summary["recommendation"].get("expected_edge_multiplier"),
            "required_edge_add_bps": summary["recommendation"].get("required_edge_add_bps"),
            "qty_scale": summary["recommendation"].get("qty_scale"),
            "action": summary["recommendation"].get("action"),
            "order_behavior": summary["recommendation"].get("order_behavior"),
            "reasons": summary["recommendation"].get("reasons"),
        }
        for symbol, summary in by_symbol.items()
    }
    return {
        "schema_version": "1.0.0",
        "artifact_type": "execution_capture_improvement_report",
        "report_date": report_date,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "recommended_next_action": recommended,
        "sample_gate": {
            "min_bucket_samples": int(min_bucket_samples),
            "samples": len(normalized),
            "sufficient": len(normalized) >= int(min_bucket_samples),
        },
        "summary": {
            "samples": len(normalized),
            "mean_expected_net_edge_bps": _mean(expected),
            "mean_realized_net_edge_bps": _mean(realized),
            "capture_ratio": _capture_ratio(expected, realized),
            "win_rate": overall.get("win_rate"),
            "profit_factor": overall.get("profit_factor"),
            "classification_counts": overall.get("classification_counts"),
            "adverse_selection_count": overall.get("adverse_selection_count"),
            "nonfill_count": overall.get("nonfill_count"),
        },
        "bad_buckets": {
            "by_symbol": _worst_buckets(by_symbol),
            "by_symbol_side_session": _worst_buckets(by_symbol_side_session),
            "by_order_context": _worst_buckets(by_order_bucket),
        },
        "bucket_diagnostics": {
            "by_symbol": by_symbol,
            "by_symbol_side_session": by_symbol_side_session,
            "by_order_context": by_order_bucket,
        },
        "edge_haircuts": {
            "global": _recommendation(
                overall,
                min_bucket_samples=min_bucket_samples,
                min_capture_ratio=min_capture_ratio,
            ),
            "by_symbol": symbol_haircuts,
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
        },
        "execution_behavior_recommendations": {
            "bad_bucket_default": "wait_and_submit_or_passive_limit",
            "skip_when": ["negative_or_zero_capture_ratio", "stale_quote", "spread_wide"],
            "cancel_faster_when": ["quote_deteriorates", "nonfill_pressure"],
            "paper_only": True,
            "manual_review_required_before_live": True,
        },
        "training_labels": {
            "recommended_targets": [
                "post_submit_adverse_movement",
                "fill_quality",
                "realized_net_edge_after_cost",
                "symbol_session_order_type_capture",
                "wait_vs_submit_counterfactual",
            ],
            "source_rows": len(normalized),
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
        },
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
    }


def _default_paths(report_date: str) -> tuple[Path, Path]:
    root = resolve_runtime_artifact_path(
        "runtime/reports",
        default_relative="runtime/reports",
        for_write=True,
    )
    compact = report_date.replace("-", "")
    return root / f"execution_capture_improvement_{compact}.json", root / "execution_capture_improvement_latest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--fills-jsonl", type=Path, default=None)
    parser.add_argument("--tca-jsonl", type=Path, default=None)
    parser.add_argument("--min-bucket-samples", type=int, default=3)
    parser.add_argument("--min-capture-ratio", type=float, default=0.35)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)
    output_json, latest_json = _default_paths(str(args.report_date))
    output_json = args.output_json or output_json
    latest_json = args.latest_json or latest_json
    report = build_execution_capture_improvement_report(
        report_date=str(args.report_date),
        fills=_read_jsonl(args.fills_jsonl, report_date=str(args.report_date)),
        tca_rows=_read_jsonl(args.tca_jsonl, report_date=str(args.report_date)),
        min_bucket_samples=int(args.min_bucket_samples),
        min_capture_ratio=float(args.min_capture_ratio),
    )
    for path in (output_json, latest_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_json), "status": report["status"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
