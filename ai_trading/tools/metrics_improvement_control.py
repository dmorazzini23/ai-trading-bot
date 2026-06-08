"""Build conservative controls from recent trading-quality metrics.

The artifact produced here is intentionally one-way: it may recommend blocking,
shadowing, downscaling, or stricter edge requirements, but it never grants
promotion, live-money, provider, or symbol authority.
"""

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
    return float(parsed) if math.isfinite(parsed) else None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return default
    return parsed


def _status(payload: Mapping[str, Any]) -> str:
    raw = payload.get("status")
    if isinstance(raw, Mapping):
        return str(raw.get("status") or "missing")
    return str(raw or "missing")


def _latest_daily_reports(root: Path, *, report_date: str, lookback_sessions: int) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    by_date: dict[str, tuple[str, Path]] = {}
    for path in root.glob("*_daily/trading_day_report.json"):
        payload = _read_json(path)
        date = str(payload.get("report_date") or "").strip()
        generated = str(payload.get("generated_at") or "")
        if not date or date > report_date:
            continue
        previous = by_date.get(date)
        if previous is None or generated > previous[0]:
            by_date[date] = (generated, path)
    selected = [path for _generated, path in sorted(by_date.values())[-max(1, lookback_sessions) :]]
    return [_read_json(path) for path in selected]


def _accumulate_from_execution_capture(
    stats: dict[str, dict[str, float]],
    payload: Mapping[str, Any],
) -> None:
    by_symbol = payload.get("by_symbol")
    if not isinstance(by_symbol, Mapping):
        return
    for symbol_raw, row_raw in by_symbol.items():
        if not isinstance(row_raw, Mapping):
            continue
        symbol = str(symbol_raw).strip().upper()
        if not symbol:
            continue
        count = max(0, _safe_int(row_raw.get("count"), 0))
        expected = _safe_float(row_raw.get("mean_expected_net_edge_bps"))
        realized = _safe_float(row_raw.get("mean_realized_net_edge_bps"))
        if count <= 0:
            continue
        row = stats[symbol]
        row["samples"] += float(count)
        if expected is not None:
            row["expected_sum_bps"] += float(expected) * float(count)
        if realized is not None:
            row["realized_sum_bps"] += float(realized) * float(count)


def _accumulate_from_surveillance(
    stats: dict[str, dict[str, float]],
    payload: Mapping[str, Any],
) -> None:
    findings = payload.get("findings")
    if not isinstance(findings, list):
        return
    for item in findings:
        if not isinstance(item, Mapping):
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        category = str(item.get("category") or "").strip().lower()
        if not symbol:
            continue
        row = stats[symbol]
        if category == "adverse_selection":
            row["adverse_findings"] += 1.0
        elif category == "reject":
            row["reject_findings"] += 1.0


def _reports_from_inputs(
    *,
    report_date: str,
    daily_report_root: Path | None,
    lookback_sessions: int,
    trading_day_json: Path | None,
) -> list[dict[str, Any]]:
    reports = []
    if daily_report_root is not None:
        reports.extend(
            _latest_daily_reports(
                daily_report_root,
                report_date=report_date,
                lookback_sessions=lookback_sessions,
            )
        )
    direct = _read_json(trading_day_json)
    if direct and str(direct.get("report_date") or "") not in {
        str(row.get("report_date") or "") for row in reports
    }:
        reports.append(direct)
    return reports[-max(1, lookback_sessions) :]


def build_metrics_improvement_control(
    *,
    report_date: str,
    reports: Sequence[Mapping[str, Any]],
    expected_edge_calibration: Mapping[str, Any] | None = None,
    execution_capture: Mapping[str, Any] | None = None,
    post_trade_surveillance: Mapping[str, Any] | None = None,
    live_cost_model: Mapping[str, Any] | None = None,
    min_symbol_samples: int = 5,
    min_capture_ratio: float = 0.25,
    hard_capture_ratio: float = 0.0,
    min_realized_edge_bps: float = 0.0,
    max_adverse_findings: int = 3,
    max_reject_findings: int = 3,
    downscale_qty_scale: float = 0.5,
    weak_bucket_qty_scale: float = 0.35,
    exploration_qty_scale: float = 0.5,
    base_min_edge_bps: float = 2.0,
    cost_p90_multiplier: float = 1.0,
    weak_bucket_edge_add_bps: float = 1.5,
    unknown_quote_metadata_edge_add_bps: float = 1.0,
    exploration_window_minutes: int = 390,
    max_exploration_orders: int = 3,
    max_exploration_orders_per_symbol: int = 1,
    cooldown_seconds: int = 1800,
) -> dict[str, Any]:
    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "samples": 0.0,
            "expected_sum_bps": 0.0,
            "realized_sum_bps": 0.0,
            "adverse_findings": 0.0,
            "reject_findings": 0.0,
        }
    )
    sessions: list[str] = []
    for report in reports:
        date = str(report.get("report_date") or "").strip()
        if date:
            sessions.append(date)
        _accumulate_from_execution_capture(
            stats,
            report.get("execution_capture") if isinstance(report.get("execution_capture"), Mapping) else {},
        )
        _accumulate_from_surveillance(
            stats,
            report.get("post_trade_surveillance")
            if isinstance(report.get("post_trade_surveillance"), Mapping)
            else {},
        )
    has_current_report = any(str(report.get("report_date") or "") == report_date for report in reports)
    if not has_current_report:
        _accumulate_from_execution_capture(stats, execution_capture or {})
        _accumulate_from_surveillance(stats, post_trade_surveillance or {})

    live_status = (
        live_cost_model.get("status")
        if isinstance(live_cost_model, Mapping)
        else {}
    )
    live_observed = (
        live_cost_model.get("observed")
        if isinstance(live_cost_model, Mapping)
        else {}
    )
    live_p90 = _safe_float(live_observed.get("p90_total_cost_bps") if isinstance(live_observed, Mapping) else None)
    if live_p90 is None:
        live_p90 = 0.0
    min_symbol_samples = max(1, int(min_symbol_samples))
    downscale_qty_scale = max(0.05, min(float(downscale_qty_scale), 1.0))
    weak_bucket_qty_scale = max(0.0, min(float(weak_bucket_qty_scale), 1.0))
    exploration_qty_scale = max(0.05, min(float(exploration_qty_scale), 1.0))
    base_required_edge = max(0.0, float(base_min_edge_bps)) + (
        max(0.0, float(live_p90)) * max(0.0, float(cost_p90_multiplier))
    )

    by_symbol: dict[str, dict[str, Any]] = {}
    totals = {
        "samples": 0,
        "expected_sum_bps": 0.0,
        "realized_sum_bps": 0.0,
        "adverse_findings": 0,
        "reject_findings": 0,
        "shadowed_or_blocked_symbols": 0,
        "downscaled_symbols": 0,
        "exploration_symbols": 0,
    }
    for symbol in sorted(stats):
        row = stats[symbol]
        samples = int(max(0.0, row["samples"]))
        expected_sum = float(row["expected_sum_bps"])
        realized_sum = float(row["realized_sum_bps"])
        adverse_count = int(max(0.0, row["adverse_findings"]))
        reject_count = int(max(0.0, row["reject_findings"]))
        mean_expected = expected_sum / float(samples) if samples > 0 else None
        mean_realized = realized_sum / float(samples) if samples > 0 else None
        capture_ratio = (
            realized_sum / expected_sum
            if expected_sum > 0.0 and samples > 0
            else None
        )
        reasons: list[str] = []
        action = "allow"
        qty_scale = 1.0
        required_edge = float(base_required_edge)
        if samples < min_symbol_samples:
            action = "explore"
            qty_scale = float(exploration_qty_scale)
            reasons.append("insufficient_symbol_samples")
        else:
            adverse_pressure = adverse_count >= int(max_adverse_findings)
            weak_capture = capture_ratio is not None and float(capture_ratio) < float(min_capture_ratio)
            weak_realized = mean_realized is not None and float(mean_realized) < float(min_realized_edge_bps)
            if adverse_pressure and (weak_realized or (capture_ratio is not None and float(capture_ratio) <= float(hard_capture_ratio))):
                action = "cooldown"
                qty_scale = 0.0
                reasons.append("adverse_selection_cooldown")
            elif capture_ratio is not None and float(capture_ratio) <= float(hard_capture_ratio):
                action = "shadow"
                qty_scale = 0.0
                reasons.append("capture_ratio_hard_breach")
            elif weak_realized:
                action = "downscale"
                qty_scale = float(weak_bucket_qty_scale)
                required_edge += float(weak_bucket_edge_add_bps)
                reasons.append("realized_edge_below_floor")
            elif weak_capture:
                action = "downscale"
                qty_scale = float(downscale_qty_scale)
                required_edge += float(weak_bucket_edge_add_bps)
                reasons.append("capture_ratio_below_floor")
            if adverse_pressure and action == "allow":
                action = "downscale"
                qty_scale = min(float(qty_scale), float(downscale_qty_scale))
                required_edge += float(weak_bucket_edge_add_bps)
                reasons.append("adverse_selection_pressure")
            if reject_count >= int(max_reject_findings) and action == "allow":
                action = "downscale"
                qty_scale = min(float(qty_scale), float(downscale_qty_scale))
                required_edge += float(weak_bucket_edge_add_bps)
                reasons.append("reject_pressure")
        if action in {"shadow", "cooldown"}:
            totals["shadowed_or_blocked_symbols"] += 1
        elif action == "downscale":
            totals["downscaled_symbols"] += 1
        elif action == "explore":
            totals["exploration_symbols"] += 1
        totals["samples"] += samples
        totals["expected_sum_bps"] += expected_sum
        totals["realized_sum_bps"] += realized_sum
        totals["adverse_findings"] += adverse_count
        totals["reject_findings"] += reject_count
        by_symbol[symbol] = {
            "symbol": symbol,
            "samples": samples,
            "mean_expected_edge_bps": mean_expected,
            "mean_realized_edge_bps": mean_realized,
            "capture_ratio": capture_ratio,
            "adverse_findings": adverse_count,
            "reject_findings": reject_count,
            "action": action,
            "qty_scale": float(qty_scale),
            "required_edge_bps": float(required_edge),
            "unknown_quote_metadata_edge_add_bps": float(unknown_quote_metadata_edge_add_bps),
            "cooldown_seconds": int(cooldown_seconds) if action == "cooldown" else 0,
            "reasons": reasons or ["metrics_ok"],
        }

    total_samples = int(totals["samples"])
    global_capture = (
        float(totals["realized_sum_bps"]) / float(totals["expected_sum_bps"])
        if float(totals["expected_sum_bps"]) > 0.0
        else None
    )
    if not by_symbol:
        status = "insufficient_samples"
        recommended = "collect_more_diagnostic_paper_samples"
    elif totals["shadowed_or_blocked_symbols"] or totals["downscaled_symbols"]:
        status = "active"
        recommended = "apply_conservative_next_session_metric_controls"
    else:
        status = "observe"
        recommended = "continue_sampling_with_current_controls"
    return {
        "schema_version": "1.0.0",
        "artifact_type": "metrics_improvement_control",
        "report_date": report_date,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "recommended_next_action": recommended,
        "summary": {
            "sessions": sorted(set(sessions)),
            "session_count": len(set(sessions)),
            "symbol_count": len(by_symbol),
            "samples": total_samples,
            "global_capture_ratio": global_capture,
            "mean_expected_edge_bps": (
                float(totals["expected_sum_bps"]) / float(total_samples)
                if total_samples > 0
                else None
            ),
            "mean_realized_edge_bps": (
                float(totals["realized_sum_bps"]) / float(total_samples)
                if total_samples > 0
                else None
            ),
            "adverse_findings": int(totals["adverse_findings"]),
            "reject_findings": int(totals["reject_findings"]),
            "shadowed_or_blocked_symbols": int(totals["shadowed_or_blocked_symbols"]),
            "downscaled_symbols": int(totals["downscaled_symbols"]),
            "exploration_symbols": int(totals["exploration_symbols"]),
        },
        "control_policy": {
            "min_symbol_samples": int(min_symbol_samples),
            "min_capture_ratio": float(min_capture_ratio),
            "hard_capture_ratio": float(hard_capture_ratio),
            "min_realized_edge_bps": float(min_realized_edge_bps),
            "max_adverse_findings": int(max_adverse_findings),
            "max_reject_findings": int(max_reject_findings),
            "base_min_edge_bps": float(base_min_edge_bps),
            "live_cost_p90_bps": float(live_p90),
            "live_cost_status": _status(live_status if isinstance(live_status, Mapping) else {}),
            "cost_p90_multiplier": float(cost_p90_multiplier),
            "weak_bucket_edge_add_bps": float(weak_bucket_edge_add_bps),
            "unknown_quote_metadata_edge_add_bps": float(unknown_quote_metadata_edge_add_bps),
            "authority_increase_allowed": False,
        },
        "exploration_budget": {
            "window_minutes": int(max(1, exploration_window_minutes)),
            "max_orders_per_window": int(max(0, max_exploration_orders)),
            "max_orders_per_symbol_per_window": int(max(0, max_exploration_orders_per_symbol)),
            "qty_scale": float(exploration_qty_scale),
        },
        "by_symbol": by_symbol,
        "inputs": {
            "expected_edge_calibration_status": _status(expected_edge_calibration or {}),
            "execution_capture_status": _status(execution_capture or {}),
            "post_trade_surveillance_status": _status(post_trade_surveillance or {}),
            "live_cost_model_status": _status(live_status if isinstance(live_status, Mapping) else {}),
            "trading_day_reports": len(reports),
        },
        "runtime_safety_control": True,
        "authority_increase_allowed": False,
        "promotion_authority": False,
        "live_money_authority": False,
        "provider_authority": False,
    }


def _default_paths(report_date: str) -> tuple[Path, Path]:
    root = resolve_runtime_artifact_path(
        "runtime/reports",
        default_relative="runtime/reports",
        for_write=True,
    )
    compact = report_date.replace("-", "")
    return root / f"metrics_improvement_control_{compact}.json", root / "metrics_improvement_control_latest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--daily-report-root", type=Path, default=None)
    parser.add_argument("--lookback-sessions", type=int, default=5)
    parser.add_argument("--trading-day-json", type=Path, default=None)
    parser.add_argument("--expected-edge-calibration-json", type=Path, default=None)
    parser.add_argument("--execution-capture-json", type=Path, default=None)
    parser.add_argument("--post-trade-surveillance-json", type=Path, default=None)
    parser.add_argument("--live-cost-model-json", type=Path, default=None)
    parser.add_argument("--min-symbol-samples", type=int, default=5)
    parser.add_argument("--min-capture-ratio", type=float, default=0.25)
    parser.add_argument("--hard-capture-ratio", type=float, default=0.0)
    parser.add_argument("--min-realized-edge-bps", type=float, default=0.0)
    parser.add_argument("--max-adverse-findings", type=int, default=3)
    parser.add_argument("--max-reject-findings", type=int, default=3)
    parser.add_argument("--base-min-edge-bps", type=float, default=2.0)
    parser.add_argument("--cost-p90-multiplier", type=float, default=1.0)
    parser.add_argument("--exploration-qty-scale", type=float, default=0.5)
    parser.add_argument("--exploration-window-minutes", type=int, default=390)
    parser.add_argument("--max-exploration-orders", type=int, default=3)
    parser.add_argument("--max-exploration-orders-per-symbol", type=int, default=1)
    parser.add_argument("--unknown-quote-metadata-edge-add-bps", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)
    output_json, latest_json = _default_paths(str(args.report_date))
    output_json = args.output_json or output_json
    latest_json = args.latest_json or latest_json
    report_root = args.daily_report_root
    if report_root is None:
        report_root = resolve_runtime_artifact_path(
            "runtime/research_reports/daily",
            default_relative="runtime/research_reports/daily",
        )
    report = build_metrics_improvement_control(
        report_date=str(args.report_date),
        reports=_reports_from_inputs(
            report_date=str(args.report_date),
            daily_report_root=report_root,
            lookback_sessions=int(args.lookback_sessions),
            trading_day_json=args.trading_day_json,
        ),
        expected_edge_calibration=_read_json(args.expected_edge_calibration_json),
        execution_capture=_read_json(args.execution_capture_json),
        post_trade_surveillance=_read_json(args.post_trade_surveillance_json),
        live_cost_model=_read_json(args.live_cost_model_json),
        min_symbol_samples=int(args.min_symbol_samples),
        min_capture_ratio=float(args.min_capture_ratio),
        hard_capture_ratio=float(args.hard_capture_ratio),
        min_realized_edge_bps=float(args.min_realized_edge_bps),
        max_adverse_findings=int(args.max_adverse_findings),
        max_reject_findings=int(args.max_reject_findings),
        base_min_edge_bps=float(args.base_min_edge_bps),
        cost_p90_multiplier=float(args.cost_p90_multiplier),
        exploration_qty_scale=float(args.exploration_qty_scale),
        exploration_window_minutes=int(args.exploration_window_minutes),
        max_exploration_orders=int(args.max_exploration_orders),
        max_exploration_orders_per_symbol=int(args.max_exploration_orders_per_symbol),
        unknown_quote_metadata_edge_add_bps=float(args.unknown_quote_metadata_edge_add_bps),
    )
    for path in (output_json, latest_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_json), "status": report["status"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
