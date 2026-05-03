#!/usr/bin/env python3
"""Monitor execution-quality metrics and after-hours training gate blockers."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _nested_float(row: dict[str, Any], *keys: str) -> float | None:
    context = row.get("context") if isinstance(row.get("context"), dict) else {}
    market = row.get("market") if isinstance(row.get("market"), dict) else {}
    cost = row.get("cost") if isinstance(row.get("cost"), dict) else {}
    for key in keys:
        parsed = _first_float(row.get(key), cost.get(key), market.get(key), context.get(key))
        if parsed is not None:
            return parsed
    return None


def _percentile(values: list[float], q: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    quantile = max(0.0, min(float(q), 1.0))
    if len(clean) == 1:
        return float(clean[0])
    raw = quantile * (len(clean) - 1)
    lo = int(math.floor(raw))
    hi = int(math.ceil(raw))
    if lo == hi:
        return float(clean[lo])
    return float(clean[lo] + (clean[hi] - clean[lo]) * (raw - lo))


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _iter_recent_report_paths(reports_dir: Path, *, limit: int = 64) -> list[Path]:
    if not reports_dir.exists():
        return []
    candidates = sorted(
        reports_dir.glob("after_hours_training_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[: max(1, int(limit))]


def _report_candidates_from_state(state: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for key in ("report_path", "daily_report_path"):
        raw = state.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        paths.append(Path(raw))
    return paths


def _load_report_payload(path: Path) -> tuple[dict[str, Any], datetime, float] | None:
    try:
        mtime = float(path.stat().st_mtime)
    except OSError:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    ts = _parse_ts(payload.get("ts")) or datetime.fromtimestamp(mtime, tz=UTC)
    return payload, ts, mtime


def _resolve_report_payload(
    *,
    state: dict[str, Any],
    reports_dir: Path,
) -> tuple[dict[str, Any], Path | None]:
    candidate_paths: list[Path] = []
    seen: set[Path] = set()
    for path in _report_candidates_from_state(state) + _iter_recent_report_paths(reports_dir):
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        candidate_paths.append(resolved)
    best_payload: dict[str, Any] = {}
    best_path: Path | None = None
    best_key: tuple[datetime, float] | None = None
    for path in candidate_paths:
        loaded = _load_report_payload(path)
        if loaded is None:
            continue
        payload, ts, mtime = loaded
        key = (ts, mtime)
        if best_key is None or key > best_key:
            best_key = key
            best_payload = payload
            best_path = path
    return best_payload, best_path


def _training_summary(state_path: Path, reports_dir: Path) -> dict[str, Any]:
    state = _load_json(state_path) if state_path.exists() else {}
    report, resolved_report_path = _resolve_report_payload(state=state, reports_dir=reports_dir)

    promotion = report.get("promotion") if isinstance(report, dict) else {}
    combined_gates = promotion.get("combined_gates") if isinstance(promotion, dict) else {}
    blockers = []
    if isinstance(combined_gates, dict):
        blockers = sorted(key for key, passed in combined_gates.items() if not bool(passed))

    model_info_raw = report.get("model") if isinstance(report, dict) else {}
    model_info = model_info_raw if isinstance(model_info_raw, dict) else {}
    sensitivity = report.get("sensitivity_sweep") if isinstance(report, dict) else {}
    sensitivity_summary = (
        sensitivity.get("summary") if isinstance(sensitivity, dict) else {}
    )
    promotion_confidence = (
        report.get("promotion_confidence_gate") if isinstance(report, dict) else {}
    )

    return {
        "model_name": model_info.get("name"),
        "model_id": state.get("model_id"),
        "governance_status": state.get("governance_status"),
        "promotion_gate_passed": state.get("promotion_gate_passed"),
        "promotion_confidence_gate_passed": state.get("promotion_confidence_gate_passed"),
        "promotion_confidence_reason": state.get("promotion_confidence_reason"),
        "effective_trades": (
            (state.get("promotion_confidence_observed") or {}).get("effective_trades")
        ),
        "gate_blockers": blockers,
        "sensitivity_summary": sensitivity_summary or {},
        "promotion_confidence": promotion_confidence if isinstance(promotion_confidence, dict) else {},
        "report_path": str(resolved_report_path) if resolved_report_path else None,
    }


def _window_metrics(events_path: Path, window_minutes: int, top_n: int) -> dict[str, Any]:
    now = datetime.now(UTC)
    cutoff = now - timedelta(minutes=window_minutes)

    submit_no_result = 0
    skipped_total = 0
    passive_low_skips = 0
    realized_sum = 0.0
    realized_samples = 0
    event_count = 0
    filled_count = 0
    blocked_count = 0
    derisked_count = 0
    spread_samples: list[float] = []
    quote_age_samples: list[float] = []

    by_symbol: Counter[str] = Counter()
    events_by_symbol: Counter[str] = Counter()
    spread_by_symbol: dict[str, list[float]] = {}
    quote_age_by_symbol: dict[str, list[float]] = {}
    by_code: Counter[str] = Counter()
    by_phase: Counter[str] = Counter()
    by_trace: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    missing_phase = 0
    missing_trace = 0

    if events_path.exists():
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = _parse_ts(row.get("ts") or row.get("timestamp"))
                if ts is None or ts < cutoff:
                    continue

                event_count += 1
                event = row.get("event")
                reason = row.get("reason")
                reason_token = str(reason or "unknown")
                symbol = str(row.get("symbol") or "UNKNOWN").upper()
                events_by_symbol[symbol] += 1
                reason_counts[reason_token] += 1
                spread_bps = _nested_float(
                    row,
                    "spread_bps",
                    "quoted_spread_bps",
                    "bid_ask_spread_bps",
                )
                quote_age_ms = _nested_float(
                    row,
                    "quote_age_ms",
                    "quote_staleness_ms",
                    "quote_age",
                )
                if spread_bps is not None:
                    spread_samples.append(float(spread_bps))
                    spread_by_symbol.setdefault(symbol, []).append(float(spread_bps))
                if quote_age_ms is not None:
                    quote_age_samples.append(float(quote_age_ms))
                    quote_age_by_symbol.setdefault(symbol, []).append(float(quote_age_ms))

                if event == "submit_failed" and reason == "submit_no_result":
                    submit_no_result += 1
                    phase = row.get("phase")
                    trace = row.get("trace_id") or row.get("client_order_id") or row.get("order_id")
                    code = row.get("error_code") or row.get("status_code") or row.get("error") or "none"
                    by_symbol[symbol] += 1
                    by_code[str(code)] += 1
                    if phase in (None, ""):
                        missing_phase += 1
                        by_phase["missing"] += 1
                    else:
                        by_phase[str(phase)] += 1
                    if trace in (None, ""):
                        missing_trace += 1
                        by_trace["missing"] += 1
                    else:
                        by_trace[str(trace)] += 1

                if event == "submit_skipped":
                    skipped_total += 1
                    if reason == "passive_fill_probability_low":
                        passive_low_skips += 1
                    if reason_token in {
                        "EXECUTION_QUALITY_SPREAD_BLOCK",
                        "EXECUTION_QUALITY_STALE_QUOTE_BLOCK",
                        "spread_bps_too_wide",
                        "quote_age_too_stale",
                    }:
                        blocked_count += 1

                if event == "submit_outcome":
                    status = str(row.get("status") or "").strip().lower()
                    if status in {"filled", "partially_filled"}:
                        filled_count += 1
                    realized = _first_float(
                        row.get("sum_realized_bps"),
                        row.get("realized_bps"),
                        row.get("realized_net_edge_bps"),
                    )
                    if realized is not None:
                        realized_sum += realized
                        realized_samples += 1
                action = str(row.get("action") or row.get("status") or "").strip().lower()
                context = row.get("context") if isinstance(row.get("context"), dict) else {}
                if action in {"derisk", "size_down", "throttle"}:
                    derisked_count += 1
                else:
                    quantity_after = _first_float(context.get("quantity_after"))
                    quantity_before = _first_float(context.get("quantity_before"))
                    if (
                        quantity_after is not None
                        and quantity_before is not None
                        and quantity_after < quantity_before
                    ):
                        derisked_count += 1

    share = float(passive_low_skips) / float(skipped_total) if skipped_total else 0.0
    by_symbol_rows: list[dict[str, Any]] = []
    for symbol, count in events_by_symbol.most_common(top_n):
        symbol_spreads = spread_by_symbol.get(symbol, [])
        symbol_ages = quote_age_by_symbol.get(symbol, [])
        by_symbol_rows.append(
            {
                "symbol": symbol,
                "events": int(count),
                "mean_spread_bps": _mean(symbol_spreads),
                "p90_spread_bps": _percentile(symbol_spreads, 0.90),
                "mean_quote_age_ms": _mean(symbol_ages),
                "p90_quote_age_ms": _percentile(symbol_ages, 0.90),
            }
        )
    return {
        "now": now,
        "window_minutes": window_minutes,
        "event_count": event_count,
        "filled_count": filled_count,
        "submit_no_result": submit_no_result,
        "passive_low_skip_share": share,
        "passive_low_skips": passive_low_skips,
        "skipped_total": skipped_total,
        "rolling_sum_realized_bps": realized_sum,
        "realized_samples": realized_samples,
        "mean_spread_bps": _mean(spread_samples),
        "p90_spread_bps": _percentile(spread_samples, 0.90),
        "mean_quote_age_ms": _mean(quote_age_samples),
        "p90_quote_age_ms": _percentile(quote_age_samples, 0.90),
        "blocked_count": blocked_count,
        "derisked_count": derisked_count,
        "reason_counts": dict(reason_counts.most_common(top_n)),
        "by_symbol": by_symbol_rows,
        "top_symbols": by_symbol.most_common(top_n),
        "top_error_codes": by_code.most_common(top_n),
        "top_phases": by_phase.most_common(top_n),
        "unique_trace_ids": len(by_trace),
        "duplicate_trace_ids": sum(1 for _, count in by_trace.items() if count > 1),
        "missing_phase": missing_phase,
        "missing_trace": missing_trace,
    }


def _governor_report(
    *,
    metrics: dict[str, Any],
    events_path: Path,
    output_path: Path | None,
) -> dict[str, Any]:
    submit_no_result = int(metrics.get("submit_no_result") or 0)
    blocked_count = int(metrics.get("blocked_count") or 0)
    failed_checks: list[str] = []
    if submit_no_result > 0:
        failed_checks.append("submit_no_result")
    status = "pass"
    mode = "observe"
    if submit_no_result > 0:
        status = "fail"
        mode = "block"
    elif blocked_count > 0:
        status = "degraded"
        mode = "derisk"
    report = {
        "schema_version": "1.0.0",
        "artifact_type": "execution_quality_governor_report",
        "generated_at": metrics["now"].isoformat().replace("+00:00", "Z"),
        "source": "execution_quality_events",
        "window": {
            "minutes": int(metrics["window_minutes"]),
            "event_count": int(metrics.get("event_count") or 0),
            "filled_count": int(metrics.get("filled_count") or 0),
        },
        "status": {
            "gate_passed": status != "fail",
            "status": status,
            "mode": mode,
            "failed_checks": failed_checks,
            "attention_flags": ["execution_quality_blocks_present"] if blocked_count else [],
        },
        "observed": {
            "mean_spread_bps": metrics.get("mean_spread_bps"),
            "p90_spread_bps": metrics.get("p90_spread_bps"),
            "mean_quote_age_ms": metrics.get("mean_quote_age_ms"),
            "p90_quote_age_ms": metrics.get("p90_quote_age_ms"),
            "submit_no_result": submit_no_result,
            "passive_fill_probability_low_skip_share": metrics.get("passive_low_skip_share"),
            "rolling_sum_realized_bps": metrics.get("rolling_sum_realized_bps"),
            "realized_samples": metrics.get("realized_samples"),
        },
        "actions": {
            "blocked_count": blocked_count,
            "derisked_count": int(metrics.get("derisked_count") or 0),
            "passive_low_skips": int(metrics.get("passive_low_skips") or 0),
            "reason_counts": metrics.get("reason_counts") or {},
        },
        "by_symbol": metrics.get("by_symbol") or [],
        "paths": {
            "events": str(events_path),
            "report": str(output_path) if output_path is not None else None,
        },
    }
    return report


def _print_snapshot(metrics: dict[str, Any], training: dict[str, Any]) -> None:
    window = int(metrics["window_minutes"])
    now = metrics["now"].isoformat()
    print(
        f"{now} | submit_no_result_{window}m={metrics['submit_no_result']} | "
        f"passive_fill_probability_low_skip_share_{window}m={metrics['passive_low_skip_share']:.3f} "
        f"({metrics['passive_low_skips']}/{metrics['skipped_total']}) | "
        f"rolling_sum_realized_bps_{window}m={metrics['rolling_sum_realized_bps']:.3f} "
        f"(samples={metrics['realized_samples']})",
        flush=True,
    )
    if metrics["submit_no_result"] > 0:
        print(
            "submit_no_result_breakdown | "
            f"top_symbols={metrics['top_symbols']} | "
            f"top_error_codes={metrics['top_error_codes']} | "
            f"top_phases={metrics['top_phases']} | "
            f"unique_trace_ids={metrics['unique_trace_ids']} | "
            f"duplicate_trace_ids={metrics['duplicate_trace_ids']} | "
            f"missing_phase={metrics['missing_phase']} | "
            f"missing_trace={metrics['missing_trace']}",
            flush=True,
        )
    print(
        "after_hours_status | "
        f"governance={training.get('governance_status')} | "
        f"model={training.get('model_name')} | "
        f"model_id={training.get('model_id')} | "
        f"promotion_gate_passed={training.get('promotion_gate_passed')} | "
        f"promotion_confidence_gate_passed={training.get('promotion_confidence_gate_passed')} | "
        f"promotion_confidence_reason={training.get('promotion_confidence_reason')} | "
        f"effective_trades={training.get('effective_trades')} | "
        f"gate_blockers={training.get('gate_blockers')}",
        flush=True,
    )
    sensitivity_summary = training.get("sensitivity_summary") or {}
    if sensitivity_summary:
        print(
            "after_hours_sensitivity | "
            f"min_support={sensitivity_summary.get('min_support')} | "
            f"observed_min_expectancy_bps={sensitivity_summary.get('observed_min_expectancy_bps')} | "
            f"observed_pass_ratio={sensitivity_summary.get('observed_pass_ratio')} | "
            f"valid_scenarios={sensitivity_summary.get('valid_scenarios')} | "
            f"total_scenarios={sensitivity_summary.get('total_scenarios')}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events-path",
        default="/var/lib/ai-trading-bot/runtime/execution_quality_events.jsonl",
        help="Execution quality events JSONL path.",
    )
    parser.add_argument(
        "--state-path",
        default="/var/lib/ai-trading-bot/runtime/after_hours_training_state.json",
        help="After-hours training state JSON path.",
    )
    parser.add_argument(
        "--reports-dir",
        default="/var/lib/ai-trading-bot/runtime/research_reports",
        help="After-hours reports directory.",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=30,
        help="Rolling metrics window in minutes.",
    )
    parser.add_argument(
        "--duration-minutes",
        type=int,
        default=60,
        help="How long to monitor before exit.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=30,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Top-N breakdown rows to print for submit_no_result.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one snapshot and exit.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path for execution-quality governor rollup JSON.",
    )
    args = parser.parse_args()

    events_path = Path(args.events_path)
    state_path = Path(args.state_path)
    reports_dir = Path(args.reports_dir)
    output_path = Path(args.output_json).expanduser() if str(args.output_json or "").strip() else None
    end_at = datetime.now(UTC) + timedelta(minutes=max(1, args.duration_minutes))

    while True:
        metrics = _window_metrics(
            events_path=events_path,
            window_minutes=max(1, args.window_minutes),
            top_n=max(1, args.top_n),
        )
        training = _training_summary(state_path=state_path, reports_dir=reports_dir)
        if output_path is not None:
            report = _governor_report(
                metrics=metrics,
                events_path=events_path,
                output_path=output_path,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        _print_snapshot(metrics=metrics, training=training)

        if args.once or datetime.now(UTC) >= end_at:
            break
        time.sleep(max(1, args.interval_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
