#!/usr/bin/env python3
"""Monitor execution-quality metrics and after-hours training gate blockers."""

from __future__ import annotations

import argparse
import json
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
        return float(value)
    except (TypeError, ValueError):
        return None


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

    model_info = report.get("model") if isinstance(report, dict) else {}
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

    by_symbol: Counter[str] = Counter()
    by_code: Counter[str] = Counter()
    by_phase: Counter[str] = Counter()
    by_trace: Counter[str] = Counter()
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

                event = row.get("event")
                reason = row.get("reason")
                if event == "submit_failed" and reason == "submit_no_result":
                    submit_no_result += 1
                    symbol = str(row.get("symbol") or "UNKNOWN")
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

                if event == "submit_outcome":
                    realized = (
                        _to_float(row.get("sum_realized_bps"))
                        or _to_float(row.get("realized_bps"))
                        or _to_float(row.get("realized_net_edge_bps"))
                    )
                    if realized is not None:
                        realized_sum += realized
                        realized_samples += 1

    share = float(passive_low_skips) / float(skipped_total) if skipped_total else 0.0
    return {
        "now": now,
        "window_minutes": window_minutes,
        "submit_no_result": submit_no_result,
        "passive_low_skip_share": share,
        "passive_low_skips": passive_low_skips,
        "skipped_total": skipped_total,
        "rolling_sum_realized_bps": realized_sum,
        "realized_samples": realized_samples,
        "top_symbols": by_symbol.most_common(top_n),
        "top_error_codes": by_code.most_common(top_n),
        "top_phases": by_phase.most_common(top_n),
        "unique_trace_ids": len(by_trace),
        "duplicate_trace_ids": sum(1 for _, count in by_trace.items() if count > 1),
        "missing_phase": missing_phase,
        "missing_trace": missing_trace,
    }


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
    args = parser.parse_args()

    events_path = Path(args.events_path)
    state_path = Path(args.state_path)
    reports_dir = Path(args.reports_dir)
    end_at = datetime.now(UTC) + timedelta(minutes=max(1, args.duration_minutes))

    while True:
        metrics = _window_metrics(
            events_path=events_path,
            window_minutes=max(1, args.window_minutes),
            top_n=max(1, args.top_n),
        )
        training = _training_summary(state_path=state_path, reports_dir=reports_dir)
        _print_snapshot(metrics=metrics, training=training)

        if args.once or datetime.now(UTC) >= end_at:
            break
        time.sleep(max(1, args.interval_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
