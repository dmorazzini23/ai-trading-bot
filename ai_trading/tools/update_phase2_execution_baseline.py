from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from ai_trading.analytics.execution_report import (
    build_phase2_execution_edge_summary,
    load_execution_records,
)
from ai_trading.config.management import get_env
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _resolve_path(value: str | None, *, env_key: str, default_relative: str) -> Path:
    raw = str(value or get_env(env_key, default_relative, cast=str) or default_relative)
    return Path(
        resolve_runtime_artifact_path(
            raw,
            default_relative=default_relative,
            for_write=False,
        )
    )


def _resolve_output_path(value: str | None) -> Path:
    configured = str(
        value
        or get_env(
            "AI_TRADING_ROADMAP_PHASE2_BASELINE_PATH",
            "runtime/phase2_execution_baseline.json",
            cast=str,
        )
        or "runtime/phase2_execution_baseline.json"
    )
    return Path(
        resolve_runtime_artifact_path(
            configured,
            default_relative="runtime/phase2_execution_baseline.json",
            for_write=True,
        )
    )


def update_phase2_execution_baseline(
    *,
    tca_path: str,
    output_path: str,
    window_days: int,
) -> dict[str, Any]:
    records = load_execution_records(tca_path)
    summary = build_phase2_execution_edge_summary(records, window_days=max(1, int(window_days)))
    metrics_raw = summary.get("metrics")
    metrics = dict(metrics_raw) if isinstance(metrics_raw, dict) else {}
    calibration_raw = summary.get("calibration")
    calibration = dict(calibration_raw) if isinstance(calibration_raw, dict) else {}
    payload: dict[str, Any] = {
        "generated_at": summary.get("evaluated_at_utc"),
        "window_days": int(summary.get("window_days", max(1, int(window_days)))),
        "records_in_window": int(summary.get("records_in_window", 0)),
        "source": {"tca_path": str(tca_path)},
        "baselines": {
            "slippage_median_abs_bps": metrics.get("slippage_median_abs_bps"),
            "target_limit_fill_rate": metrics.get("target_limit_fill_rate"),
            "stale_pending_count": metrics.get("stale_pending_count"),
        },
        "calibration": calibration,
    }
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return {
        "ok": True,
        "output_path": str(destination),
        "window_days": payload["window_days"],
        "records_in_window": payload["records_in_window"],
        "baselines": payload["baselines"],
        "calibration": payload["calibration"],
    }


def _format_env_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _format_calibration_env_suggestions(calibration: Mapping[str, Any]) -> str:
    thresholds_raw = calibration.get("recommended_routing_thresholds")
    if not isinstance(thresholds_raw, Mapping):
        return ""
    lines = [
        "# Phase 2 execution-edge routing threshold suggestions",
        "# Applies calibration thresholds only; enable routing separately after paper validation.",
    ]
    for key in sorted(str(name) for name in thresholds_raw):
        if not key.startswith("AI_TRADING_PHASE2_EXECUTION_EDGE_"):
            continue
        value = thresholds_raw.get(key)
        if value is None:
            continue
        lines.append(f"export {key}={_format_env_value(value)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute and persist Phase 2 execution-edge baselines from TCA records.",
    )
    parser.add_argument("--tca-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument(
        "--print-env-suggestions",
        action="store_true",
        help=(
            "After the JSON summary, print shell export suggestions for calibrated "
            "routing thresholds. Routing remains disabled in the suggestions."
        ),
    )
    args = parser.parse_args(argv)

    tca_path = _resolve_path(
        args.tca_path,
        env_key="AI_TRADING_TCA_PATH",
        default_relative="runtime/tca_records.jsonl",
    )
    output_path = _resolve_output_path(args.output_path)
    summary = update_phase2_execution_baseline(
        tca_path=str(tca_path),
        output_path=str(output_path),
        window_days=max(1, int(args.window_days)),
    )
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    if args.print_env_suggestions:
        suggestions = _format_calibration_env_suggestions(summary.get("calibration", {}))
        if suggestions:
            sys.stdout.write(suggestions + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
