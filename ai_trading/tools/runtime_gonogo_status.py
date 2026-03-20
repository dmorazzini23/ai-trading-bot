from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from ai_trading.env import ensure_dotenv_loaded
from ai_trading.tools import runtime_performance_report as runtime_perf_report


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _as_int(value: Any) -> int | None:
    parsed = _as_float(value)
    if parsed is None:
        return None
    try:
        return int(parsed)
    except (TypeError, ValueError):
        return None


def _resolve_thresholds() -> dict[str, Any]:
    return runtime_perf_report.resolve_runtime_gonogo_thresholds()


def _resolve_paths() -> dict[str, Path | None]:
    resolved = runtime_perf_report.resolve_runtime_report_paths()
    return {
        "trade_history": resolved.get("trade_history"),
        "gate_summary": resolved.get("gate_summary"),
        "gate_log": resolved.get("gate_log"),
    }


def build_runtime_gonogo_status() -> dict[str, Any]:
    ensure_dotenv_loaded()
    paths = _resolve_paths()
    thresholds = _resolve_thresholds()
    report = runtime_perf_report.build_report(
        trade_history_path=Path(paths["trade_history"] or "runtime/trade_history.parquet"),
        gate_summary_path=Path(paths["gate_summary"] or "runtime/gate_effectiveness_summary.json"),
        gate_log_path=(Path(paths["gate_log"]) if isinstance(paths["gate_log"], Path) else None),
    )
    decision = runtime_perf_report.evaluate_go_no_go(report, thresholds=thresholds)
    return {
        "gate_passed": bool(decision.get("gate_passed")),
        "failed_checks": list(decision.get("failed_checks", [])),
        "checks": dict(decision.get("checks", {})),
        "thresholds": dict(decision.get("thresholds", {})),
        "observed": dict(decision.get("observed", {})),
        "paths": {k: (str(v) if v is not None else "") for k, v in paths.items()},
    }


def _fmt_float(value: Any) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "na"
    return f"{parsed:.4f}"


def format_status_line(payload: Mapping[str, Any]) -> str:
    failed = payload.get("failed_checks", [])
    if isinstance(failed, list) and failed:
        failed_text = ",".join(str(item) for item in failed)
    else:
        failed_text = "none"
    observed = payload.get("observed", {})
    thresholds = payload.get("thresholds", {})
    if not isinstance(observed, Mapping):
        observed = {}
    if not isinstance(thresholds, Mapping):
        thresholds = {}
    state = "PASS" if bool(payload.get("gate_passed")) else "FAIL"
    return (
        "RUNTIME_GONOGO_STATUS "
        f"state={state} "
        f"failed={failed_text} "
        f"fill_source={str(thresholds.get('trade_fill_source') or observed.get('trade_fill_source') or 'all')} "
        f"lookback_days={int(_as_int(thresholds.get('lookback_days')) or 0)} "
        f"min_used_days={int(_as_int(thresholds.get('min_used_days')) or 0)} "
        f"trade_used_days={int(_as_int(observed.get('trade_used_days')) or 0)} "
        f"gate_used_days={int(_as_int(observed.get('gate_used_days')) or 0)} "
        f"closed_trades={int(_as_int(observed.get('closed_trades')) or 0)} "
        f"profit_factor={_fmt_float(observed.get('profit_factor'))} "
        f"win_rate={_fmt_float(observed.get('win_rate'))} "
        f"net_pnl={_fmt_float(observed.get('net_pnl'))} "
        f"acceptance_rate={_fmt_float(observed.get('acceptance_rate'))}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-line runtime go/no-go status for daily operations.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full JSON payload instead of one-line status.",
    )
    args = parser.parse_args(argv)

    try:
        payload = build_runtime_gonogo_status()
    except Exception as exc:
        if args.json:
            sys.stdout.write(
                f"{json.dumps({'state': 'ERROR', 'error': str(exc)}, sort_keys=True)}\n"
            )
        else:
            sys.stdout.write(f"RUNTIME_GONOGO_STATUS state=ERROR error={exc}\n")
        return 1

    if args.json:
        sys.stdout.write(f"{json.dumps(payload, sort_keys=True)}\n")
    else:
        sys.stdout.write(f"{format_status_line(payload)}\n")
    return 0 if bool(payload.get("gate_passed")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
