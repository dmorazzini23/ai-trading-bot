from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from ai_trading.config.management import get_env
from ai_trading.env import ensure_dotenv_loaded
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
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
    min_closed_trades = _as_int(get_env("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_CLOSED_TRADES", None, cast=int))
    if min_closed_trades is None:
        min_closed_trades = int(get_env("AI_TRADING_RUNTIME_GONOGO_MIN_CLOSED_TRADES", 20, cast=int))
    min_profit_factor = _as_float(get_env("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_PROFIT_FACTOR", None, cast=float))
    if min_profit_factor is None:
        min_profit_factor = float(get_env("AI_TRADING_RUNTIME_GONOGO_MIN_PROFIT_FACTOR", 1.1, cast=float))
    min_win_rate = _as_float(get_env("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_WIN_RATE", None, cast=float))
    if min_win_rate is None:
        min_win_rate = float(get_env("AI_TRADING_RUNTIME_GONOGO_MIN_WIN_RATE", 0.5, cast=float))
    min_net_pnl = _as_float(get_env("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_NET_PNL", None, cast=float))
    if min_net_pnl is None:
        min_net_pnl = float(get_env("AI_TRADING_RUNTIME_GONOGO_MIN_NET_PNL", 0.0, cast=float))
    min_acceptance_rate = _as_float(
        get_env("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_ACCEPTANCE_RATE", None, cast=float)
    )
    if min_acceptance_rate is None:
        min_acceptance_rate = float(get_env("AI_TRADING_RUNTIME_GONOGO_MIN_ACCEPTANCE_RATE", 0.05, cast=float))
    min_expected_net_edge_bps = _as_float(
        get_env("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_EXPECTED_NET_EDGE_BPS", None, cast=float)
    )
    if min_expected_net_edge_bps is None:
        min_expected_net_edge_bps = float(
            get_env("AI_TRADING_RUNTIME_GONOGO_MIN_EXPECTED_NET_EDGE_BPS", -50.0, cast=float)
        )
    lookback_days = _as_int(get_env("AI_TRADING_EXECUTION_RUNTIME_GONOGO_LOOKBACK_DAYS", None, cast=int))
    if lookback_days is None:
        lookback_days = int(get_env("AI_TRADING_RUNTIME_GONOGO_LOOKBACK_DAYS", 5, cast=int))
    min_used_days = _as_int(get_env("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_USED_DAYS", None, cast=int))
    if min_used_days is None:
        min_used_days = int(get_env("AI_TRADING_RUNTIME_GONOGO_MIN_USED_DAYS", 0, cast=int))
    require_pnl_available = get_env("AI_TRADING_EXECUTION_RUNTIME_GONOGO_REQUIRE_PNL_AVAILABLE", None, cast=bool)
    if require_pnl_available is None:
        require_pnl_available = bool(get_env("AI_TRADING_RUNTIME_GONOGO_REQUIRE_PNL_AVAILABLE", True, cast=bool))
    require_gate_valid = get_env("AI_TRADING_EXECUTION_RUNTIME_GONOGO_REQUIRE_GATE_VALID", None, cast=bool)
    if require_gate_valid is None:
        require_gate_valid = bool(get_env("AI_TRADING_RUNTIME_GONOGO_REQUIRE_GATE_VALID", False, cast=bool))
    trade_fill_source = str(
        (
            get_env(
                "AI_TRADING_EXECUTION_RUNTIME_GONOGO_TRADE_FILL_SOURCE",
                None,
                cast=str,
            )
            or get_env(
                "AI_TRADING_RUNTIME_GONOGO_TRADE_FILL_SOURCE",
                "all",
                cast=str,
            )
            or "all"
        )
    ).strip() or "all"
    return {
        "min_closed_trades": int(max(0, min_closed_trades)),
        "min_profit_factor": float(min_profit_factor),
        "min_win_rate": float(max(0.0, min(1.0, min_win_rate))),
        "min_net_pnl": float(min_net_pnl),
        "min_acceptance_rate": float(max(0.0, min(1.0, min_acceptance_rate))),
        "min_expected_net_edge_bps": float(min_expected_net_edge_bps),
        "min_used_days": int(max(0, min_used_days)),
        "lookback_days": int(max(0, lookback_days)),
        "trade_fill_source": trade_fill_source,
        "require_pnl_available": bool(require_pnl_available),
        "require_gate_valid": bool(require_gate_valid),
    }


def _resolve_paths() -> dict[str, Path | None]:
    default_trade_history = str(
        get_env(
            "AI_TRADING_TRADE_HISTORY_PATH",
            "runtime/trade_history.parquet",
            cast=str,
        )
        or ""
    ).strip() or "runtime/trade_history.parquet"
    trade_history_configured = str(
        get_env(
            "AI_TRADING_RUNTIME_PERF_TRADE_HISTORY_PATH",
            default_trade_history,
            cast=str,
        )
        or ""
    ).strip()
    gate_summary_configured = str(
        get_env(
            "AI_TRADING_RUNTIME_PERF_GATE_SUMMARY_PATH",
            "runtime/gate_effectiveness_summary.json",
            cast=str,
        )
        or ""
    ).strip()
    gate_log_configured = str(
        get_env("AI_TRADING_RUNTIME_PERF_GATE_LOG_PATH", "", cast=str) or ""
    ).strip()
    trade_history_path = resolve_runtime_artifact_path(
        trade_history_configured or default_trade_history,
        default_relative=default_trade_history,
    )
    gate_summary_path = resolve_runtime_artifact_path(
        gate_summary_configured or "runtime/gate_effectiveness_summary.json",
        default_relative="runtime/gate_effectiveness_summary.json",
    )
    gate_log_path: Path | None = None
    if gate_log_configured:
        gate_log_path = resolve_runtime_artifact_path(
            gate_log_configured,
            default_relative="runtime/gate_effectiveness.jsonl",
        )
    return {
        "trade_history": trade_history_path,
        "gate_summary": gate_summary_path,
        "gate_log": gate_log_path,
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
