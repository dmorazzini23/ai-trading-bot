#!/usr/bin/env python3
"""One-command ops shift summary for pre-open, midday, and post-close checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import mcp_market_events_server as market_srv
from tools import mcp_metrics_query_server as metrics_srv
from tools import mcp_observability_server as obs_srv
from tools import mcp_ops_server as ops_srv
from tools import mcp_slack_alerts_server as slack_srv


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_phase(phase_arg: str) -> str:
    phase = phase_arg.strip().lower()
    if phase in {"pre_open", "midday", "post_close"}:
        return phase
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return "post_close"
    hhmm = now_et.hour * 60 + now_et.minute
    open_min = 9 * 60 + 30
    close_min = 16 * 60
    if hhmm < open_min:
        return "pre_open"
    if hhmm < close_min:
        return "midday"
    return "post_close"


def _safe_call(name: str, fn: Any, args: dict[str, Any]) -> dict[str, Any]:
    try:
        result = fn(args)
        return {"ok": True, "name": name, "result": result}
    except Exception as exc:  # pragma: no cover - runtime guard
        return {
            "ok": False,
            "name": name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def build_shift_summary(phase: str) -> dict[str, Any]:
    health_port = int(os.getenv("HEALTHCHECK_PORT", "8081"))
    checks: list[dict[str, Any]] = [
        _safe_call("health_probe", ops_srv.tool_health_probe, {"port": health_port}),
        _safe_call("service_status", obs_srv.tool_service_status, {"unit": "ai-trading"}),
        _safe_call("runtime_kpi_snapshot", obs_srv.tool_runtime_kpi_snapshot, {}),
    ]

    if phase == "pre_open":
        checks.extend(
            [
                _safe_call("market_risk_window", market_srv.tool_market_risk_window, {"horizon_hours": 24}),
                _safe_call("metrics_backend_status", metrics_srv.tool_metrics_backend_status, {}),
            ]
        )
    elif phase == "midday":
        checks.extend(
            [
                _safe_call(
                    "execution_trends_snapshot",
                    metrics_srv.tool_execution_trends_snapshot,
                    {"duration_minutes": 180, "step_s": 60},
                ),
                _safe_call("market_risk_window", market_srv.tool_market_risk_window, {"horizon_hours": 8}),
            ]
        )
    else:
        checks.extend(
            [
                _safe_call("runtime_eod_summary_snapshot", slack_srv.tool_runtime_eod_summary_snapshot, {}),
                _safe_call("market_sessions", market_srv.tool_market_sessions, {"days": 2}),
            ]
        )

    failures = [item for item in checks if not bool(item.get("ok", False))]
    summary = {
        "ts": _now_utc(),
        "phase": phase,
        "ok": len(failures) == 0,
        "failure_count": len(failures),
        "checks": checks,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ops shift summary checks.")
    parser.add_argument(
        "--phase",
        default="auto",
        choices=["auto", "pre_open", "midday", "post_close"],
        help="Shift phase to run (default: auto by ET clock).",
    )
    args = parser.parse_args(argv)

    phase = _resolve_phase(args.phase)
    payload = build_shift_summary(phase)
    print(json.dumps(payload, sort_keys=True))
    return 0 if bool(payload.get("ok", False)) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
