from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger
from ai_trading.metrics import get_gauge

_log = get_logger(__name__)

_DEFAULT_RUNTIME_REPORT_PATH = "/var/lib/ai-trading-bot/runtime/daily_performance_report.json"


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _runtime_report_path(path: str | None = None) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    configured = str(
        get_env(
            "AI_TRADING_RUNTIME_DAILY_REPORT_PATH",
            _DEFAULT_RUNTIME_REPORT_PATH,
            cast=str,
        )
        or _DEFAULT_RUNTIME_REPORT_PATH
    ).strip()
    if not configured:
        configured = _DEFAULT_RUNTIME_REPORT_PATH
    return Path(configured).expanduser().resolve()


def _gauges() -> dict[str, Any]:
    return {
        "slippage_drag_bps": get_gauge(
            "ai_trading_slippage_drag_bps",
            "Execution slippage drag in bps from runtime report.",
        ),
        "execution_capture_ratio": get_gauge(
            "ai_trading_execution_capture_ratio",
            "Execution capture ratio from runtime report.",
        ),
        "order_reject_rate_pct": get_gauge(
            "ai_trading_order_reject_rate_pct",
            "Order reject rate percent from runtime report/go-no-go observed data.",
        ),
        "runtime_report_age_seconds": get_gauge(
            "ai_trading_runtime_report_age_seconds",
            "Age in seconds of the runtime daily performance report file.",
        ),
        "runtime_report_present": get_gauge(
            "ai_trading_runtime_report_present",
            "Whether the runtime daily performance report file is present.",
        ),
        "runtime_report_stale": get_gauge(
            "ai_trading_runtime_report_stale",
            "Whether the runtime daily performance report file is missing or older than the configured maximum age.",
        ),
    }


def refresh_runtime_execution_metrics(report_path: str | None = None) -> dict[str, float | None]:
    """Refresh `ai_trading_*` execution gauges from runtime report JSON."""

    output: dict[str, float | None] = {
        "slippage_drag_bps": None,
        "execution_capture_ratio": None,
        "order_reject_rate_pct": None,
        "runtime_report_age_seconds": None,
        "runtime_report_present": 0.0,
        "runtime_report_stale": 1.0,
    }

    gauges = _gauges()
    path = _runtime_report_path(report_path)
    if not path.exists():
        gauges["runtime_report_present"].set(0.0)
        gauges["runtime_report_stale"].set(1.0)
        return output

    output["runtime_report_present"] = 1.0
    gauges["runtime_report_present"].set(1.0)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _log.debug(
            "PROM_RUNTIME_REPORT_READ_FAILED",
            extra={"path": str(path), "error": str(exc)},
        )
        gauges["runtime_report_stale"].set(1.0)
        return output
    if not isinstance(raw, dict):
        gauges["runtime_report_stale"].set(1.0)
        return output

    execution = raw.get("execution_vs_alpha")
    go_no_go = raw.get("go_no_go")
    if not isinstance(execution, dict):
        execution = {}
    if not isinstance(go_no_go, dict):
        go_no_go = {}
    observed = go_no_go.get("observed")
    if not isinstance(observed, dict):
        observed = {}

    slippage_drag_bps = _as_float(execution.get("slippage_drag_bps"))
    if slippage_drag_bps is None:
        slippage_drag_bps = _as_float(observed.get("slippage_drag_bps"))

    execution_capture_ratio = _as_float(execution.get("execution_capture_ratio"))
    if execution_capture_ratio is None:
        execution_capture_ratio = _as_float(observed.get("execution_capture_ratio"))

    order_reject_rate_pct = _as_float(observed.get("order_reject_rate_pct"))
    if order_reject_rate_pct is None:
        order_reject_rate_pct = 0.0

    try:
        report_age_seconds = max(0.0, time.time() - float(path.stat().st_mtime))
    except OSError:
        report_age_seconds = None
    max_age_seconds = _as_float(
        get_env(
            "AI_TRADING_RUNTIME_REPORT_MAX_AGE_SECONDS",
            90_000,
            cast=float,
        )
    )
    if max_age_seconds is None or max_age_seconds <= 0:
        max_age_seconds = 90_000.0
    report_stale = (
        1.0
        if report_age_seconds is None or report_age_seconds > max_age_seconds
        else 0.0
    )

    output["slippage_drag_bps"] = slippage_drag_bps
    output["execution_capture_ratio"] = execution_capture_ratio
    output["order_reject_rate_pct"] = order_reject_rate_pct
    output["runtime_report_age_seconds"] = report_age_seconds
    output["runtime_report_stale"] = report_stale

    if slippage_drag_bps is not None:
        gauges["slippage_drag_bps"].set(float(slippage_drag_bps))
    if execution_capture_ratio is not None:
        gauges["execution_capture_ratio"].set(float(execution_capture_ratio))
    gauges["order_reject_rate_pct"].set(float(order_reject_rate_pct))
    if report_age_seconds is not None:
        gauges["runtime_report_age_seconds"].set(float(report_age_seconds))
    gauges["runtime_report_stale"].set(float(report_stale))

    return output
