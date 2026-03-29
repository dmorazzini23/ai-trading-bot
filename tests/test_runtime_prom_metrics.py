from __future__ import annotations

import json

import pytest

from ai_trading.metrics import PROMETHEUS_AVAILABLE, get_registry, reset_registry
from ai_trading.telemetry import runtime_prom_metrics


def test_refresh_runtime_execution_metrics_reads_daily_report(tmp_path, monkeypatch) -> None:
    if not PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus client unavailable")

    report_path = tmp_path / "daily_performance_report.json"
    report_path.write_text(
        json.dumps(
            {
                "execution_vs_alpha": {
                    "slippage_drag_bps": 6.25,
                    "execution_capture_ratio": 0.42,
                },
                "go_no_go": {
                    "observed": {
                        "order_reject_rate_pct": 1.75,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runtime_prom_metrics,
        "get_env",
        lambda name, default=None, cast=None: (
            str(report_path)
            if name == "AI_TRADING_RUNTIME_DAILY_REPORT_PATH"
            else default
        ),
    )

    reset_registry()
    values = runtime_prom_metrics.refresh_runtime_execution_metrics()

    assert values["slippage_drag_bps"] == 6.25
    assert values["execution_capture_ratio"] == 0.42
    assert values["order_reject_rate_pct"] == 1.75
    assert values["runtime_report_age_seconds"] is not None

    from prometheus_client import generate_latest

    metrics_text = generate_latest(get_registry()).decode("utf-8", errors="replace")
    assert "ai_trading_slippage_drag_bps" in metrics_text
    assert "ai_trading_execution_capture_ratio" in metrics_text
    assert "ai_trading_order_reject_rate_pct" in metrics_text


def test_refresh_runtime_execution_metrics_defaults_reject_rate_when_missing(
    tmp_path, monkeypatch
) -> None:
    if not PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus client unavailable")

    report_path = tmp_path / "daily_performance_report.json"
    report_path.write_text(
        json.dumps(
            {
                "execution_vs_alpha": {
                    "slippage_drag_bps": 7.0,
                    "execution_capture_ratio": 0.18,
                },
                "go_no_go": {"observed": {}},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runtime_prom_metrics,
        "get_env",
        lambda name, default=None, cast=None: (
            str(report_path)
            if name == "AI_TRADING_RUNTIME_DAILY_REPORT_PATH"
            else default
        ),
    )

    reset_registry()
    values = runtime_prom_metrics.refresh_runtime_execution_metrics()
    assert values["order_reject_rate_pct"] == 0.0

