from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trading.tools import model_data_drift_monitor


def _payload(generated_at: str) -> dict[str, object]:
    return {
        "generated_at": generated_at,
        "features": {
            "momentum_5m": {"mean": 1.0, "std": 2.0, "missing_rate": 0.01},
            "spread_bps": {"mean": 4.0, "std": 1.0, "missing_rate": 0.0},
        },
        "labels": {"positive_rate": 0.52, "mean_return_bps": 3.0},
        "calibration": {"capture_ratio": 0.70, "brier_score": 0.12},
        "live_cost": {"mean_total_cost_bps": 3.0, "p90_total_cost_bps": 5.0},
        "symbols": {"counts": {"AAPL": 80, "AMZN": 20}},
        "providers": {"counts": {"alpaca": 100}},
        "regimes": {"counts": {"midday": 70, "opening": 30}},
    }


def test_model_data_drift_monitor_covers_all_categories_and_detects_drift() -> None:
    now = datetime(2026, 5, 8, 15, 0, tzinfo=UTC)
    baseline = _payload(now.isoformat())
    current = _payload(now.isoformat())
    current["features"]["momentum_5m"]["mean"] = 2.0
    current["labels"]["positive_rate"] = 0.30
    current["calibration"]["capture_ratio"] = 0.30
    current["live_cost"]["p90_total_cost_bps"] = 10.0
    current["symbols"] = {"counts": {"AAPL": 30, "AMZN": 70}}
    current["providers"] = {"counts": {"alpaca": 50, "backup": 50}}
    current["regimes"] = {"counts": {"midday": 20, "opening": 80}}

    report = model_data_drift_monitor.build_model_data_drift_monitor(
        baseline=baseline,
        current=current,
        now=now,
    )

    assert report["status"] == "drift_detected"
    assert report["summary"]["covered_categories"] == [
        "feature",
        "label",
        "calibration",
        "live_cost",
        "symbol",
        "provider",
        "regime",
    ]
    assert set(report["summary"]["drift_categories"]) == set(report["summary"]["covered_categories"])


def test_model_data_drift_monitor_blocks_stale_baseline() -> None:
    now = datetime(2026, 5, 8, 15, 0, tzinfo=UTC)
    baseline = _payload((now - timedelta(days=10)).isoformat())
    current = _payload(now.isoformat())

    report = model_data_drift_monitor.build_model_data_drift_monitor(
        baseline=baseline,
        current=current,
        max_baseline_age_hours=24.0,
        now=now,
    )

    assert report["status"] == "blocked"
    assert "baseline_stale" in report["reasons"]
    assert report["freshness"]["baseline"]["fresh"] is False


def test_model_data_drift_monitor_cli_writes_artifact(tmp_path: Path) -> None:
    now = datetime.now(UTC).isoformat()
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    output = tmp_path / "drift.json"
    baseline_path.write_text(json.dumps(_payload(now)), encoding="utf-8")
    current_path.write_text(json.dumps(_payload(now)), encoding="utf-8")

    rc = model_data_drift_monitor.main(
        [
            "--baseline-json",
            str(baseline_path),
            "--current-json",
            str(current_path),
            "--output-json",
            str(output),
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "model_data_drift_monitor"
    assert payload["status"] == "ok"

