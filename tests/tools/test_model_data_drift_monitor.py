from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trading.tools import model_data_drift_monitor
from ai_trading.tools import model_data_drift_baseline


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


def _governed_evidence(now: datetime, *, model_id: str = "shadow-1") -> dict[str, object]:
    fills = [
        {
            "ts": (now - timedelta(minutes=index)).isoformat(),
            "symbol": "AAPL" if index % 2 == 0 else "AMZN",
            "confidence": 0.60,
            "expected_net_edge_bps": 3.0,
            "realized_net_edge_bps": 1.0 if index % 3 else -0.5,
            "slippage_bps": 0.5,
            "fee_bps": 0.1,
        }
        for index in range(30)
    ]
    tca = [
        {
            "ts": (now - timedelta(minutes=index)).isoformat(),
            "provider": "alpaca",
            "market_regime": "sideways" if index % 2 == 0 else "downtrend",
            "fill_latency_ms": 100.0,
            "spread_paid_bps": 1.0,
            "decision_quote_age_ms": 50.0,
        }
        for index in range(30)
    ]
    return model_data_drift_baseline.build_model_data_drift_evidence(
        fills=fills,
        tca_rows=tca,
        generated_at=now,
        min_samples=25,
        model_id=model_id,
        model_hash="abc123",
    )


def test_model_data_drift_monitor_accepts_approved_compatible_baseline() -> None:
    now = datetime(2026, 7, 18, 5, 0, tzinfo=UTC)
    current = _governed_evidence(now)
    baseline = model_data_drift_baseline.build_governed_drift_baseline(
        current,
        baseline_id="shadow-1-20260718",
        approved_by="operator",
        approved_at=now,
    )

    report = model_data_drift_monitor.build_model_data_drift_monitor(
        baseline=baseline,
        current=current,
        now=now,
    )

    assert report["status"] == "ok"
    assert report["reasons"] == []
    assert report["contract"]["baseline_model_id"] == "shadow-1"
    assert report["contract"]["baseline_coverage"]["complete"] is True


def test_model_data_drift_monitor_rejects_unapproved_or_mismatched_evidence() -> None:
    now = datetime(2026, 7, 18, 5, 0, tzinfo=UTC)
    current = _governed_evidence(now, model_id="shadow-2")
    baseline = dict(_governed_evidence(now, model_id="shadow-1"))
    baseline.update(
        {
            "artifact_type": "model_data_drift_baseline",
            "status": "proposed",
            "approval": {"approved": False},
        }
    )

    report = model_data_drift_monitor.build_model_data_drift_monitor(
        baseline=baseline,
        current=current,
        now=now,
    )

    assert report["status"] == "blocked"
    assert "baseline_unapproved" in report["reasons"]
    assert "model_id_mismatch" in report["reasons"]


def test_model_data_drift_monitor_rejects_wrong_current_contract() -> None:
    now = datetime(2026, 7, 18, 5, 0, tzinfo=UTC)
    evidence = _governed_evidence(now)
    baseline = model_data_drift_baseline.build_governed_drift_baseline(
        evidence,
        baseline_id="shadow-1-20260718",
        approved_by="operator",
        approved_at=now,
    )
    wrong_current = dict(evidence)
    wrong_current["artifact_type"] = "expected_edge_calibration_report"

    report = model_data_drift_monitor.build_model_data_drift_monitor(
        baseline=baseline,
        current=wrong_current,
        now=now,
    )

    assert report["status"] == "blocked"
    assert "current_contract_invalid" in report["reasons"]
