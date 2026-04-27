from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trading.analytics import execution_report
from ai_trading.analytics.execution_report import write_daily_execution_report
from ai_trading.analytics.tca import (
    ExecutionBenchmark,
    FillSummary,
    build_tca_record,
    write_tca_record,
)


def test_execution_report_daily_rollup(tmp_path: Path) -> None:
    tca_path = tmp_path / "tca.jsonl"
    bench = ExecutionBenchmark(arrival_price=100.0, mid_at_arrival=100.0)
    filled = FillSummary(fill_vwap=100.2, total_qty=10.0, fees=0.0, status="filled")
    rec1 = build_tca_record(
        client_order_id="cid-1",
        symbol="AAPL",
        side="buy",
        benchmark=bench,
        fill=filled,
        sleeve="day",
        regime_profile="balanced",
        provider="alpaca",
        order_type="limit",
    )
    rec2 = build_tca_record(
        client_order_id="cid-2",
        symbol="AAPL",
        side="sell",
        benchmark=bench,
        fill=filled,
        sleeve="day",
        regime_profile="balanced",
        provider="alpaca",
        order_type="limit",
    )
    write_tca_record(str(tca_path), rec1)
    write_tca_record(str(tca_path), rec2)

    out_dir = tmp_path / "reports"
    report = write_daily_execution_report(
        tca_path=str(tca_path),
        output_dir=str(out_dir),
        formats=("json", "csv"),
    )
    assert report["records"] == 2
    assert report["groups"]
    assert any(path.suffix == ".json" for path in out_dir.iterdir())
    assert any(path.suffix == ".csv" for path in out_dir.iterdir())


def test_execution_report_rollup_tz_controls_output_day(tmp_path: Path, monkeypatch) -> None:
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            base = datetime(2025, 1, 2, 4, 30, tzinfo=UTC)
            if tz is None:
                return base
            return base.astimezone(tz)

    monkeypatch.setattr(execution_report, "datetime", _FixedDatetime)
    tca_path = tmp_path / "tca.jsonl"
    write_tca_record(
        str(tca_path),
        {
            "client_order_id": "cid-ny",
            "symbol": "AAPL",
            "is_bps": 3.0,
            "spread_paid_bps": 1.0,
            "status": "filled",
            "partial_fill": False,
            "blocked_by_gate": False,
            "sleeve": "day",
            "regime_profile": "balanced",
            "order_type": "limit",
            "provider": "alpaca",
        },
    )
    out_dir = tmp_path / "reports"
    write_daily_execution_report(
        tca_path=str(tca_path),
        output_dir=str(out_dir),
        formats=("json",),
        rollup_tz="America/New_York",
    )
    assert (out_dir / "execution_report_20250101.json").exists()


def test_execution_report_groups_preserve_lineage_dimensions(tmp_path: Path) -> None:
    records = [
        {
            "symbol": "AAPL",
            "sleeve": "day",
            "regime_profile": "balanced",
            "order_type": "limit",
            "provider": "alpaca",
            "status": "filled",
            "is_bps": 4.0,
            "spread_paid_bps": 1.0,
            "model_id": "model-a",
            "model_version": "v1",
            "config_snapshot_hash": "cfg-a",
            "order_side": "buy",
            "rank_reason": "EDGE_RANKED",
        },
        {
            "symbol": "AAPL",
            "sleeve": "day",
            "regime_profile": "balanced",
            "order_type": "limit",
            "provider": "alpaca",
            "status": "filled",
            "is_bps": 5.0,
            "spread_paid_bps": 1.2,
            "model_id": "model-b",
            "model_version": "v2",
            "config_snapshot_hash": "cfg-b",
            "order_side": "sell_short",
            "rank_reasons": ["SHORT_EDGE_RANKED"],
        },
    ]

    report = execution_report.build_daily_execution_report(records)

    assert len(report["groups"]) == 2
    groups_by_model = {group["model_id"]: group for group in report["groups"]}
    assert groups_by_model["model-a"]["config_snapshot_hash"] == "cfg-a"
    assert groups_by_model["model-a"]["order_side"] == "buy"
    assert groups_by_model["model-a"]["rank_reason"] == "EDGE_RANKED"
    assert groups_by_model["model-b"]["model_version"] == "v2"
    assert groups_by_model["model-b"]["order_side"] == "sell_short"
    assert groups_by_model["model-b"]["rank_reason"] == "SHORT_EDGE_RANKED"

    csv_path = tmp_path / "report.csv"
    execution_report._write_csv(csv_path, report["groups"])  # noqa: SLF001 - report surface check
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "model_id" in header
    assert "config_snapshot_hash" in header
    assert "order_side" in header


def test_execution_report_phase2_gate_passes_with_baselines(monkeypatch) -> None:
    now = datetime.now(UTC)
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_BASELINE_SLIPPAGE_MEDIAN_BPS", "10.0")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_BASELINE_FILL_RATE", "0.90")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_BASELINE_STALE_PENDING_COUNT", "2")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_MIN_SLIPPAGE_IMPROVEMENT_PCT", "10.0")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_MAX_FILL_RATE_DEGRADATION_PCT", "5.0")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_MAX_REJECT_RATE", "0.10")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_MAX_EXECUTION_DRIFT_BPS", "10.0")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_MAX_STALE_PENDING_INCREASE", "0")
    records: list[dict[str, object]] = []
    for idx in range(10):
        records.append(
            {
                "ts": (now - timedelta(hours=1)).isoformat(),
                "symbol": "AAPL",
                "status": "filled",
                "is_bps": 8.0,
                "order_type": "limit",
                "midpoint_offset_bps": 4.0,
                "execution_drift_bps": 5.0,
                "partial_fill": idx == 0,
            }
        )
    records.append(
        {
            "ts": (now - timedelta(hours=1)).isoformat(),
            "symbol": "AAPL",
            "status": "rejected",
            "is_bps": None,
            "order_type": "limit",
            "midpoint_offset_bps": 3.0,
            "execution_drift_bps": 4.0,
        }
    )
    records.append(
        {
            "ts": (now - timedelta(hours=1)).isoformat(),
            "symbol": "AAPL",
            "status": "canceled",
            "order_type": "limit",
            "pending_terminal_nonfill": True,
        }
    )
    records.append(
        {
            "ts": (now - timedelta(hours=1)).isoformat(),
            "symbol": "AAPL",
            "status": "canceled",
            "order_type": "limit",
            "pending_terminal_nonfill": True,
        }
    )
    report = execution_report.build_daily_execution_report(records)
    phase2 = report["roadmap"]["phase_2_execution_edge"]
    assert phase2["enabled"] is True
    assert phase2["gate_passed"] is True
    assert phase2["effective_gates"]["slippage_improvement"] is True
    assert phase2["effective_gates"]["fill_rate_degradation"] is True
    assert phase2["effective_gates"]["reject_rate_slo"] is True
    assert phase2["effective_gates"]["execution_drift_slo"] is True
    assert phase2["effective_gates"]["stale_pending_incidents"] is True


def test_execution_report_phase2_gate_fails_on_reject_and_stale_pending(monkeypatch) -> None:
    now = datetime.now(UTC)
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_BASELINE_SLIPPAGE_MEDIAN_BPS", "12.0")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_BASELINE_FILL_RATE", "0.95")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_BASELINE_STALE_PENDING_COUNT", "0")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_MAX_REJECT_RATE", "0.05")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_MAX_EXECUTION_DRIFT_BPS", "8.0")
    records = [
        {
            "ts": (now - timedelta(hours=2)).isoformat(),
            "symbol": "MSFT",
            "status": "filled",
            "is_bps": 9.0,
            "order_type": "limit",
            "midpoint_offset_bps": 3.0,
            "execution_drift_bps": 3.0,
        },
        {
            "ts": (now - timedelta(hours=2)).isoformat(),
            "symbol": "MSFT",
            "status": "rejected",
            "order_type": "limit",
            "midpoint_offset_bps": 3.0,
        },
        {
            "ts": (now - timedelta(hours=2)).isoformat(),
            "symbol": "MSFT",
            "status": "rejected",
            "order_type": "limit",
            "midpoint_offset_bps": 3.0,
        },
        {
            "ts": (now - timedelta(hours=2)).isoformat(),
            "symbol": "MSFT",
            "status": "canceled",
            "order_type": "limit",
            "pending_terminal_nonfill": True,
        },
    ]
    report = execution_report.build_daily_execution_report(records)
    phase2 = report["roadmap"]["phase_2_execution_edge"]
    assert phase2["enabled"] is True
    assert phase2["gate_passed"] is False
    assert phase2["effective_gates"]["reject_rate_slo"] is False
    assert phase2["effective_gates"]["stale_pending_incidents"] is False


def test_execution_report_phase2_calibration_hints_stay_disabled(monkeypatch) -> None:
    now = datetime.now(UTC)
    monkeypatch.setenv("AI_TRADING_PHASE2_EXECUTION_EDGE_MIN_SAMPLES", "2")
    records = [
        {
            "ts": (now - timedelta(hours=1)).isoformat(),
            "symbol": "AAPL",
            "status": "filled",
            "order_type": "limit",
            "midpoint_offset_bps": 4.0,
            "is_bps": 4.0,
        },
        {
            "ts": (now - timedelta(hours=1)).isoformat(),
            "symbol": "AAPL",
            "status": "canceled",
            "order_type": "limit",
            "midpoint_offset_bps": 4.0,
        },
    ]

    report = execution_report.build_daily_execution_report(records)

    calibration = report["roadmap"]["phase_2_execution_edge"]["calibration"]
    assert calibration["sufficient"] is True
    assert calibration["missing"] == []
    threshold_hints = calibration["recommended_routing_thresholds"]
    assert threshold_hints["AI_TRADING_PHASE2_EXECUTION_EDGE_ROUTING_ENABLED"] is False
    assert threshold_hints["AI_TRADING_PHASE2_EXECUTION_EDGE_MIN_SAMPLES"] == 2
    assert threshold_hints["AI_TRADING_PHASE2_EXECUTION_EDGE_TARGET_FILL_RATE"] == 0.5
    assert threshold_hints["AI_TRADING_PHASE2_EXECUTION_EDGE_TARGET_SLIPPAGE_BPS"] == 5.0
    assert threshold_hints["AI_TRADING_PHASE2_EXECUTION_EDGE_OFFSET_WEIGHT"] == 1.0


def test_execution_report_phase2_loads_baselines_from_file(tmp_path: Path, monkeypatch) -> None:
    baseline_path = tmp_path / "phase2_execution_baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "baselines": {
                    "slippage_median_abs_bps": 11.0,
                    "target_limit_fill_rate": 0.88,
                    "stale_pending_count": 3.0,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_BASELINE_PATH", str(baseline_path))
    monkeypatch.delenv("AI_TRADING_ROADMAP_PHASE2_BASELINE_SLIPPAGE_MEDIAN_BPS", raising=False)
    monkeypatch.delenv("AI_TRADING_ROADMAP_PHASE2_BASELINE_FILL_RATE", raising=False)
    monkeypatch.delenv("AI_TRADING_ROADMAP_PHASE2_BASELINE_STALE_PENDING_COUNT", raising=False)
    report = execution_report.build_daily_execution_report([])
    phase2 = report["roadmap"]["phase_2_execution_edge"]
    assert phase2["baselines"]["slippage_median_abs_bps"] == 11.0
    assert phase2["baselines"]["target_limit_fill_rate"] == 0.88
    assert phase2["baselines"]["stale_pending_count"] == 3.0
    assert phase2["baselines"]["sources"]["slippage_median_abs_bps"] == "file"


def test_execution_report_phase2_env_overrides_file_baselines(
    tmp_path: Path, monkeypatch
) -> None:
    baseline_path = tmp_path / "phase2_execution_baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "baselines": {
                    "slippage_median_abs_bps": 20.0,
                    "target_limit_fill_rate": 0.50,
                    "stale_pending_count": 9.0,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_BASELINE_PATH", str(baseline_path))
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_BASELINE_SLIPPAGE_MEDIAN_BPS", "10.0")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_BASELINE_FILL_RATE", "0.95")
    monkeypatch.setenv("AI_TRADING_ROADMAP_PHASE2_BASELINE_STALE_PENDING_COUNT", "1")
    report = execution_report.build_daily_execution_report([])
    phase2 = report["roadmap"]["phase_2_execution_edge"]
    assert phase2["baselines"]["slippage_median_abs_bps"] == 10.0
    assert phase2["baselines"]["target_limit_fill_rate"] == 0.95
    assert phase2["baselines"]["stale_pending_count"] == 1.0
    assert phase2["baselines"]["sources"]["target_limit_fill_rate"] == "env"
