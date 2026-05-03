from __future__ import annotations

import importlib.util
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path


def _load_monitor_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "monitor_execution_quality_60m.py"
    spec = importlib.util.spec_from_file_location("monitor_execution_quality_60m", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_summary_prefers_fresh_report_over_stale_state_pointer(
    tmp_path: Path,
) -> None:
    monitor = _load_monitor_module()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stale_report = reports_dir / "after_hours_training_20260411.json"
    stale_report.write_text(
        json.dumps(
            {
                "ts": "2026-04-11T10:00:00+00:00",
                "model": {"name": "stale_model"},
                "promotion": {
                    "combined_gates": {"expectancy": False, "drawdown": True},
                },
            }
        ),
        encoding="utf-8",
    )
    fresh_report = reports_dir / "after_hours_training_20260411_230000.json"
    fresh_report.write_text(
        json.dumps(
            {
                "ts": "2026-04-11T23:00:00+00:00",
                "model": {"name": "fresh_model"},
                "promotion": {
                    "combined_gates": {"expectancy": True, "drawdown": True},
                },
            }
        ),
        encoding="utf-8",
    )
    state_path = tmp_path / "after_hours_training_state.json"
    state_path.write_text(
        json.dumps(
            {
                "report_path": str(stale_report),
                "model_id": "state-model-id",
                "governance_status": "shadow",
            }
        ),
        encoding="utf-8",
    )

    summary = monitor._training_summary(state_path=state_path, reports_dir=reports_dir)
    assert summary["report_path"] == str(fresh_report)
    assert summary["model_name"] == "fresh_model"
    assert summary["gate_blockers"] == []
    assert summary["model_id"] == "state-model-id"


def test_resolve_report_payload_falls_back_to_mtime_when_ts_missing(
    tmp_path: Path,
) -> None:
    monitor = _load_monitor_module()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    older = reports_dir / "after_hours_training_20260411_120000.json"
    newer = reports_dir / "after_hours_training_20260411_130000.json"
    older.write_text(json.dumps({"model": {"name": "older"}}), encoding="utf-8")
    newer.write_text(json.dumps({"model": {"name": "newer"}}), encoding="utf-8")
    older_mtime = datetime(2026, 4, 11, 12, 0, tzinfo=UTC).timestamp()
    newer_mtime = datetime(2026, 4, 11, 13, 0, tzinfo=UTC).timestamp()
    os.utime(older, (older_mtime, older_mtime))
    os.utime(newer, (newer_mtime, newer_mtime))

    payload, selected = monitor._resolve_report_payload(state={}, reports_dir=reports_dir)
    assert selected == newer
    assert payload.get("model", {}).get("name") == "newer"


def test_training_summary_handles_null_model_payload(tmp_path: Path) -> None:
    monitor = _load_monitor_module()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "after_hours_training_20260411_230000.json"
    report_path.write_text(
        json.dumps(
            {
                "ts": "2026-04-11T23:00:00+00:00",
                "model": None,
                "promotion": {"combined_gates": {"expectancy": False}},
            }
        ),
        encoding="utf-8",
    )
    state_path = tmp_path / "after_hours_training_state.json"
    state_path.write_text(json.dumps({"model_id": "state-model-id"}), encoding="utf-8")

    summary = monitor._training_summary(state_path=state_path, reports_dir=reports_dir)
    assert summary["model_name"] is None
    assert summary["model_id"] == "state-model-id"
    assert summary["gate_blockers"] == ["expectancy"]


def test_window_metrics_preserves_zero_realized_and_microstructure_fields(
    tmp_path: Path,
) -> None:
    monitor = _load_monitor_module()
    events_path = tmp_path / "execution_quality_events.jsonl"
    now = datetime.now(UTC)
    rows = [
        {
            "ts": (now - timedelta(minutes=5)).isoformat(),
            "event": "submit_outcome",
            "status": "filled",
            "symbol": "AAPL",
            "realized_bps": 0.0,
            "realized_net_edge_bps": 4.0,
            "spread_bps": 6.5,
            "quote_age_ms": 225.0,
        },
        {
            "ts": (now - timedelta(minutes=4)).isoformat(),
            "event": "submit_skipped",
            "status": "skipped",
            "symbol": "MSFT",
            "reason": "EXECUTION_QUALITY_SPREAD_BLOCK",
            "context": {"spread_bps": 55.0, "quote_age_ms": 800.0},
        },
    ]
    events_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    metrics = monitor._window_metrics(events_path, window_minutes=30, top_n=5)

    assert metrics["event_count"] == 2
    assert metrics["filled_count"] == 1
    assert metrics["rolling_sum_realized_bps"] == 0.0
    assert metrics["realized_samples"] == 1
    assert metrics["blocked_count"] == 1
    assert metrics["mean_spread_bps"] == 30.75
    assert metrics["mean_quote_age_ms"] == 512.5
    assert metrics["by_symbol"][0]["symbol"] in {"AAPL", "MSFT"}


def test_governor_report_serializes_operator_artifact(tmp_path: Path) -> None:
    monitor = _load_monitor_module()
    events_path = tmp_path / "execution_quality_events.jsonl"
    output_path = tmp_path / "execution_quality_governor_latest.json"
    metrics = {
        "now": datetime(2026, 5, 1, 15, 0, tzinfo=UTC),
        "window_minutes": 60,
        "event_count": 3,
        "filled_count": 1,
        "submit_no_result": 0,
        "blocked_count": 2,
        "derisked_count": 1,
        "passive_low_skips": 0,
        "reason_counts": {"EXECUTION_QUALITY_SPREAD_BLOCK": 2},
        "mean_spread_bps": 22.0,
        "p90_spread_bps": 50.0,
        "mean_quote_age_ms": 350.0,
        "p90_quote_age_ms": 900.0,
        "passive_low_skip_share": 0.0,
        "rolling_sum_realized_bps": 1.25,
        "realized_samples": 1,
        "by_symbol": [{"symbol": "MSFT", "events": 2}],
    }

    report = monitor._governor_report(
        metrics=metrics,
        events_path=events_path,
        output_path=output_path,
    )

    assert report["artifact_type"] == "execution_quality_governor_report"
    assert report["status"]["gate_passed"] is True
    assert report["status"]["mode"] == "derisk"
    assert report["actions"]["blocked_count"] == 2
    assert report["observed"]["p90_spread_bps"] == 50.0
    assert report["paths"]["report"] == str(output_path)
