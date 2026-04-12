from __future__ import annotations

import importlib.util
import json
import os
from datetime import UTC, datetime
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
