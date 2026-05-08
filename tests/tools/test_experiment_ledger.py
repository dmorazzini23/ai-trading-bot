from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from ai_trading.tools import experiment_ledger


def test_success_run_records_hashes_and_completion(tmp_path: Path) -> None:
    input_path = tmp_path / "shadow.jsonl"
    input_path.write_text('{"symbol":"AAPL"}\n', encoding="utf-8")

    ledger = experiment_ledger.build_experiment_ledger(
        run_id="run-1",
        workflow="multi_horizon_research",
        status="success",
        conclusion="candidate improved net edge",
        input_paths=[input_path],
        config={"horizons": [1, 5], "symbols": ["AAPL"]},
        generated_at=datetime(2026, 5, 5, 21, 0, tzinfo=UTC),
        reported_complete=True,
    )

    expected_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert ledger["status"] == "recorded"
    assert ledger["latest_run"]["reported_complete"] is True
    assert ledger["latest_run"]["inputs"][0]["sha256"] == expected_hash
    assert ledger["latest_run"]["config_hash"]
    assert ledger["summary"]["reported_complete"] == 1


def test_failed_run_cannot_be_reported_complete() -> None:
    ledger = experiment_ledger.build_experiment_ledger(
        run_id="run-2",
        workflow="multi_horizon_research",
        status="failed",
        conclusion="training crashed before evaluation",
        generated_at=datetime(2026, 5, 5, 21, 0, tzinfo=UTC),
        reported_complete=True,
    )

    assert ledger["status"] == "blocked"
    assert ledger["latest_run"]["reported_complete"] is False
    assert ledger["latest_run"]["completion_guard"] == "blocked"
    assert "failed_run_cannot_report_complete" in ledger["blocked_reasons"]


def test_blocked_and_dry_run_statuses_are_recorded_without_completion() -> None:
    generated = datetime(2026, 5, 5, 21, 0, tzinfo=UTC)
    ledger = experiment_ledger.build_experiment_ledger(
        run_id="run-blocked",
        workflow="promotion_research",
        status="blocked",
        conclusion="runtime health gate failed",
        generated_at=generated,
    )
    ledger = experiment_ledger.build_experiment_ledger(
        run_id="run-dry",
        workflow="promotion_research",
        status="dry-run",
        conclusion="planned only",
        previous_ledger=ledger,
        generated_at=generated,
    )

    assert ledger["status"] == "recorded"
    assert ledger["summary"]["blocked"] == 1
    assert ledger["summary"]["dry_run"] == 1
    assert ledger["summary"]["reported_complete"] == 0


def test_experiment_ledger_cli_writes_dated_and_latest_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "ledger"
    latest = output_dir / "experiment_ledger_latest.json"
    input_path.write_text('{"rows": 10}\n', encoding="utf-8")
    config_path.write_text(json.dumps({"alpha": 0.1}), encoding="utf-8")

    exit_code = experiment_ledger.main(
        [
            "--run-id",
            "run-1",
            "--workflow",
            "shadow_eval",
            "--status",
            "success",
            "--conclusion",
            "shadow evaluation passed",
            "--input-path",
            str(input_path),
            "--config-json",
            str(config_path),
            "--reported-complete",
            "--output-dir",
            str(output_dir),
        ]
    )

    dated = sorted(output_dir.glob("experiment_ledger_*.json"))
    assert exit_code == 0
    assert latest.is_file()
    assert dated
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "experiment_ledger"
    assert payload["latest_run"]["run_id"] == "run-1"
    assert payload["latest_run"]["reported_complete"] is True
    assert payload["paths"]["latest"] == str(latest)


def test_experiment_ledger_cli_returns_blocked_for_failed_complete(
    tmp_path: Path,
) -> None:
    output = tmp_path / "ledger.json"
    latest = tmp_path / "ledger_latest.json"

    exit_code = experiment_ledger.main(
        [
            "--run-id",
            "run-failed",
            "--workflow",
            "shadow_eval",
            "--status",
            "failed",
            "--conclusion",
            "input artifact missing",
            "--reported-complete",
            "--output-json",
            str(output),
            "--latest-json",
            str(latest),
        ]
    )

    assert exit_code == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["latest_run"]["reported_complete"] is False
    assert latest.is_file()
