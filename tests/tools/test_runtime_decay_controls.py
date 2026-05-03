from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ai_trading.tools import runtime_decay_controls


def test_runtime_decay_controls_disables_entries_on_execution_pause() -> None:
    artifact = runtime_decay_controls.build_runtime_decay_controls(
        execution_quality_governor={
            "status": {"available": True, "pause_active": True},
        },
        generated_at=datetime(2026, 5, 1, 15, 0, tzinfo=UTC),
    )

    assert artifact["actions"]["entries_allowed"] is False
    assert artifact["actions"]["max_action"] == "disable_new_entries"
    assert artifact["actions"]["size_scale"] == 0.0
    assert "execution_quality_pause" in artifact["actions"]["reasons"]


def test_runtime_decay_controls_reduces_size_on_cost_and_symbol_decay() -> None:
    artifact = runtime_decay_controls.build_runtime_decay_controls(
        runtime_gonogo={"gate_passed": False, "failed_checks": ["profit_factor"]},
        live_cost_model={"status": {"available": True, "breach_count": 1}},
        symbol_universe_scorecard={
            "symbols": [
                {"symbol": "MSFT", "effective_mode": "disabled"},
                {"symbol": "AAPL", "effective_mode": "allow"},
            ]
        },
        generated_at=datetime(2026, 5, 1, 15, 0, tzinfo=UTC),
    )

    assert artifact["actions"]["entries_allowed"] is True
    assert artifact["actions"]["max_action"] == "reduce_size"
    assert artifact["actions"]["size_scale"] == 0.5
    assert artifact["observed"]["disabled_symbols"] == 1
    assert set(artifact["actions"]["reasons"]) >= {
        "runtime_gonogo_failed",
        "live_cost_breach",
        "symbol_universe_disabled",
    }


def test_runtime_decay_controls_cli_writes_report(tmp_path: Path) -> None:
    live_cost = tmp_path / "live_cost_model_latest.json"
    output = tmp_path / "runtime_decay_controls_latest.json"
    live_cost.write_text(
        json.dumps(
            {
                "artifact_type": "live_cost_model",
                "status": {"available": True, "status": "ready", "breach_count": 2},
            }
        ),
        encoding="utf-8",
    )

    exit_code = runtime_decay_controls.main(
        [
            "--live-cost-model-json",
            str(live_cost),
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "runtime_decay_controls"
    assert payload["actions"]["max_action"] == "reduce_size"
    assert payload["paths"]["report"] == str(output)
