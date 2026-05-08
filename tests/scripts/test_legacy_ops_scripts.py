from __future__ import annotations

from pathlib import Path

from scripts import health_check


def test_legacy_health_check_exits_nonzero_on_critical(monkeypatch) -> None:
    monkeypatch.setattr(
        health_check,
        "get_health_status",
        lambda: {"overall_status": "critical", "checks": {}},
    )

    assert health_check.main() == 1


def test_legacy_health_check_exits_zero_when_not_critical(monkeypatch) -> None:
    monkeypatch.setattr(
        health_check,
        "get_health_status",
        lambda: {"overall_status": "warning", "checks": {}},
    )

    assert health_check.main() == 0


def test_runtime_prune_script_covers_retention_planner_artifacts() -> None:
    script = Path("scripts/prune_runtime_jsonl.sh").read_text(encoding="utf-8")

    for filename in (
        "decision_records.jsonl",
        "config_snapshots.jsonl",
        "gate_effectiveness.jsonl",
        "ml_shadow_predictions.jsonl",
        "order_events.jsonl",
        "fill_events.jsonl",
        "tca_records.jsonl",
        "oms_events.jsonl",
        "memory_samples.jsonl",
    ):
        assert filename in script
