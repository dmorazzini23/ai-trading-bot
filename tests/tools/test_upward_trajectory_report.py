from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import upward_trajectory_report as report_tool


def _candidate(name: str, validation_edge: float, replay_expectancy: float) -> dict[str, object]:
    return {
        "model_name": name,
        "horizon_bars": 15,
        "label_objective": "risk_adjusted",
        "dataset": {"rows": 100, "timestamp_authoritative": True},
        "validation": {"roc_auc": 0.64},
        "threshold_sweep": [{"mean_net_markout_bps": validation_edge}],
        "replay": {
            "expectancy_bps": replay_expectancy,
            "profit_factor": 0.5 if replay_expectancy < 0 else 1.4,
            "win_rate": 0.35,
            "total_trades": 50,
        },
    }


def test_upward_trajectory_reports_requested_but_unapplied_live_cost_labels() -> None:
    payload = report_tool.build_upward_trajectory_report(
        training_accelerator={"status": "complete", "config": {"use_live_cost_model": True}},
        multi_horizon_report={
            "status": "complete",
            "ranked_candidates": [
                {
                    **_candidate("gap-model", 40.0, -24.0),
                    "live_cost_model": {
                        "requested": True,
                        "usable": False,
                        "reason": "status_warming_up",
                    },
                }
            ],
        },
        live_cost_model={"status": {"status": "warming_up"}},
    )

    labels = payload["execution_aware_model_labels"]
    assert labels["live_cost_model_requested"] is True
    assert labels["live_cost_model_applied_candidate_count"] == 0
    assert labels["coverage"]["live_cost_adjusted_net_edge"] is False
    assert labels["live_cost_model_unusable_reasons"] == {"status_warming_up": 1}


def test_upward_trajectory_reports_applied_live_cost_labels() -> None:
    payload = report_tool.build_upward_trajectory_report(
        training_accelerator={"status": "complete", "config": {"use_live_cost_model": True}},
        multi_horizon_report={
            "status": "complete",
            "ranked_candidates": [
                {
                    **_candidate("ok-model", 5.0, 2.0),
                    "live_cost_model": {
                        "requested": True,
                        "usable": True,
                        "reason": "loaded",
                    },
                }
            ],
        },
        live_cost_model={"status": {"status": "ready"}},
    )

    labels = payload["execution_aware_model_labels"]
    assert labels["live_cost_model_applied_candidate_count"] == 1
    assert labels["coverage"]["live_cost_adjusted_net_edge"] is True


def test_upward_trajectory_report_flags_validation_replay_gap() -> None:
    payload = report_tool.build_upward_trajectory_report(
        report_date="2026-05-18",
        expected_edge_calibration={"status": "insufficient_samples"},
        execution_capture={"status": "insufficient_samples", "summary": {}},
        training_accelerator={"status": "complete"},
        multi_horizon_report={
            "status": "complete",
            "ranked_candidates": [_candidate("gap-model", 40.0, -24.0)],
        },
        symbol_lifecycle={
            "status": "ready",
            "symbols": [{"symbol": "AAPL", "recommendation": "collect_more_evidence"}],
        },
        live_cost_model={"status": {"status": "warming_up"}},
    )

    assert payload["status"] == "ready"
    assert payload["authority"]["runtime_authority"] is False
    assert payload["summary"]["feature_count"] == 8
    assert payload["summary"]["recommended_next_action"] == "debug_validation_replay_gap_before_promotion"
    gap = payload["validation_to_replay_gap_analyzer"]
    assert gap["diagnosis"] == "replay_gap_detected"
    assert gap["classification_counts"]["validation_positive_replay_negative"] == 1
    assert payload["evidence_acceleration_engine"]["diagnostic_sampling_limits"]["paper_only"] is True
    assert payload["active_learning_paper_trades"]["proposal_count"] == 1


def test_upward_trajectory_cli_writes_latest_copies(tmp_path: Path) -> None:
    multi = tmp_path / "multi.json"
    output = tmp_path / "out.json"
    latest = tmp_path / "latest.json"
    research_latest = tmp_path / "research_latest.json"
    multi.write_text(
        json.dumps({"status": "complete", "ranked_candidates": [_candidate("ok", 5.0, 2.0)]}),
        encoding="utf-8",
    )

    exit_code = report_tool.main(
        [
            "--report-date",
            "2026-05-18",
            "--multi-horizon-json",
            str(multi),
            "--output-json",
            str(output),
            "--latest-json",
            str(latest),
            "--research-latest-json",
            str(research_latest),
        ]
    )

    assert exit_code == 0
    for path in (output, latest, research_latest):
        assert json.loads(path.read_text(encoding="utf-8"))["artifact_type"] == "upward_trajectory_report"
