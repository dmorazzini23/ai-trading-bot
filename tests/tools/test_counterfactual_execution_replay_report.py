from __future__ import annotations

from ai_trading.tools import counterfactual_execution_replay_report as report_tool


def test_counterfactual_replay_flags_missed_positive_edge() -> None:
    decisions = [
        {"decision_id": "a", "symbol": "AAPL", "status": "rejected", "reason": "spread", "counterfactual_net_edge_bps": 20.0},
        {"decision_id": "b", "symbol": "MSFT", "status": "rejected", "reason": "spread", "counterfactual_net_edge_bps": 10.0},
        {"decision_id": "c", "symbol": "NVDA", "status": "rejected", "reason": "risk", "counterfactual_net_edge_bps": -4.0},
    ]

    payload = report_tool.build_counterfactual_execution_replay_report(
        report_date="2026-05-05",
        decisions=decisions,
        min_counterfactual_samples=3,
        max_missed_edge_bps=15.0,
    )

    assert payload["status"] == "needs_review"
    assert payload["counterfactual"]["passed"] is False
    assert payload["summary"]["missed_positive_edge_bps"] == 30.0
    assert payload["rejection_reasons"] == {"risk": 1, "spread": 2}


def test_counterfactual_replay_passes_when_rejections_avoid_negative_edge() -> None:
    decisions = [
        {"decision_id": "a", "symbol": "AAPL", "status": "rejected", "counterfactual_net_edge_bps": -2.0},
        {"decision_id": "b", "symbol": "MSFT", "status": "rejected", "counterfactual_net_edge_bps": -3.0},
    ]

    payload = report_tool.build_counterfactual_execution_replay_report(
        report_date="2026-05-05",
        decisions=decisions,
        min_counterfactual_samples=2,
    )

    assert payload["status"] == "passed"
    assert payload["summary"]["avoided_negative_count"] == 2


def test_counterfactual_replay_consumes_only_hypothetical_resolved_outcomes() -> None:
    decisions = [
        {
            "decision_id": "decision-a",
            "prediction_id": "prediction-a",
            "symbol": "AAPL",
            "status": "rejected",
            "reason": "spread",
        },
        {
            "decision_id": "decision-b",
            "prediction_id": "prediction-b",
            "symbol": "MSFT",
            "status": "rejected",
            "reason": "risk",
        },
        {
            "decision_id": "decision-c",
            "prediction_id": "prediction-c",
            "symbol": "NVDA",
            "status": "accepted",
        },
    ]
    outcomes = [
        {
            "prediction_id": "prediction-a",
            "model_role": "challenger",
            "horizon_bars": 5,
            "primary_horizon": True,
            "evidence_type": "hypothetical",
            "executed": False,
            "counterfactual_net_edge_bps": 12.0,
            "model_id": "model-a",
            "feature_version": "features-v2",
        },
        {
            "prediction_id": "prediction-a",
            "model_role": "challenger",
            "horizon_bars": 5,
            "evidence_type": "executed",
            "executed": True,
            "counterfactual_net_edge_bps": 999.0,
        },
        {
            "prediction_id": "prediction-b",
            "model_role": "challenger",
            "horizon_bars": 5,
            "primary_horizon": True,
            "evidence_type": "hypothetical",
            "executed": False,
            "counterfactual_net_edge_bps": -3.0,
        },
        {
            "prediction_id": "prediction-c",
            "model_role": "champion",
            "horizon_bars": 5,
            "primary_horizon": True,
            "evidence_type": "hypothetical",
            "executed": False,
            "counterfactual_net_edge_bps": 50.0,
        },
    ]

    payload = report_tool.build_counterfactual_execution_replay_report(
        report_date="2026-05-05",
        decisions=decisions,
        outcomes=outcomes,
        min_counterfactual_samples=2,
        max_missed_edge_bps=20.0,
    )

    assert payload["status"] == "passed"
    assert payload["summary"]["accepted_realized_samples"] == 0
    assert payload["summary"]["rejected_counterfactual_samples"] == 2
    assert payload["summary"]["hypothetical_outcome_samples"] == 2
    assert payload["summary"]["executed_outcome_rows_ignored"] == 1
    assert payload["summary"]["missed_positive_edge_bps"] == 12.0
    missed = payload["missed_positive_decisions"][0]
    assert missed["prediction_id"] == "prediction-a"
    assert missed["model_id"] == "model-a"
    assert missed["feature_version"] == "features-v2"
    assert missed["evidence_type"] == "hypothetical"
    assert payload["promotion_authority"] is False
    assert payload["live_money_authority"] is False
