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
