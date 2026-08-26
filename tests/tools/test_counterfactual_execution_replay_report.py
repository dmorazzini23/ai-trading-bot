from __future__ import annotations

import json
from pathlib import Path

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
    spread = next(
        row
        for row in payload["filter_ablations"]["rows"]
        if row["rejection_reason"] == "spread"
    )
    assert spread["disposition"] == "collect_more_shadow_evidence"
    assert spread["runtime_filter_change_authorized"] is False


def test_counterfactual_replay_marks_only_strong_filters_for_shadow_ablation() -> None:
    decisions = [
        {
            "decision_id": f"positive-{index}",
            "status": "rejected",
            "reason": "EDGE_FLOOR",
            "counterfactual_net_edge_bps": edge,
        }
        for index, edge in enumerate((5.0, 4.0, 3.0, 2.0, -1.0))
    ] + [
        {
            "decision_id": f"negative-{index}",
            "status": "rejected",
            "reason": "STALE_QUOTE",
            "counterfactual_net_edge_bps": edge,
        }
        for index, edge in enumerate((-5.0, -4.0, -3.0, 1.0, 1.0))
    ]

    payload = report_tool.build_counterfactual_execution_replay_report(
        report_date="2026-05-05",
        decisions=decisions,
        min_counterfactual_samples=10,
        min_filter_ablation_samples=5,
    )

    rows = {
        row["rejection_reason"]: row
        for row in payload["filter_ablations"]["rows"]
    }
    assert rows["EDGE_FLOOR"]["disposition"] == "eligible_for_bounded_shadow_ablation"
    assert rows["STALE_QUOTE"]["disposition"] == "keep_filter"
    assert payload["filter_ablations"]["research_only"] is True
    assert payload["filter_ablations"]["runtime_filter_change_authorized"] is False


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


def test_counterfactual_replay_consumes_canonical_shadow_markout_schema(
    tmp_path: Path,
) -> None:
    outcome = {
        "outcome_id": "corr-a:shadow_counterfactual:h1:v1",
        "correlation_id": "corr-a",
        "symbol": "AAPL",
        "horizon_bars": 1,
        "source_timestamp": "2026-05-05T15:00:00Z",
        "label_status": "resolved",
        "evidence_type": "shadow_counterfactual",
        "evidence_partition": "shadow",
        "research_only": True,
        "executed": False,
        "net_markout_bps": 7.5,
        "fill_based_evidence": False,
        "promotion_eligible": False,
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
    }
    path = tmp_path / "markouts.json"
    path.write_text(json.dumps({"outcomes": [outcome]}), encoding="utf-8")

    payload = report_tool.build_counterfactual_execution_replay_report(
        report_date="2026-05-05",
        decisions=[
            {
                "decision_id": "decision-a",
                "correlation_id": "corr-a",
                "symbol": "AAPL",
                "status": "rejected",
            }
        ],
        outcomes=report_tool._read_outcomes(path, report_date="2026-05-05"),
        min_counterfactual_samples=1,
    )

    assert payload["summary"]["rejected_counterfactual_samples"] == 1
    assert payload["summary"]["hypothetical_outcome_samples"] == 1
    assert payload["summary"]["rejected_decisions_without_linked_outcomes"] == 0
    assert payload["missed_positive_decisions"][0]["correlation_id"] == "corr-a"
    assert payload["research_evidence"]["fill_based_evidence"] is False


def test_counterfactual_replay_rejects_shadow_rows_with_authority() -> None:
    payload = report_tool.build_counterfactual_execution_replay_report(
        report_date="2026-05-05",
        decisions=[{"correlation_id": "corr-a", "status": "rejected"}],
        outcomes=[
            {
                "correlation_id": "corr-a",
                "label_status": "resolved",
                "evidence_type": "shadow_counterfactual",
                "evidence_partition": "shadow",
                "research_only": True,
                "executed": False,
                "net_markout_bps": 100.0,
                "fill_based_evidence": False,
                "promotion_eligible": True,
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
            }
        ],
        min_counterfactual_samples=1,
    )

    assert payload["summary"]["rejected_counterfactual_samples"] == 0


def test_counterfactual_replay_understands_canonical_decision_record() -> None:
    payload = report_tool.build_counterfactual_execution_replay_report(
        report_date="2026-05-05",
        decisions=[
            {
                "correlation_id": "corr-decision-record",
                "symbol": "MSFT",
                "bar_ts": "2026-05-05T15:00:00Z",
                "decision_journal": {
                    "accepted": False,
                    "submitted": False,
                    "decision_ts": "2026-05-05T15:00:00Z",
                },
                "metrics": {"terminal_reason": "NET_EDGE_FLOOR"},
            }
        ],
        outcomes=[
            {
                "correlation_id": "corr-decision-record",
                "symbol": "MSFT",
                "horizon_bars": 1,
                "label_status": "resolved",
                "evidence_type": "shadow_counterfactual",
                "evidence_partition": "shadow",
                "research_only": True,
                "executed": False,
                "net_markout_bps": 4.0,
                "fill_based_evidence": False,
                "promotion_eligible": False,
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
            }
        ],
        min_counterfactual_samples=1,
    )

    assert payload["summary"]["rejected_decisions"] == 1
    assert payload["summary"]["rejected_counterfactual_samples"] == 1
    assert payload["rejection_reasons"] == {"NET_EDGE_FLOOR": 1}


def test_counterfactual_replay_uses_symbol_timestamp_when_identity_is_missing() -> None:
    payload = report_tool.build_counterfactual_execution_replay_report(
        report_date="2026-05-05",
        decisions=[
            {
                "symbol": "MSFT",
                "bar_ts": "2026-05-05T15:00:00Z",
                "decision_journal": {"accepted": False},
                "metrics": {"terminal_reason": "EDGE_FLOOR_GATE"},
            }
        ],
        outcomes=[
            {
                "correlation_id": "opp-msft",
                "symbol": "MSFT",
                "decision_timestamp": "2026-05-05T15:00:00+00:00",
                "horizon_bars": 1,
                "label_status": "resolved",
                "evidence_type": "shadow_counterfactual",
                "evidence_partition": "shadow",
                "research_only": True,
                "net_markout_bps": 3.0,
                "fill_based_evidence": False,
                "promotion_eligible": False,
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
            }
        ],
        min_counterfactual_samples=1,
    )

    assert payload["summary"]["rejected_counterfactual_samples"] == 1
    assert payload["summary"]["outcome_join_methods"] == {
        "symbol_decision_timestamp": 1
    }
    assert payload["missed_positive_decisions"][0]["correlation_id"] == "opp-msft"
    assert payload["missed_positive_decisions"][0]["reason"] == "EDGE_FLOOR_GATE"
