from __future__ import annotations

from ai_trading.tools import decision_receipts_report as report_tool


def test_decision_receipts_tie_decision_to_order_and_fill() -> None:
    payload = report_tool.build_decision_receipts_report(
        report_date="2026-05-05",
        decisions=[
            {
                "decision_id": "d1",
                "ts": "2026-05-05T14:00:00Z",
                "symbol": "AAPL",
                "status": "accepted",
                "expected_net_edge_bps": 5.0,
            }
        ],
        order_intents=[{"decision_id": "d1", "symbol": "AAPL", "status": "submitted"}],
        fills=[{"decision_id": "d1", "symbol": "AAPL", "status": "filled", "realized_net_edge_bps": 4.0}],
    )

    assert payload["status"] == "complete"
    assert payload["receipts"][0]["receipt_complete"] is True
    assert payload["receipts"][0]["realized_net_edge_bps"] == 4.0


def test_decision_receipts_flags_missing_terminal_link() -> None:
    payload = report_tool.build_decision_receipts_report(
        report_date="2026-05-05",
        decisions=[{"decision_id": "d1", "symbol": "AAPL", "status": "accepted"}],
        order_intents=[{"decision_id": "d1", "symbol": "AAPL", "status": "submitted"}],
    )

    assert payload["status"] == "gaps"
    assert payload["summary"]["incomplete"] == 1


def test_decision_receipts_complete_nested_rejected_decision_record() -> None:
    payload = report_tool.build_decision_receipts_report(
        report_date="2026-05-12",
        decisions=[
            {
                "bar_ts": "2026-05-12T19:57:00+00:00",
                "decision_journal": {
                    "accepted": False,
                    "decision_trace_id": "trace-a",
                    "symbol": "AAPL",
                    "reasons": ["PAPER_SAMPLING_SHORT_BLOCK"],
                    "risk_decision": {"gates": ["RISK_FACTOR_SOFT_THROTTLE"]},
                },
            }
        ],
    )

    assert payload["status"] == "complete"
    assert payload["summary"]["complete"] == 1
    receipt = payload["receipts"][0]
    assert receipt["decision_id"] == "trace-a"
    assert receipt["symbol"] == "AAPL"
    assert receipt["receipt_complete"] is True
    assert "PAPER_SAMPLING_SHORT_BLOCK" in receipt["reasons"]


def test_decision_receipts_complete_aggregate_gate_summary_rows() -> None:
    payload = report_tool.build_decision_receipts_report(
        report_date="2026-05-12",
        decisions=[
            {
                "ts": "2026-05-12T20:00:01Z",
                "records_total": 2,
                "accepted_records": 0,
                "rejected_records": 2,
                "gate_counts": {"NET_EDGE_FLOOR_GATE": 1, "RISK_FACTOR_SOFT_THROTTLE": 1},
                "symbol_attribution": {"AAPL": {}, "AMZN": {}},
            }
        ],
    )

    assert payload["status"] == "complete"
    receipt = payload["receipts"][0]
    assert receipt["receipt_granularity"] == "aggregate_gate_summary"
    assert receipt["records_total"] == 2
    assert receipt["rejected_records"] == 2


def test_decision_receipts_disambiguate_children_with_same_correlation() -> None:
    decisions = [
        {
            "correlation_id": "opp-shared",
            "decision_trace_id": "trace-shared",
            "symbol": "AAPL",
            "status": "accepted",
            "order": {"client_order_id": child, "correlation_id": "opp-shared"},
        }
        for child in ("child-1", "child-2")
    ]
    payload = report_tool.build_decision_receipts_report(
        report_date="2026-07-21",
        decisions=decisions,
        order_intents=[
            {
                "correlation_id": "opp-shared",
                "decision_trace_id": "trace-shared",
                "client_order_id": child,
                "status": "submitted",
            }
            for child in ("child-1", "child-2")
        ],
        fills=[
            {
                "correlation_id": "opp-shared",
                "decision_trace_id": "trace-shared",
                "client_order_id": "child-1",
                "status": "filled",
                "realized_net_edge_bps": 1.0,
            },
            {
                "correlation_id": "opp-shared",
                "decision_trace_id": "trace-shared",
                "client_order_id": "child-2",
                "status": "filled",
                "realized_net_edge_bps": 2.0,
            },
        ],
    )

    assert payload["status"] == "complete"
    receipts = {
        receipt["realized_net_edge_bps"]: receipt
        for receipt in payload["receipts"]
    }
    assert set(receipts) == {1.0, 2.0}
    assert all(
        receipt["link_methods"]["fill"] == "correlation_and_order_id"
        for receipt in receipts.values()
    )


def test_decision_receipts_keep_controlled_skip_joinable_without_fill_evidence() -> None:
    payload = report_tool.build_decision_receipts_report(
        report_date="2026-07-21",
        decisions=[
            {
                "correlation_id": "opp-controlled-skip",
                "symbol": "MSFT",
                "status": "skipped",
                "reasons": ["METRICS_IMPROVEMENT_CONTROLLED_SKIP"],
            }
        ],
        fills=[
            {
                "correlation_id": "opp-controlled-skip",
                "evidence_type": "shadow_counterfactual",
                "evidence_partition": "shadow",
                "fill_based_evidence": False,
                "executed": False,
                "realized_net_edge_bps": 5.0,
            }
        ],
    )

    receipt = payload["receipts"][0]
    assert receipt["receipt_complete"] is True
    assert receipt["correlation_id"] == "opp-controlled-skip"
    assert receipt["fill_present"] is False
    assert receipt["fill_based_evidence"] is False
    assert receipt["promotion_eligible"] is False
    assert payload["summary"]["non_fill_rows_excluded"] == 1
