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
