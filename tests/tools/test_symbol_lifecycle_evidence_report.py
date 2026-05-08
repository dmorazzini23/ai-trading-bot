from __future__ import annotations

from ai_trading.tools import symbol_lifecycle_evidence_report as report_tool


def test_symbol_lifecycle_evidence_reports_gaps_by_symbol() -> None:
    payload = report_tool.build_symbol_lifecycle_evidence_report(
        report_date="2026-05-05",
        symbols=["AAPL", "MSFT"],
        candidates=[
            {"ts": "2026-05-05T14:00:00Z", "symbol": "AAPL", "status": "accepted"},
            {"ts": "2026-05-05T14:00:00Z", "symbol": "MSFT", "status": "rejected"},
        ],
        order_intents=[
            {"ts": "2026-05-05T14:01:00Z", "decision_id": "a", "symbol": "AAPL", "status": "submitted"}
        ],
        fills=[
            {"ts": "2026-05-05T14:02:00Z", "decision_id": "a", "symbol": "AAPL", "status": "filled"}
        ],
    )

    by_symbol = {row["symbol"]: row for row in payload["symbols"]}
    assert payload["status"] == "gaps"
    assert by_symbol["AAPL"]["lifecycle_status"] == "evidence_ready"
    assert by_symbol["MSFT"]["lifecycle_status"] == "no_submitted_orders"


def test_symbol_lifecycle_evidence_complete_when_each_symbol_has_fill() -> None:
    payload = report_tool.build_symbol_lifecycle_evidence_report(
        report_date="2026-05-05",
        symbols=["AAPL"],
        candidates=[{"symbol": "AAPL", "status": "accepted"}],
        order_intents=[{"symbol": "AAPL", "status": "submitted"}],
        fills=[{"symbol": "AAPL", "status": "filled"}],
    )

    assert payload["status"] == "complete"
    assert payload["summary"]["evidence_ready"] == 1
