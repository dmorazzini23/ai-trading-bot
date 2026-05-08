from __future__ import annotations

from ai_trading.tools import portfolio_edge_control_report as report_tool


def test_portfolio_edge_control_flags_symbol_concentration() -> None:
    fills = [
        {"symbol": "AAPL", "expected_net_edge_bps": 20.0, "realized_net_edge_bps": 8.0},
        {"symbol": "AAPL", "expected_net_edge_bps": 20.0, "realized_net_edge_bps": 8.0},
        {"symbol": "MSFT", "expected_net_edge_bps": 5.0, "realized_net_edge_bps": 4.0},
    ]

    payload = report_tool.build_portfolio_edge_control_report(
        report_date="2026-05-05",
        fills=fills,
        min_samples=3,
        max_symbol_edge_share=0.60,
    )

    assert payload["status"] == "control_breach"
    assert "symbol_edge_concentration" in payload["controls"]["breaches"]
    assert payload["summary"]["dominant_symbol"] == "AAPL"


def test_portfolio_edge_control_uses_calibration_fallback() -> None:
    calibration = {
        "bucketed_by_symbol": {
            "AAPL": {
                "count": 2,
                "mean_expected_net_edge_bps": 4.0,
                "mean_realized_net_edge_bps": 3.0,
            }
        }
    }

    payload = report_tool.build_portfolio_edge_control_report(
        report_date="2026-05-05",
        expected_edge_calibration=calibration,
        min_samples=2,
    )

    assert payload["status"] == "control_breach"
    assert payload["source"] == "expected_edge_calibration"
    assert payload["summary"]["portfolio_capture_ratio"] == 0.75
