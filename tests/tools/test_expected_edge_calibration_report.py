from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import expected_edge_calibration_report as report_tool


def _rows(count: int, *, expected: float, realized: float, symbol: str = "AAPL") -> list[dict[str, object]]:
    return [
        {
            "ts": f"2026-05-05T14:{index:02d}:00Z",
            "symbol": symbol,
            "side": "buy",
            "session_regime": "midday",
            "expected_net_edge_bps": expected,
            "realized_net_edge_bps": realized,
            "slippage_bps": 1.0,
            "spread_bps": 4.0,
            "quote_age_ms": 250.0,
            "order_type": "limit",
        }
        for index in range(count)
    ]


def test_expected_edge_calibration_classifies_calibrated_edge() -> None:
    payload = report_tool.build_expected_edge_calibration_report(
        report_date="2026-05-05",
        fills=_rows(12, expected=8.0, realized=6.0),
        min_samples=10,
    )

    assert payload["status"] == "calibrated"
    assert payload["summary"]["capture_ratio"] == 0.75
    assert payload["execution_capture_diagnosis"]["attribution_counts"] == {
        "captured_expected_edge": 12
    }


def test_expected_edge_calibration_classifies_overestimated_edge() -> None:
    payload = report_tool.build_expected_edge_calibration_report(
        report_date="2026-05-05",
        fills=_rows(12, expected=30.0, realized=-2.0),
        min_samples=10,
    )

    assert payload["status"] == "overestimated"
    assert payload["recommended_next_action"] == "keep_tiny_sampling_and_recalibrate_signal"
    assert payload["execution_capture_diagnosis"]["worst_buckets"][0]["bucket"] == "edge_25_50"


def test_expected_edge_calibration_classifies_inverted_edge() -> None:
    fills = _rows(6, expected=2.0, realized=3.0) + _rows(6, expected=50.0, realized=-4.0)

    payload = report_tool.build_expected_edge_calibration_report(
        report_date="2026-05-05",
        fills=fills,
        min_samples=10,
    )

    assert payload["status"] == "inverted"
    assert payload["recommended_next_action"] == "pause_scaling_and_retrain_expected_edge_labels"
    assert payload["calibration_correction"]["expected_edge_multiplier"] == 0.0
    assert payload["calibration_correction"]["production_scaling_allowed"] is False


def test_expected_edge_calibration_reports_sell_exit_quality() -> None:
    buy_rows = _rows(8, expected=4.0, realized=2.0)
    sell_rows = [
        {
            **row,
            "side": "sell",
            "expected_net_edge_bps": 1.0,
            "realized_net_edge_bps": -3.0,
        }
        for row in _rows(8, expected=1.0, realized=-3.0, symbol="MSFT")
    ]

    payload = report_tool.build_expected_edge_calibration_report(
        report_date="2026-05-05",
        fills=[*buy_rows, *sell_rows],
        min_samples=10,
    )

    assert payload["exit_quality_diagnostics"]["status"] == "inverted"
    assert (
        payload["exit_quality_diagnostics"]["recommended_action"]
        == "review_exit_timing_and_order_type_before_scaling"
    )
    assert payload["calibration_correction"]["side_multipliers"]["sell"]["expected_edge_multiplier"] == 0.0


def test_expected_edge_calibration_cli_writes_latest(tmp_path: Path) -> None:
    fills = tmp_path / "fills.jsonl"
    out = tmp_path / "calibration.json"
    latest = tmp_path / "latest.json"
    fills.write_text(
        "\n".join(json.dumps(row) for row in _rows(3, expected=8.0, realized=1.0)) + "\n",
        encoding="utf-8",
    )

    rc = report_tool.main(
        [
            "--report-date",
            "2026-05-05",
            "--fills-jsonl",
            str(fills),
            "--min-samples",
            "10",
            "--output-json",
            str(out),
            "--latest-json",
            str(latest),
        ]
    )

    assert rc == 0
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "insufficient_samples"
    assert latest.is_file()
