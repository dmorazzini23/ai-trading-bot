from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import execution_capture_classification_report as report_tool


def test_execution_capture_classifies_dominant_failures() -> None:
    fills = [
        {"symbol": "AAPL", "expected_net_edge_bps": 12.0, "realized_net_edge_bps": -1.0},
        {"symbol": "MSFT", "expected_net_edge_bps": 8.0, "realized_net_edge_bps": 7.0},
        {
            "symbol": "NVDA",
            "expected_net_edge_bps": 8.0,
            "realized_net_edge_bps": 1.0,
            "quote_age_ms": 9000,
        },
    ]

    payload = report_tool.build_execution_capture_classification_report(
        report_date="2026-05-05",
        fills=fills,
        min_samples=3,
        min_capture_ratio=0.50,
    )

    assert payload["status"] == "needs_review"
    assert payload["summary"]["classification_counts"] == {
        "adverse_selection": 1,
        "captured_expected_edge": 1,
        "stale_quote": 1,
    }


def test_execution_capture_cli_writes_latest(tmp_path: Path) -> None:
    fills = tmp_path / "fills.jsonl"
    out = tmp_path / "capture.json"
    latest = tmp_path / "latest.json"
    fills.write_text(
        json.dumps({"ts": "2026-05-05T14:00:00Z", "symbol": "AAPL", "expected_net_edge_bps": 5, "realized_net_edge_bps": 4}) + "\n",
        encoding="utf-8",
    )

    rc = report_tool.main([
        "--report-date", "2026-05-05",
        "--fills-jsonl", str(fills),
        "--output-json", str(out),
        "--latest-json", str(latest),
    ])

    assert rc == 0
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "insufficient_samples"
    assert latest.is_file()
