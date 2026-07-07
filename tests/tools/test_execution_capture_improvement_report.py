from __future__ import annotations

import json

from ai_trading.tools.execution_capture_improvement_report import (
    build_execution_capture_improvement_report,
    main,
)


def _bad_fill(symbol: str = "AAPL") -> dict[str, object]:
    return {
        "ts": "2026-07-01T15:30:00Z",
        "symbol": symbol,
        "side": "buy",
        "session": "rth",
        "order_type": "market",
        "regime": "normal",
        "expected_net_edge_bps": 4.0,
        "realized_net_edge_bps": -2.0,
        "spread_bps": 8.0,
        "quote_age_ms": 250.0,
        "qty": 1,
    }


def test_execution_capture_improvement_builds_bad_buckets_and_training_labels() -> None:
    report = build_execution_capture_improvement_report(
        report_date="2026-07-01",
        fills=[_bad_fill(), _bad_fill(), _bad_fill("MSFT")],
        min_bucket_samples=2,
        min_capture_ratio=0.35,
    )

    assert report["artifact_type"] == "execution_capture_improvement_report"
    assert report["status"] == "needs_review"
    assert report["runtime_authority"] is False
    assert report["promotion_authority"] is False
    assert report["live_money_authority"] is False
    assert report["summary"]["adverse_selection_count"] == 3
    assert report["bad_buckets"]["by_symbol"][0]["bucket_key"] == "AAPL"
    assert report["edge_haircuts"]["by_symbol"]["AAPL"]["action"] == "shadow"
    assert report["edge_haircuts"]["by_symbol"]["AAPL"]["qty_scale"] == 0.0
    assert "realized_net_edge_after_cost" in report["training_labels"]["recommended_targets"]
    assert report["training_labels"]["runtime_authority"] is False


def test_execution_capture_improvement_cli_writes_latest(tmp_path) -> None:
    fills = tmp_path / "fills.jsonl"
    output = tmp_path / "execution_capture_improvement.json"
    latest = tmp_path / "execution_capture_improvement_latest.json"
    fills.write_text(
        "\n".join(json.dumps(_bad_fill()) for _ in range(2)) + "\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "--report-date",
            "2026-07-01",
            "--fills-jsonl",
            str(fills),
            "--min-bucket-samples",
            "2",
            "--output-json",
            str(output),
            "--latest-json",
            str(latest),
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    latest_payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "needs_review"
    assert latest_payload["edge_haircuts"]["runtime_authority"] is False
