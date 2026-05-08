from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import post_trade_surveillance_report as surveillance


def test_post_trade_surveillance_flags_required_failure_modes() -> None:
    decisions = [
        {"intent_id": "intent-dup", "symbol": "AAPL", "status": "rejected"},
        {"intent_id": "intent-dup", "symbol": "AAPL", "status": "accepted"},
    ]
    orders = [
        {"client_order_id": "short-1", "symbol": "MSFT", "side": "sell_short", "qty": 10, "filled_qty": 3},
        {"client_order_id": "missing-oms", "symbol": "NVDA", "status": "filled", "qty": 5, "filled_qty": 5},
    ]
    fills = [
        {
            "client_order_id": "fill-1",
            "symbol": "TSLA",
            "expected_net_edge_bps": 8.0,
            "realized_net_edge_bps": -2.0,
            "slippage_bps": 40.0,
        },
        {
            "client_order_id": "close-1",
            "symbol": "AMZN",
            "action": "close",
            "remaining_position": 1,
        },
    ]
    oms_events = [{"client_order_id": "short-1", "event_type": "SUBMITTED"}]

    payload = surveillance.build_post_trade_surveillance_report(
        report_date="2026-05-05",
        decisions=decisions,
        orders=orders,
        fills=fills,
        oms_events=oms_events,
        max_slippage_bps=25.0,
    )

    categories = payload["summary"]["category_counts"]
    assert payload["status"] == "control_breach"
    assert categories["reject"] == 1
    assert categories["sell_short_attempt"] == 1
    assert categories["duplicate_intent"] == 1
    assert categories["partial_fill_issue"] == 1
    assert categories["adverse_selection"] == 1
    assert categories["slippage_breach"] == 1
    assert categories["non_flat_close"] == 1
    assert categories["oms_mismatch"] >= 1


def test_post_trade_surveillance_cli_writes_latest(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.jsonl"
    output = tmp_path / "surveillance.json"
    latest = tmp_path / "latest.json"
    decisions.write_text(
        json.dumps({"ts": "2026-05-05T14:00:00Z", "intent_id": "a", "symbol": "AAPL", "status": "rejected"}) + "\n",
        encoding="utf-8",
    )

    rc = surveillance.main(
        [
            "--report-date",
            "2026-05-05",
            "--decisions-jsonl",
            str(decisions),
            "--output-json",
            str(output),
            "--latest-json",
            str(latest),
        ]
    )

    assert rc == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "watchlist"
    assert latest.is_file()
