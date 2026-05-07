from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import trading_day_report


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_trading_day_report_attributes_rejections_and_symbol_pnl():
    report = trading_day_report.build_trading_day_report(
        report_date="2026-05-05",
        order_intents=[
            {"ts": "2026-05-05T14:00:00Z", "symbol": "AAPL", "status": "SUBMITTED"},
            {"ts": "2026-05-04T14:00:00Z", "symbol": "AMZN", "status": "SUBMITTED"},
        ],
        fills=[
            {
                "ts": "2026-05-05T14:01:00Z",
                "symbol": "AAPL",
                "realized_pnl": "3.25",
                "realized_net_edge_bps": "1.5",
                "expected_net_edge_bps": "2.0",
                "slippage_bps": "0.5",
            },
            {
                "ts": "2026-05-05T14:01:30Z",
                "symbol": "AAPL",
                "realized_pnl": "1.75",
                "realized_net_edge_bps": "2.5",
                "expected_net_edge_bps": "4.0",
                "slippage_bps": "1.5",
            },
            {
                "ts": "2026-05-05T14:02:00Z",
                "symbol": "AMZN",
                "pnl": "-1.0",
                "realized_net_edge_bps": "-3.0",
                "expected_net_edge_bps": "1.0",
                "slippage_bps": "3.0",
            },
        ],
        shadow_rows=[
            {
                "ts": "2026-05-05T14:00:00Z",
                "symbol": "MSFT",
                "challenger_would_trade": True,
                "champion_would_trade": False,
            }
        ],
        gate_rows=[
            {"ts": "2026-05-05T14:00:01Z", "symbol": "AAPL", "status": "blocked", "reason": "spread_cap"},
            {"ts": "2026-05-05T14:00:02Z", "symbol": "AMZN", "action": "reject", "gate": "quote_age"},
        ],
        live_cost_model={"status": {"status": "ready"}},
        symbol_scorecard={"summary": {"allow": 2}, "symbols": []},
    )

    assert report["desired_trades"]["count"] == 1
    assert report["submitted_trades"]["count"] == 1
    assert report["rejected_trades"]["reasons"] == {"quote_age": 1, "spread_cap": 1}
    assert report["symbol_contribution"] == {"AAPL": 5.0, "AMZN": -1.0}
    assert report["symbol_realized_edge_bps"] == {"AAPL": 2.0, "AMZN": -3.0}
    assert report["symbol_expected_edge_bps"] == {"AAPL": 3.0, "AMZN": 1.0}
    assert report["symbol_slippage_bps"] == {"AAPL": 1.0, "AMZN": 3.0}
    assert report["edge_quality"]["mean_realized_edge_bps"] == 1 / 3
    assert report["symbol_trade_flow"]["AAPL"] == {
        "desired": 1,
        "submitted": 1,
        "rejected": 1,
        "fills": 2,
    }
    assert report["missed_opportunities"]["shadow_only_count"] == 1
    assert report["missed_opportunities"]["symbols"] == {"MSFT": 1}


def test_trading_day_report_cli_writes_latest_json_and_markdown(tmp_path: Path):
    intents = tmp_path / "intents.jsonl"
    fills = tmp_path / "fills.jsonl"
    gates = tmp_path / "gates.jsonl"
    out = tmp_path / "trading_day.json"
    latest = tmp_path / "latest.json"
    md = tmp_path / "latest.md"
    _write_jsonl(intents, [{"ts": "2026-05-05T14:00:00Z", "status": "FILLED"}])
    _write_jsonl(fills, [{"ts": "2026-05-05T14:00:02Z", "symbol": "AAPL", "pnl": 2.0}])
    _write_jsonl(gates, [{"ts": "2026-05-05T14:00:01Z", "status": "blocked", "reason": "spread_cap"}])

    rc = trading_day_report.main(
        [
            "--report-date",
            "2026-05-05",
            "--order-intents-jsonl",
            str(intents),
            "--fills-jsonl",
            str(fills),
            "--gate-jsonl",
            str(gates),
            "--output-json",
            str(out),
            "--latest-json",
            str(latest),
            "--latest-md",
            str(md),
        ]
    )

    assert rc == 0
    assert json.loads(out.read_text(encoding="utf-8"))["realized_fills"]["count"] == 1
    assert latest.is_file()
    assert "Trading Day 2026-05-05" in md.read_text(encoding="utf-8")
