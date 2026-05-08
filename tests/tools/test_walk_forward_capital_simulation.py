from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import walk_forward_capital_simulation as sim_tool


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) if not isinstance(row, str) else row for row in rows) + "\n",
        encoding="utf-8",
    )


def test_walk_forward_sim_applies_launch_profile_constraints() -> None:
    rows = [
        {
            "ts": "2026-05-01T14:00:00Z",
            "symbol": "AAPL",
            "side": "buy",
            "notional": 500.0,
            "realized_return_bps": 100.0,
        },
        {
            "ts": "2026-05-01T14:01:00Z",
            "symbol": "MSFT",
            "side": "buy",
            "notional": 50.0,
            "realized_return_bps": 100.0,
        },
        {
            "ts": "2026-05-01T14:02:00Z",
            "symbol": "AMZN",
            "side": "sell_short",
            "notional": 50.0,
            "realized_return_bps": 100.0,
        },
    ]

    report = sim_tool.build_walk_forward_capital_simulation(
        rows=rows,
        initial_capital=10_000.0,
        launch_profile_name="live_canary",
    )

    assert report["live_enabled"] is False
    assert report["mode"] == "research_shadow"
    assert report["summary"]["accepted_orders"] == 1
    assert report["summary"]["scaled_orders"] == 1
    assert report["summary"]["ending_capital"] == 10_001.0
    reasons = report["constraints"]["blocked_reason_counts"]
    assert reasons["symbol_not_allowed_by_launch_profile"] == 1
    assert reasons["shorts_not_allowed_by_launch_profile"] == 1


def test_walk_forward_sim_cli_writes_artifact(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    output = tmp_path / "capital_sim.json"
    _write_jsonl(
        events,
        [
            {
                "ts": "2026-05-01T14:00:00Z",
                "symbol": "AAPL",
                "side": "buy",
                "notional": 100.0,
                "realized_pnl": 2.0,
            }
        ],
    )

    rc = sim_tool.main(
        [
            "--events-jsonl",
            str(events),
            "--output-json",
            str(output),
            "--initial-capital",
            "10000",
            "--launch-profile",
            "live_canary",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "walk_forward_capital_simulation"
    assert payload["live_enabled"] is False
    assert payload["sources"]["events_jsonl"]["valid_rows"] == 1

