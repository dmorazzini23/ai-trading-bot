from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ai_trading.core import bot_engine


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def test_runtime_truth_report_writes_daily_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    now = datetime(2026, 3, 10, 22, 30, tzinfo=UTC)
    data_root = tmp_path / "data-root"
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_root))
    monkeypatch.setenv("AI_TRADING_RUNTIME_TRUTH_REPORT_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_REQUIRE_GATE_VALID", "1")
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_REQUIRE_PNL_AVAILABLE", "1")

    trade_history = data_root / "runtime" / "tca_records.jsonl"
    _write_jsonl(
        trade_history,
        [
            {
                "symbol": "AAPL",
                "side": "buy",
                "qty": 1,
                "entry_price": 100.0,
                "exit_price": 101.0,
                "pnl": 1.0,
                "timestamp": "2026-03-10T21:00:00+00:00",
            }
        ],
    )
    gate_summary = data_root / "runtime" / "gate_effectiveness_summary.json"
    gate_summary.parent.mkdir(parents=True, exist_ok=True)
    gate_summary.write_text(
        json.dumps(
            {
                "total_records": 10,
                "total_accepted_records": 3,
                "total_rejected_records": 7,
                "total_expected_net_edge_bps": -12.5,
                "gate_totals": {"COST_GATE": 7},
                "gate_attribution": {},
                "symbol_attribution": {},
                "regime_attribution": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    state = bot_engine.BotState()
    bot_engine._run_runtime_truth_report(state, now=now, market_open_now=False)

    report_path = data_root / "runtime" / "reports" / "runtime_performance_20260310.json"
    latest_path = data_root / "runtime" / "reports" / "runtime_performance_latest.json"
    assert report_path.exists()
    assert latest_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["date"] == "2026-03-10"
    assert "go_no_go" in payload
    assert payload["paths"]["trade_history"] == str(trade_history)
    assert payload["paths"]["gate_summary"] == str(gate_summary)
    assert state.last_runtime_truth_report_date == now.date()


def test_runtime_truth_report_skips_when_already_written_for_day(
    tmp_path: Path,
    monkeypatch,
) -> None:
    now = datetime(2026, 3, 10, 22, 30, tzinfo=UTC)
    data_root = tmp_path / "data-root"
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_root))
    monkeypatch.setenv("AI_TRADING_RUNTIME_TRUTH_REPORT_ENABLED", "1")

    state = bot_engine.BotState()
    state.last_runtime_truth_report_date = now.date()
    bot_engine._run_runtime_truth_report(state, now=now, market_open_now=False)

    report_path = data_root / "runtime" / "reports" / "runtime_performance_20260310.json"
    assert not report_path.exists()
