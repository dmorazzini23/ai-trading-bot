from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_trading.tools import runtime_performance_report as rpt


def test_default_trade_history_path_uses_runtime_tca_records() -> None:
    assert rpt._DEFAULT_TRADE_HISTORY_PATH == "runtime/tca_records.jsonl"


def test_build_report_summarizes_trade_and_gate_data(tmp_path: Path) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"

    trade_history_path.write_text(
        json.dumps(
            [
                {"symbol": "AAPL", "side": "buy", "pnl": 10},
                {"symbol": "MSFT", "side": "sell", "pnl": -4},
                {"symbol": "NVDA", "side": "buy", "pnl": 6},
            ]
        ),
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps(
            {
                "total_records": 20,
                "total_accepted_records": 5,
                "total_rejected_records": 15,
                "total_expected_net_edge_bps": -12.5,
                "gate_totals": {"COST_GATE": 11, "STOP_LOCK": 3, "OK_TRADE": 5},
            }
        ),
        encoding="utf-8",
    )

    report = rpt.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
    )

    trade = report["trade_history"]
    gate = report["gate_effectiveness"]
    assert trade["pnl_available"] is True
    assert trade["pnl_sum"] == 12
    assert trade["win_rate"] == 2 / 3
    assert gate["acceptance_rate"] == 0.25
    assert gate["top_gates"][0]["gate"] == "COST_GATE"


def test_format_text_report_handles_missing_inputs(tmp_path: Path) -> None:
    report = rpt.build_report(
        trade_history_path=tmp_path / "missing_trade_history.json",
        gate_summary_path=tmp_path / "missing_gate_summary.json",
    )

    output = rpt.format_text_report(report)

    assert "exists=False" in output
    assert "Realized pnl: unavailable" in output


def test_build_report_reconstructs_fifo_realized_pnl(tmp_path: Path) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"

    trade_history_path.write_text(
        json.dumps(
            [
                {
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": 10,
                    "entry_price": 100.0,
                    "entry_time": "2026-01-02T14:30:00+00:00",
                    "fee_bps": 10,
                    "slippage_bps": 2,
                    "strategy": "alpha",
                },
                {
                    "symbol": "AAPL",
                    "side": "sell",
                    "qty": 10,
                    "entry_price": 103.0,
                    "entry_time": "2026-01-02T20:30:00+00:00",
                    "fee_bps": 10,
                    "slippage_bps": 1,
                    "strategy": "alpha",
                },
                {
                    "symbol": "MSFT",
                    "side": "sell",
                    "qty": 5,
                    "entry_price": 200.0,
                    "entry_time": "2026-01-02T15:00:00+00:00",
                    "fee_bps": 10,
                    "slippage_bps": 3,
                    "strategy": "beta",
                },
                {
                    "symbol": "MSFT",
                    "side": "buy",
                    "qty": 5,
                    "entry_price": 210.0,
                    "entry_time": "2026-01-03T16:00:00+00:00",
                    "fee_bps": 10,
                    "slippage_bps": 2,
                    "strategy": "beta",
                },
            ]
        ),
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps({"total_records": 0, "total_accepted_records": 0, "total_rejected_records": 0}),
        encoding="utf-8",
    )

    report = rpt.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
    )
    trade = report["trade_history"]

    assert trade["pnl_available"] is True
    assert trade["pnl_source"] == "fifo_reconstructed_from_fills"
    assert trade["pnl_records"] == 2
    assert trade["open_lot_count"] == 0
    assert trade["pnl_sum"] == pytest.approx(-24.893, rel=1e-6)
    assert trade["top_loss_drivers"]["symbols"][0]["name"] == "MSFT"
    assert trade["daily_expectancy"][-1]["date"] == "2026-01-03"


def test_evaluate_go_no_go_uses_thresholds() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "closed_trades": 40,
            "profit_factor": 1.35,
            "win_rate": 0.58,
            "pnl_sum": 245.0,
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.18,
            "total_expected_net_edge_bps": 22.0,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "min_closed_trades": 20,
            "min_profit_factor": 1.1,
            "min_win_rate": 0.5,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.1,
            "min_expected_net_edge_bps": 0.0,
            "require_pnl_available": True,
            "require_gate_valid": True,
        },
    )

    assert decision["gate_passed"] is True
    assert decision["failed_checks"] == []


def test_evaluate_go_no_go_fails_when_metrics_are_weak() -> None:
    report = {
        "trade_history": {
            "pnl_available": False,
            "closed_trades": 3,
            "profit_factor": 0.7,
            "win_rate": 0.2,
            "pnl_sum": -50.0,
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.01,
            "total_expected_net_edge_bps": -120.0,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "min_closed_trades": 20,
            "min_profit_factor": 1.1,
            "min_win_rate": 0.5,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.05,
            "min_expected_net_edge_bps": -10.0,
            "require_pnl_available": True,
            "require_gate_valid": True,
        },
    )

    assert decision["gate_passed"] is False
    assert "pnl_available" in decision["failed_checks"]
    assert "profit_factor" in decision["failed_checks"]


def test_main_fail_on_no_go_returns_nonzero(tmp_path: Path) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"
    trade_history_path.write_text(
        json.dumps([{"symbol": "AAPL", "side": "buy", "pnl": -5}]),
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps(
            {
                "total_records": 10,
                "total_accepted_records": 0,
                "total_rejected_records": 10,
                "total_expected_net_edge_bps": -100,
            }
        ),
        encoding="utf-8",
    )

    exit_code = rpt.main(
        [
            "--trade-history",
            str(trade_history_path),
            "--gate-summary",
            str(gate_summary_path),
            "--go-no-go",
            "--fail-on-no-go",
            "--min-closed-trades",
            "10",
            "--min-profit-factor",
            "1.2",
        ]
    )

    assert exit_code == 2
