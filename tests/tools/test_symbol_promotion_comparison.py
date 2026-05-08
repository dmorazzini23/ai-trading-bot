from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ai_trading.tools import symbol_promotion_comparison


def _sample_inputs() -> dict[str, object]:
    return {
        "live_cost_model": {
            "by_symbol_side_session": [
                {
                    "symbol": "AAPL",
                    "sample_count": 30,
                    "mean_total_cost_bps": 4.0,
                    "p90_total_cost_bps": 8.0,
                    "p90_spread_bps": 9.0,
                },
                {
                    "symbol": "AMZN",
                    "sample_count": 30,
                    "mean_total_cost_bps": 5.0,
                    "p90_total_cost_bps": 9.0,
                    "p90_spread_bps": 10.0,
                },
                {
                    "symbol": "MSFT",
                    "sample_count": 30,
                    "mean_total_cost_bps": 22.0,
                    "p90_total_cost_bps": 40.0,
                    "p90_spread_bps": 65.0,
                },
            ]
        },
        "replay_report": {
            "replay_symbol_summary": {
                "AAPL": {
                    "sample_count": 30,
                    "net_edge_bps": 3.0,
                    "win_rate": 0.56,
                    "profit_factor": 1.30,
                },
                "AMZN": {
                    "sample_count": 30,
                    "net_edge_bps": 5.0,
                    "win_rate": 0.60,
                    "profit_factor": 1.50,
                },
                "MSFT": {
                    "sample_count": 30,
                    "net_edge_bps": -30.0,
                    "win_rate": 0.35,
                    "profit_factor": 0.60,
                },
            }
        },
        "shadow_report": {
            "markout_summary": {
                "best_symbols": [
                    {"symbol": "AMZN", "samples": 30, "mean_net_markout_bps": 6.0, "positive_rate": 0.62},
                    {"symbol": "AAPL", "samples": 30, "mean_net_markout_bps": 3.0, "positive_rate": 0.57},
                ],
                "worst_symbols": [
                    {"symbol": "MSFT", "samples": 30, "mean_net_markout_bps": -31.0, "positive_rate": 0.32}
                ],
            }
        },
        "trading_day_report": {
            "symbol_trade_flow": {
                "AAPL": {"desired": 6, "submitted": 6, "rejected": 0, "fills": 6},
                "AMZN": {"desired": 4, "submitted": 4, "rejected": 0, "fills": 4},
                "MSFT": {"desired": 5, "submitted": 1, "rejected": 4, "fills": 1},
            },
            "symbol_slippage_bps": {"AAPL": 2.0, "AMZN": 2.5, "MSFT": 18.0},
            "symbol_contribution": {"AAPL": 12.0, "AMZN": 4.0, "MSFT": -7.0},
        },
        "symbol_scorecard": {
            "policy": {"allowed_symbols": ["AAPL", "MSFT"], "shadow_only_symbols": ["AMZN"]},
            "symbols": [
                {"symbol": "AAPL", "effective_mode": "allow", "sample_count": 30},
                {"symbol": "AMZN", "effective_mode": "allow", "sample_count": 30},
                {"symbol": "MSFT", "effective_mode": "allow", "sample_count": 30},
            ],
        },
    }


def test_symbol_promotion_comparison_recommends_per_symbol_actions() -> None:
    inputs = _sample_inputs()

    report = symbol_promotion_comparison.build_symbol_promotion_comparison(
        report_date="2026-05-08",
        symbols={"AMZN", "AAPL", "MSFT"},
        canary_symbols={"AAPL"},
        min_samples=10,
        now=datetime(2026, 5, 8, 14, 30, tzinfo=UTC),
        **inputs,
    )

    by_symbol = {row["symbol"]: row for row in report["symbols"]}
    assert report["promotion_authority"] is False
    assert report["runtime_symbol_gating_changed"] is False
    assert by_symbol["AAPL"]["recommendation"] == "keep_canary"
    assert by_symbol["AMZN"]["recommendation"] == "consider_promotion"
    assert by_symbol["MSFT"]["recommendation"] == "disable"
    assert by_symbol["AMZN"]["sample_sufficiency"]["sufficient"] is True
    assert by_symbol["MSFT"]["metrics"]["trading_day"]["rejected"] == 4
    assert "p90_total_cost_bps_disable" in by_symbol["MSFT"]["reasons"]


def test_symbol_promotion_comparison_collects_more_evidence_when_sparse() -> None:
    report = symbol_promotion_comparison.build_symbol_promotion_comparison(
        report_date="2026-05-08",
        symbols={"AMZN"},
        replay_report={"replay_symbol_summary": {"AMZN": {"sample_count": 2, "win_rate": 0.60}}},
        shadow_symbols={"AMZN"},
        min_samples=10,
    )

    row = report["symbols"][0]
    assert row["recommendation"] == "collect_more_evidence"
    assert row["confidence"] == "low"
    assert row["sample_sufficiency"]["sufficient"] is False


def test_symbol_promotion_comparison_cli_writes_report_and_latest(tmp_path: Path) -> None:
    inputs = _sample_inputs()
    live = tmp_path / "live.json"
    replay = tmp_path / "replay.json"
    shadow = tmp_path / "shadow.json"
    trading = tmp_path / "trading.json"
    scorecard = tmp_path / "scorecard.json"
    output = tmp_path / "symbol_promotion_20260508.json"
    latest = tmp_path / "symbol_promotion_latest.json"
    live.write_text(json.dumps(inputs["live_cost_model"]), encoding="utf-8")
    replay.write_text(json.dumps(inputs["replay_report"]), encoding="utf-8")
    shadow.write_text(json.dumps(inputs["shadow_report"]), encoding="utf-8")
    trading.write_text(json.dumps(inputs["trading_day_report"]), encoding="utf-8")
    scorecard.write_text(json.dumps(inputs["symbol_scorecard"]), encoding="utf-8")

    rc = symbol_promotion_comparison.main(
        [
            "--report-date",
            "2026-05-08",
            "--symbols",
            "AMZN,AAPL,MSFT",
            "--canary-symbols",
            "AAPL",
            "--min-samples",
            "10",
            "--live-cost-model-json",
            str(live),
            "--replay-report-json",
            str(replay),
            "--shadow-report-json",
            str(shadow),
            "--trading-day-json",
            str(trading),
            "--symbol-scorecard-json",
            str(scorecard),
            "--output-json",
            str(output),
            "--latest-json",
            str(latest),
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    latest_payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "symbol_promotion_comparison"
    assert payload["paths"]["latest"] == str(latest)
    assert latest_payload["summary"]["recommendations"]["consider_promotion"] == 1
