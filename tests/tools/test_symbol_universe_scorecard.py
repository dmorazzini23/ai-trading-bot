from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ai_trading.tools import symbol_universe_scorecard


def test_symbol_universe_scorecard_combines_cost_markout_and_persistence() -> None:
    now = datetime(2026, 5, 1, 15, 0, tzinfo=UTC)
    live_cost = {
        "by_symbol_side_session": [
            {
                "symbol": "MSFT",
                "side": "buy",
                "session_regime": "midday",
                "sample_count": 30,
                "mean_total_cost_bps": 28.0,
                "p90_total_cost_bps": 42.0,
                "mean_spread_bps": 35.0,
                "p90_spread_bps": 70.0,
            },
            {
                "symbol": "AAPL",
                "side": "buy",
                "session_regime": "midday",
                "sample_count": 30,
                "mean_total_cost_bps": 3.0,
                "p90_total_cost_bps": 5.0,
                "mean_spread_bps": 2.0,
                "p90_spread_bps": 4.0,
            },
        ]
    }
    shadow = {
        "markout_summary": {
            "best_symbols": [
                {
                    "symbol": "AAPL",
                    "samples": 30,
                    "mean_net_markout_bps": 6.0,
                    "positive_rate": 0.6,
                }
            ],
            "worst_symbols": [
                {
                    "symbol": "MSFT",
                    "samples": 30,
                    "mean_net_markout_bps": -30.0,
                    "positive_rate": 0.3,
                }
            ],
        }
    }
    previous = {
        "symbols": [
            {
                "symbol": "MSFT",
                "recommended_mode": "disabled",
                "persistence_count": 1,
            }
        ]
    }

    report = symbol_universe_scorecard.build_symbol_universe_scorecard(
        live_cost_model=live_cost,
        shadow_report=shadow,
        previous_scorecard=previous,
        min_samples=25,
        min_persistence=2,
        now=now,
    )

    by_symbol = {row["symbol"]: row for row in report["symbols"]}
    assert report["status"]["status"] == "ready"
    assert by_symbol["MSFT"]["effective_mode"] == "disabled"
    assert by_symbol["MSFT"]["persistence_count"] == 2
    assert by_symbol["AAPL"]["effective_mode"] == "allow"
    assert report["policy"]["disabled_symbols"] == ["MSFT"]


def test_symbol_universe_scorecard_waits_for_persistence() -> None:
    report = symbol_universe_scorecard.build_symbol_universe_scorecard(
        live_cost_model={
            "by_symbol_side_session": [
                {
                    "symbol": "MSFT",
                    "side": "buy",
                    "session_regime": "midday",
                    "sample_count": 30,
                    "p90_total_cost_bps": 45.0,
                }
            ]
        },
        min_samples=25,
        min_persistence=2,
        now=datetime(2026, 5, 1, 15, 0, tzinfo=UTC),
    )

    row = report["symbols"][0]
    assert row["recommended_mode"] == "disabled"
    assert row["effective_mode"] == "allow"
    assert "awaiting_persistence" in row["reasons"]


def test_symbol_universe_scorecard_uses_replay_symbol_summary() -> None:
    report = symbol_universe_scorecard.build_symbol_universe_scorecard(
        replay_report={
            "replay_symbol_summary": {
                "AAPL": {
                    "sample_count": 50,
                    "net_edge_bps": 4.0,
                    "win_rate": 0.58,
                    "profit_factor": 1.2,
                },
                "MSFT": {
                    "sample_count": 50,
                    "net_edge_bps": -30.0,
                    "win_rate": 0.04,
                    "profit_factor": 0.03,
                },
            }
        },
        previous_scorecard={
            "symbols": [
                {
                    "symbol": "MSFT",
                    "recommended_mode": "disabled",
                    "persistence_count": 1,
                }
            ]
        },
        min_samples=25,
        min_persistence=2,
        now=datetime(2026, 5, 1, 15, 0, tzinfo=UTC),
    )

    by_symbol = {row["symbol"]: row for row in report["symbols"]}
    assert by_symbol["AAPL"]["replay_samples"] == 50
    assert by_symbol["AAPL"]["positive_rate"] == 0.58
    assert by_symbol["MSFT"]["effective_mode"] == "disabled"
    assert by_symbol["MSFT"]["profit_factor"] == 0.03


def test_symbol_universe_scorecard_suggests_shadow_promotion() -> None:
    report = symbol_universe_scorecard.build_symbol_universe_scorecard(
        replay_report={
            "replay_symbol_summary": {
                "AAPL": {
                    "sample_count": 40,
                    "net_edge_bps": -4.0,
                    "win_rate": 0.45,
                },
                "AMZN": {
                    "sample_count": 40,
                    "net_edge_bps": 2.0,
                    "win_rate": 0.58,
                },
            }
        },
        executable_symbols={"AAPL"},
        shadow_symbols={"AMZN", "MSFT"},
        shadow_promotion_min_score_delta=0.5,
        shadow_promotion_min_samples=10,
        min_samples=25,
        now=datetime(2026, 5, 1, 15, 0, tzinfo=UTC),
    )

    promotion = report["shadow_promotion"]
    assert promotion["available"] is True
    assert [row["symbol"] for row in promotion["suggestions"]] == ["AMZN"]
    assert promotion["suggestions"][0]["recommended_action"] == (
        "consider_promote_shadow_to_canary"
    )


def test_symbol_universe_scorecard_flags_universe_mismatch_and_starvation() -> None:
    report = symbol_universe_scorecard.build_symbol_universe_scorecard(
        replay_report={
            "replay_symbol_summary": {
                "AAPL": {
                    "sample_count": 100,
                    "net_edge_bps": 2.0,
                    "win_rate": 0.55,
                }
            }
        },
        executable_symbols={"AAPL", "AMZN"},
        shadow_symbols={"MSFT"},
        min_samples=25,
        starvation_threshold=0.95,
        now=datetime(2026, 5, 1, 15, 0, tzinfo=UTC),
    )

    diagnostics = report["diagnostics"]
    assert diagnostics["universe_mismatch"] is True
    assert diagnostics["configured_without_evidence"] == ["AMZN", "MSFT"]
    assert diagnostics["symbol_starvation"] is True
    assert diagnostics["dominant_symbol"] == "AAPL"


def test_symbol_universe_scorecard_cli_writes_artifact(tmp_path: Path) -> None:
    live_cost_path = tmp_path / "live_cost_model_latest.json"
    output_path = tmp_path / "symbol_universe_scorecard_latest.json"
    live_cost_path.write_text(
        json.dumps(
            {
                "by_symbol_side_session": [
                    {
                        "symbol": "AAPL",
                        "side": "buy",
                        "session_regime": "midday",
                        "sample_count": 5,
                        "p90_total_cost_bps": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = symbol_universe_scorecard.main(
        [
            "--live-cost-model-json",
            str(live_cost_path),
            "--output-json",
            str(output_path),
            "--min-samples",
            "1",
            "--min-persistence",
            "1",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "symbol_universe_scorecard"
    assert payload["symbols"][0]["symbol"] == "AAPL"
