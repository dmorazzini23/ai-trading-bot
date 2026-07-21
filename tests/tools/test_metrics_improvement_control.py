from __future__ import annotations

import json

from ai_trading.tools.metrics_improvement_control import build_metrics_improvement_control, main


def _trading_day_report() -> dict:
    return {
        "report_date": "2026-05-22",
        "execution_capture": {
            "status": "insufficient_samples",
            "by_symbol": {
                "AAPL": {
                    "count": 5,
                    "mean_expected_net_edge_bps": 4.0,
                    "mean_realized_net_edge_bps": 0.5,
                },
                "AMZN": {
                    "count": 5,
                    "mean_expected_net_edge_bps": 2.0,
                    "mean_realized_net_edge_bps": -3.0,
                },
            },
        },
        "post_trade_surveillance": {
            "findings": [
                {"symbol": "AMZN", "category": "adverse_selection"},
                {"symbol": "AMZN", "category": "adverse_selection"},
                {"symbol": "AMZN", "category": "adverse_selection"},
                {"symbol": "AMZN", "category": "reject"},
            ]
        },
    }


def test_metrics_improvement_control_builds_conservative_symbol_controls() -> None:
    report = build_metrics_improvement_control(
        report_date="2026-05-22",
        reports=[_trading_day_report()],
        live_cost_model={
            "status": {"status": "warming_up"},
            "observed": {"p90_total_cost_bps": 3.0},
        },
        min_symbol_samples=5,
        max_adverse_findings=3,
    )

    assert report["status"] == "active"
    assert report["authority_increase_allowed"] is False
    assert report["promotion_authority"] is False
    assert report["live_money_authority"] is False
    assert report["by_symbol"]["AMZN"]["action"] == "cooldown"
    assert report["by_symbol"]["AMZN"]["qty_scale"] == 0.0
    assert report["by_symbol"]["AAPL"]["action"] == "downscale"
    assert report["by_symbol"]["AAPL"]["required_edge_bps"] > 3.0


def test_metrics_improvement_control_marks_low_sample_symbols_as_exploration() -> None:
    report = build_metrics_improvement_control(
        report_date="2026-05-22",
        reports=[
            {
                "report_date": "2026-05-22",
                "execution_capture": {
                    "by_symbol": {
                        "MSFT": {
                            "count": 1,
                            "mean_expected_net_edge_bps": 5.0,
                            "mean_realized_net_edge_bps": 2.0,
                        }
                    }
                },
            }
        ],
        min_symbol_samples=5,
    )

    assert report["by_symbol"]["MSFT"]["action"] == "explore"
    assert report["exploration_budget"]["max_orders_per_symbol_per_window"] == 1


def test_metrics_improvement_control_includes_configured_zero_sample_symbols() -> None:
    report = build_metrics_improvement_control(
        report_date="2026-05-22",
        reports=[_trading_day_report()],
        configured_symbols=["AAPL", "MSFT", "NVDA"],
        min_symbol_samples=5,
    )

    assert report["by_symbol"]["MSFT"]["samples"] == 0
    assert report["by_symbol"]["MSFT"]["action"] == "explore"
    assert "NVDA" not in report["by_symbol"]
    assert report["summary"]["configured_without_samples"] == ["MSFT"]
    assert report["summary"]["invalid_configured_symbols"] == ["NVDA"]
    assert report["control_policy"]["configured_symbols"] == ["AAPL", "MSFT"]


def test_metrics_improvement_control_excludes_observed_googl_from_actionable_routing() -> None:
    report = build_metrics_improvement_control(
        report_date="2026-07-20",
        reports=[
            {
                "report_date": "2026-07-20",
                "execution_capture": {
                    "by_symbol": {
                        "AAPL": {"count": 0},
                        "AMZN": {"count": 0},
                        "GOOGL": {
                            "count": 1,
                            "mean_expected_net_edge_bps": 5.0,
                            "mean_realized_net_edge_bps": 2.0,
                        },
                        "MSFT": {
                            "count": 4,
                            "mean_expected_net_edge_bps": 4.0,
                            "mean_realized_net_edge_bps": -2.0,
                        },
                    }
                },
            }
        ],
        configured_symbols=["AAPL", "AMZN", "MSFT"],
        min_symbol_samples=5,
    )

    actionable_lists = [
        report["routing"]["allowed_symbols"],
        report["routing"]["downscaled_symbols"],
        report["routing"]["shadowed_symbols"],
        report["routing"]["cooldown_symbols"],
        report["routing"]["exploration_symbols"],
        report["routing"]["sample_collection_priority"],
    ]
    assert all("GOOGL" not in symbols for symbols in actionable_lists)
    assert report["by_symbol"]["GOOGL"]["actionable"] is False
    assert report["by_symbol"]["GOOGL"]["action"] == "observe_only"
    assert report["by_symbol"]["GOOGL"]["qty_scale"] == 0.0
    assert report["summary"]["ignored_observed_symbols"] == ["GOOGL"]
    assert report["routing"]["ignored_observed_symbols"] == ["GOOGL"]
    assert set(report["routing"]["sample_collection_priority"]) <= {
        "AAPL",
        "AMZN",
        "MSFT",
    }


def test_metrics_improvement_control_builds_side_controls_from_calibration() -> None:
    report = build_metrics_improvement_control(
        report_date="2026-05-22",
        reports=[],
        expected_edge_calibration={
            "status": "inverted",
            "bucketed_by_side": {
                "buy": {
                    "count": 18,
                    "mean_expected_net_edge_bps": 3.8,
                    "mean_realized_net_edge_bps": 1.1,
                    "capture_ratio": 0.29,
                    "profit_factor": 1.3,
                    "win_rate": 0.56,
                },
                "sell": {
                    "count": 14,
                    "mean_expected_net_edge_bps": 0.7,
                    "mean_realized_net_edge_bps": -5.2,
                    "capture_ratio": -7.1,
                    "profit_factor": 0.24,
                    "win_rate": 0.29,
                },
            },
        },
        min_side_samples=8,
        min_symbol_samples=5,
        base_min_edge_bps=0.25,
        cost_p90_multiplier=0.0,
    )

    assert report["status"] == "active"
    assert report["by_side"]["sell"]["action"] == "shadow"
    assert report["by_side"]["sell"]["qty_scale"] == 0.0
    assert report["by_side"]["sell"]["reasons"] == ["side_capture_ratio_hard_breach"]
    assert report["by_side"]["sell"]["closing_position_authority_action"] == "allow_reduce_only"
    assert report["exit_quality_control"]["recommended_action"] == "review_exit_timing_and_order_type"
    assert report["by_side"]["buy"]["action"] == "allow"
    assert report["summary"]["side_shadowed_or_blocked"] == 1
    assert report["authority_increase_allowed"] is False
    assert report["live_money_authority"] is False


def test_metrics_improvement_control_cli_writes_latest(tmp_path) -> None:
    daily_root = tmp_path / "daily"
    run_dir = daily_root / "20260522T210000Z_daily"
    run_dir.mkdir(parents=True)
    (run_dir / "trading_day_report.json").write_text(
        json.dumps(_trading_day_report()),
        encoding="utf-8",
    )
    output = tmp_path / "metrics.json"
    latest = tmp_path / "latest.json"

    rc = main(
        [
            "--report-date",
            "2026-05-22",
            "--daily-report-root",
            str(daily_root),
            "--output-json",
            str(output),
            "--latest-json",
            str(latest),
            "--base-min-edge-bps",
            "0.25",
            "--cost-p90-multiplier",
            "0",
            "--max-exploration-orders",
            "20",
            "--max-exploration-orders-per-symbol",
            "5",
            "--min-side-samples",
            "8",
            "--configured-symbols",
            "AAPL,MSFT,NVDA",
            "--unknown-quote-metadata-edge-add-bps",
            "0",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "metrics_improvement_control"
    assert payload["control_policy"]["base_min_edge_bps"] == 0.25
    assert payload["control_policy"]["cost_p90_multiplier"] == 0.0
    assert payload["control_policy"]["min_side_samples"] == 8
    assert payload["control_policy"]["unknown_quote_metadata_edge_add_bps"] == 0.0
    assert payload["exploration_budget"]["max_orders_per_window"] == 20
    assert payload["exploration_budget"]["max_orders_per_symbol_per_window"] == 5
    assert "NVDA" not in payload["by_symbol"]
    assert payload["summary"]["invalid_configured_symbols"] == ["NVDA"]
    assert payload["routing"]["sample_collection_priority"]
    assert latest.exists()
