from __future__ import annotations

from ai_trading.tools.daily_research_pipeline import build_daily_research_report
from ai_trading.tools.metrics_improvement_control import build_metrics_improvement_control
from ai_trading.tools.trading_day_report import build_trading_day_report


def _improvement() -> dict[str, object]:
    return {
        "status": "needs_review",
        "recommended_next_action": "apply_conservative_execution_capture_haircuts",
        "summary": {"samples": 4, "capture_ratio": -0.5},
        "bad_buckets": {"by_symbol": [{"bucket_key": "AAPL", "count": 4}]},
        "edge_haircuts": {
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
            "by_symbol": {
                "AAPL": {
                    "action": "downscale",
                    "qty_scale": 0.25,
                    "required_edge_add_bps": 3.0,
                    "order_behavior": "wait_and_submit_or_passive_limit",
                    "reasons": ["weak_capture_ratio"],
                }
            },
        },
        "training_labels": {
            "recommended_targets": ["realized_net_edge_after_cost"],
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
        },
        "runtime_authority": False,
        "promotion_authority": False,
        "live_money_authority": False,
    }


def test_metrics_control_consumes_execution_capture_haircuts_conservatively() -> None:
    report = build_metrics_improvement_control(
        report_date="2026-07-01",
        reports=[
            {
                "report_date": "2026-07-01",
                "execution_capture": {
                    "by_symbol": {
                        "AAPL": {
                            "count": 8,
                            "mean_expected_net_edge_bps": 5.0,
                            "mean_realized_net_edge_bps": 4.0,
                        }
                    }
                },
            }
        ],
        execution_capture_improvement=_improvement(),
        min_symbol_samples=5,
        base_min_edge_bps=0.25,
        cost_p90_multiplier=0.0,
    )

    row = report["by_symbol"]["AAPL"]
    assert row["action"] == "downscale"
    assert row["qty_scale"] == 0.25
    assert row["required_edge_bps"] == 3.25
    assert "execution_capture_edge_haircut" in row["reasons"]
    assert "execution_capture_qty_haircut" in row["reasons"]
    assert report["inputs"]["execution_capture_improvement_status"] == "needs_review"
    assert report["authority_increase_allowed"] is False
    assert report["live_money_authority"] is False


def test_daily_research_report_surfaces_execution_capture_improvement_without_authority() -> None:
    report = build_daily_research_report(
        report_date="2026-07-01",
        execution_capture_improvement=_improvement(),
    )

    summary = report["execution_capture_improvement"]
    assert summary["status"] == "needs_review"
    assert summary["recommended_next_action"] == "apply_conservative_execution_capture_haircuts"
    assert summary["training_labels"]["recommended_targets"] == ["realized_net_edge_after_cost"]
    assert summary["runtime_authority"] is False
    assert summary["promotion_authority"] is False
    assert summary["live_money_authority"] is False


def test_trading_day_report_surfaces_execution_capture_improvement_without_authority() -> None:
    report = build_trading_day_report(
        report_date="2026-07-01",
        order_intents=[],
        fills=[],
        shadow_rows=[],
        gate_rows=[],
        live_cost_model={},
        symbol_scorecard={},
        execution_capture_improvement=_improvement(),
    )

    summary = report["execution_capture_improvement"]
    assert summary["status"] == "needs_review"
    assert summary["runtime_authority"] is False
    assert summary["promotion_authority"] is False
    assert summary["live_money_authority"] is False
    assert report["health_report_summary"]["execution_capture_improvement_status"] == "needs_review"
