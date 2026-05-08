from __future__ import annotations

from datetime import UTC, datetime

from ai_trading.tools.replay_live_cost_alignment_report import (
    build_replay_live_cost_alignment_report,
)


def test_replay_live_cost_alignment_report_clamps_cheaper_live_cost() -> None:
    live_cost = {
        "generated_at": "2026-05-08T12:00:00Z",
        "status": {"status": "ready"},
        "by_symbol_side_session_order_type_volatility": [
            {
                "symbol": "AMZN",
                "side": "buy",
                "session_regime": "midday",
                "order_type": "limit",
                "volatility_bucket": "normal",
                "sample_count": 50,
                "sufficient_samples": True,
                "p90_total_cost_bps": 1.0,
                "last_observed_at": "2026-05-08T11:59:00Z",
            }
        ],
    }
    replay = {
        "replay_cost_rows": [
            {
                "symbol": "AMZN",
                "side": "buy",
                "session_bucket": "midday",
                "order_type": "limit",
                "volatility_bucket": "normal",
                "fallback_cost_bps": 3.0,
            }
        ]
    }

    report = build_replay_live_cost_alignment_report(
        live_cost_model=live_cost,
        replay_report=replay,
        fallback_cost_bps=3.0,
        min_samples=5,
        max_age_seconds=3600.0,
        now=datetime(2026, 5, 8, 12, 0, tzinfo=UTC),
    )

    assert report["cost_realism"]["status"] == "conservative_fallback_clamped_optimism"
    assert report["items"][0]["alignment"] == "optimism"
    assert report["items"][0]["resolved_cost_bps"] == 3.0


def test_replay_live_cost_alignment_report_marks_stale_model() -> None:
    live_cost = {
        "generated_at": "2026-05-07T12:00:00Z",
        "status": {"status": "ready"},
        "by_symbol_side_session": [
            {
                "symbol": "AAPL",
                "side": "buy",
                "session_regime": "midday",
                "sample_count": 50,
                "p90_total_cost_bps": 5.0,
            }
        ],
    }

    report = build_replay_live_cost_alignment_report(
        live_cost_model=live_cost,
        replay_report={},
        fallback_cost_bps=3.0,
        min_samples=5,
        max_age_seconds=60.0,
        now=datetime(2026, 5, 8, 12, 0, tzinfo=UTC),
    )

    assert report["cost_realism"]["status"] == "stale"
    assert report["summary"]["stale_count"] == 1
