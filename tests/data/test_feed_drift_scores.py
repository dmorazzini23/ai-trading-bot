from __future__ import annotations

from ai_trading.analytics.feed_drift import build_symbol_reliability_scores


def test_build_symbol_reliability_scores() -> None:
    rows = [
        {
            "symbol": "AAPL",
            "outcome": "submitted",
            "metrics": {
                "price_drift_bps": 12.0,
                "spread_ratio": 1.2,
                "volume_coverage_ratio": 0.8,
            },
        },
        {
            "symbol": "AAPL",
            "outcome": "submitted",
            "metrics": {
                "price_drift_bps": -35.0,
                "spread_ratio": 1.5,
                "volume_coverage_ratio": 0.6,
            },
            "signal_metrics": {
                "decision_class": "trigger",
                "signal_disagreement": False,
            },
        },
        {
            "symbol": "AAPL",
            "outcome": "skipped_quote_gate",
            "metrics": {
                "price_drift_bps": 30.0,
                "spread_ratio": 1.0,
                "volume_coverage_ratio": 0.7,
            },
            "signal_metrics": {
                "decision_class": "gate",
                "signal_disagreement": True,
            },
        },
    ]

    scores = build_symbol_reliability_scores(rows, drift_disagreement_bps=25.0, min_samples=3)
    assert "AAPL" in scores
    score = scores["AAPL"]
    assert score["sample_count"] == 3
    assert score["median_abs_price_drift_bps"] is not None
    assert score["trigger_disagreement_rate"] is not None
    assert score["gate_disagreement_rate"] is not None
    assert score["trigger_disagreement_rate"] == 0.0
    assert score["gate_disagreement_rate"] == 1.0
    assert score["trigger_agreement_rate"] == 1.0
    assert score["gate_agreement_rate"] == 0.0
    reliability = score["reliability_score"]
    assert isinstance(reliability, float)
    assert 0.0 <= reliability <= 1.0
