from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.replay.live_cost_alignment import (
    resolve_live_cost_alignment,
    resolve_live_cost_alignments,
)


def _live_cost_model(
    now: datetime,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "artifact_type": "live_cost_model",
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "status": {"available": True, "status": "ready"},
        "by_symbol_side_session_order_type_volatility": rows,
    }


def test_resolves_exact_live_cost_bucket_and_uses_pessimistic_cost() -> None:
    now = datetime(2026, 5, 8, 15, 0, tzinfo=UTC)
    payload = _live_cost_model(
        now,
        [
            {
                "symbol": "AAPL",
                "side": "buy",
                "session_regime": "midday",
                "order_type": "limit",
                "volatility_bucket": "normal",
                "sample_count": 8,
                "sufficient_samples": True,
                "p90_total_cost_bps": 14.0,
                "last_observed_at": now.isoformat().replace("+00:00", "Z"),
            },
            {
                "symbol": "AAPL",
                "side": "buy",
                "session_regime": "midday",
                "order_type": "market",
                "volatility_bucket": "normal",
                "sample_count": 8,
                "sufficient_samples": True,
                "p90_total_cost_bps": 99.0,
                "last_observed_at": now.isoformat().replace("+00:00", "Z"),
            },
        ],
    )

    result = resolve_live_cost_alignment(
        payload,
        symbol="aapl",
        side="buy",
        session_bucket="MIDDAY",
        order_type="limit",
        volatility_bucket="normal",
        fallback_cost_bps=9.0,
        now=now,
        min_samples=5,
        max_age_seconds=600,
    )

    assert result["resolved_cost_bps"] == pytest.approx(14.0)
    assert result["observed_live_cost_bps"] == pytest.approx(14.0)
    assert result["fallback_cost_bps"] == pytest.approx(9.0)
    assert result["sample_count"] == 8
    assert result["freshness"]["fresh"] is True
    assert result["alignment"] == "pessimism"
    assert result["pessimistic"] is True
    assert result["source"] == "live"


def test_cheaper_live_cost_is_clamped_to_fallback_to_avoid_false_optimism() -> None:
    now = datetime(2026, 5, 8, 15, 0, tzinfo=UTC)
    payload = _live_cost_model(
        now,
        [
            {
                "symbol": "MSFT",
                "side": "sell_short",
                "session_regime": "opening",
                "order_type": "marketable_limit",
                "volatility_bucket": "high",
                "sample_count": 12,
                "sufficient_samples": True,
                "p90_total_cost_bps": 3.0,
                "last_observed_at": now.isoformat().replace("+00:00", "Z"),
            }
        ],
    )

    result = resolve_live_cost_alignment(
        payload,
        symbol="MSFT",
        side="short",
        session_bucket="opening",
        order_type="marketable-limit",
        volatility_bucket="high",
        fallback_cost_bps=7.5,
        now=now,
        min_samples=5,
        max_age_seconds=600,
    )

    assert result["resolved_cost_bps"] == pytest.approx(7.5)
    assert result["observed_live_cost_bps"] == pytest.approx(3.0)
    assert result["alignment"] == "optimism"
    assert result["optimistic"] is True
    assert result["source"] == "fallback"


def test_stale_live_cost_uses_fallback_and_reports_freshness() -> None:
    now = datetime(2026, 5, 8, 15, 0, tzinfo=UTC)
    stale_generated_at = now - timedelta(hours=4)
    payload = _live_cost_model(
        stale_generated_at,
        [
            {
                "symbol": "AAPL",
                "side": "buy",
                "session_regime": "closing",
                "order_type": "limit",
                "volatility_bucket": "wide",
                "sample_count": 9,
                "sufficient_samples": True,
                "p90_total_cost_bps": 20.0,
                "last_observed_at": stale_generated_at.isoformat().replace("+00:00", "Z"),
            }
        ],
    )

    result = resolve_live_cost_alignment(
        payload,
        symbol="AAPL",
        side="buy",
        session_bucket="closing",
        order_type="limit",
        volatility_bucket="wide",
        fallback_cost_bps=6.0,
        now=now,
        min_samples=5,
        max_age_seconds=600,
    )

    assert result["resolved_cost_bps"] == pytest.approx(6.0)
    assert result["observed_live_cost_bps"] == pytest.approx(20.0)
    assert result["alignment"] == "stale"
    assert result["freshness"]["fresh"] is False
    assert result["freshness"]["stale"] is True
    assert result["freshness"]["reason"] == "stale_model"


def test_insufficient_live_cost_samples_use_fallback() -> None:
    now = datetime(2026, 5, 8, 15, 0, tzinfo=UTC)
    payload = _live_cost_model(
        now,
        [
            {
                "symbol": "AAPL",
                "side": "sell",
                "session_regime": "midday",
                "order_type": "limit",
                "volatility_bucket": "normal",
                "sample_count": 4,
                "sufficient_samples": True,
                "p90_total_cost_bps": 30.0,
                "last_observed_at": now.isoformat().replace("+00:00", "Z"),
            }
        ],
    )

    result = resolve_live_cost_alignment(
        payload,
        symbol="AAPL",
        side="sell",
        session_bucket="midday",
        order_type="limit",
        volatility_bucket="normal",
        fallback_cost_bps=8.0,
        now=now,
        min_samples=5,
        max_age_seconds=600,
    )

    assert result["resolved_cost_bps"] == pytest.approx(8.0)
    assert result["observed_live_cost_bps"] == pytest.approx(30.0)
    assert result["sample_count"] == 4
    assert result["alignment"] == "insufficient_samples"
    assert result["source"] == "fallback"


def test_batch_alignment_reports_optimism_pessimism_and_stale_counts() -> None:
    now = datetime(2026, 5, 8, 15, 0, tzinfo=UTC)
    payload = _live_cost_model(
        now,
        [
            {
                "symbol": "AAPL",
                "side": "buy",
                "session_regime": "midday",
                "order_type": "limit",
                "volatility_bucket": "normal",
                "sample_count": 8,
                "sufficient_samples": True,
                "p90_total_cost_bps": 2.0,
                "last_observed_at": now.isoformat().replace("+00:00", "Z"),
            },
            {
                "symbol": "MSFT",
                "side": "buy",
                "session_regime": "midday",
                "order_type": "limit",
                "volatility_bucket": "normal",
                "sample_count": 8,
                "sufficient_samples": True,
                "p90_total_cost_bps": 11.0,
                "last_observed_at": now.isoformat().replace("+00:00", "Z"),
            },
        ],
    )

    result = resolve_live_cost_alignments(
        payload,
        [
            {
                "symbol": "AAPL",
                "side": "buy",
                "session_regime": "midday",
                "order_type": "limit",
                "volatility_bucket": "normal",
                "fallback_cost_bps": 5.0,
            },
            {
                "symbol": "MSFT",
                "side": "buy",
                "session_regime": "midday",
                "order_type": "limit",
                "volatility_bucket": "normal",
                "fallback_cost_bps": 5.0,
            },
            {
                "symbol": "TSLA",
                "side": "buy",
                "session_regime": "midday",
                "order_type": "limit",
                "volatility_bucket": "normal",
                "fallback_cost_bps": 5.0,
            },
        ],
        now=now,
        min_samples=5,
        max_age_seconds=600,
    )

    assert result["summary"]["alignment_counts"] == {
        "missing_live_cost_bucket": 1,
        "optimism": 1,
        "pessimism": 1,
    }
    assert result["summary"]["optimism_count"] == 1
    assert result["summary"]["pessimism_count"] == 1
    assert result["summary"]["stale_count"] == 0
