from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_trading.runtime.regime_entry_throttle import (
    RegimeEntryThrottleConfig,
    build_regime_entry_throttle_report,
    classify_session_regime,
    evaluate_regime_entry_throttle,
)


def test_classify_session_regime_opening_midday_closing() -> None:
    assert classify_session_regime(datetime(2026, 5, 5, 13, 45, tzinfo=UTC)) == "opening"
    assert classify_session_regime(datetime(2026, 5, 5, 17, 0, tzinfo=UTC)) == "midday"
    assert classify_session_regime(datetime(2026, 5, 5, 19, 45, tzinfo=UTC)) == "closing"


def test_evaluate_allows_fresh_sufficient_midday_normal_evidence() -> None:
    now = datetime(2026, 5, 5, 17, 0, tzinfo=UTC)

    result = evaluate_regime_entry_throttle(
        {
            "observed_at": (now - timedelta(seconds=30)).isoformat(),
            "sample_count": 40,
            "volatility_bps": 20.0,
            "spread_bps": 2.5,
            "provider_success_rate": 0.995,
        },
        now=now,
    )

    assert result["session_regime"] == "midday"
    assert result["volatility_regime"] == "normal"
    assert result["spread_regime"] == "normal"
    assert result["provider_regime"] == "healthy"
    assert result["action"] == "allow"
    assert result["qty_scale"] == 1.0
    assert result["sample_sufficiency"]["sufficient"] is True
    assert result["freshness"]["fresh"] is True


def test_evaluate_missing_evidence_blocks_new_entries_conservatively() -> None:
    result = evaluate_regime_entry_throttle(None, now=datetime(2026, 5, 5, 17, 0, tzinfo=UTC))

    assert result["action"] == "block_new_entries"
    assert result["qty_scale"] == 0.0
    assert result["volatility_regime"] == "high"
    assert result["spread_regime"] == "wide"
    assert result["provider_regime"] == "degraded"
    assert "sample_insufficient" in result["reasons"]
    assert "evidence_missing" in result["reasons"]
    assert result["sample_sufficiency"]["sufficient"] is False
    assert result["freshness"]["fresh"] is False


def test_evaluate_live_canary_blocks_where_non_canary_reduces() -> None:
    now = datetime(2026, 5, 5, 17, 0, tzinfo=UTC)
    evidence = {
        "observed_at": now.isoformat(),
        "sample_count": 100,
        "volatility_bps": 85.0,
        "spread_bps": 3.0,
        "provider_healthy": True,
    }

    normal = evaluate_regime_entry_throttle(evidence, now=now, live_canary=False)
    canary = evaluate_regime_entry_throttle(evidence, now=now, live_canary=True)

    assert normal["action"] == "reduce_size"
    assert normal["qty_scale"] == 0.5
    assert canary["action"] == "block_new_entries"
    assert canary["qty_scale"] == 0.0
    assert "live_canary_strict_regime_gate" in canary["reasons"]


def test_evaluate_observe_mode_surfaces_reasons_without_enforcing() -> None:
    now = datetime(2026, 5, 5, 19, 45, tzinfo=UTC)

    result = evaluate_regime_entry_throttle(
        {
            "observed_at": now.isoformat(),
            "sample_count": 80,
            "volatility_bps": 15.0,
            "spread_bps": 1.5,
            "provider_healthy": True,
        },
        now=now,
        enforce=False,
    )

    assert result["action"] == "observe"
    assert result["session_regime"] == "closing"
    assert "observe_only" in result["reasons"]


def test_build_regime_entry_throttle_report_summarizes_rows() -> None:
    cfg = RegimeEntryThrottleConfig(min_sample_count=2)
    now = datetime(2026, 5, 5, 17, 0, tzinfo=UTC)
    allow = evaluate_regime_entry_throttle(
        {
            "observed_at": now.isoformat(),
            "sample_count": 3,
            "volatility_bps": 10.0,
            "spread_bps": 1.0,
            "provider_healthy": True,
        },
        now=now,
        config=cfg,
    )
    block = evaluate_regime_entry_throttle({}, now=now, config=cfg)

    report = build_regime_entry_throttle_report([allow, block], report_date="2026-05-05")

    assert report["evaluations"] == 2
    assert report["actions"] == {"allow": 1, "block_new_entries": 1}
    assert report["sample_sufficiency"] == {"insufficient": 1, "sufficient": 1}
    assert report["freshness"] == {"fresh": 1, "stale_or_missing": 1}
    assert report["regimes"]["provider"] == {"degraded": 1, "healthy": 1}
