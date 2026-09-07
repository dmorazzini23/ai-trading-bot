from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_trading.research.focused_cycle import adjusted_daily_bounds, build_opportunity_ledger, summarize_period


def _bars():
    index = pd.date_range("2026-08-03T09:00:00Z", periods=660, freq="min")
    bars = pd.DataFrame({"open": 100.0 + np.arange(len(index)) * 0.01}, index=index)
    features = pd.DataFrame({"sma_spread": 1.0, "macd_signal_gap": 1.0}, index=index)
    return bars, features


def test_exact_next_open_and_session_cutoff_without_overlap():
    bars, features = _bars()
    ledger, _ = build_opportunity_ledger(bars, features, horizon_minutes=60)
    assert len(ledger) == 4
    assert ledger.iloc[0]["entry_timestamp"] == "2026-08-03T14:01:00+00:00"
    assert ledger.iloc[0]["exit_timestamp"] == "2026-08-03T15:01:00+00:00"
    for previous, following in zip(ledger["exit_timestamp"], ledger["entry_timestamp"].iloc[1:]):
        assert previous <= following
    changed = bars.copy()
    changed.loc[changed.index > pd.Timestamp(ledger.iloc[0]["exit_timestamp"]), "open"] *= 2
    other, _ = build_opportunity_ledger(changed, features, horizon_minutes=60)
    assert other.iloc[0]["gross_return_bps"] == ledger.iloc[0]["gross_return_bps"]


def test_missing_bar_rejected_and_abstention_pays_no_cost():
    bars, features = _bars()
    missing = pd.Timestamp("2026-08-03T14:15:00Z")
    bars = bars.drop(missing)
    features = features.drop(missing) * -1
    ledger, diagnostics = build_opportunity_ledger(bars, features, horizon_minutes=60)
    assert diagnostics["missing_contiguous_bars"] == 1
    summary = summarize_period(ledger, sessions=["2026-08-03"], one_way_cost_bps=6.0)
    assert summary["completed_trades"] == 0
    assert summary["mean_daily_net_bps"] == 0.0
    assert summary["break_even_one_way_cost_bps"] is None
    with pytest.raises(ValueError):
        summarize_period(ledger, sessions=["2026-08-03"], one_way_cost_bps=-1)


def test_paired_block_bounds_and_multiple_testing_adjustment():
    daily = [{"candidate_net_bps": 2.0, "baseline_net_bps": 1.0}] * 50
    settings = {"bootstrap_block_sessions": 5, "bootstrap_replicates": 1000, "bootstrap_seed": 2, "familywise_alpha": 0.05, "simultaneous_bounds": 8}
    bounds = adjusted_daily_bounds(daily, settings=settings)
    assert bounds["net_lower_bps"] == 2.0
    assert bounds["improvement_lower_bps"] == 1.0
    assert bounds["per_bound_alpha"] == 0.00625
    assert adjusted_daily_bounds(daily[:10], settings=settings)["status"] == "insufficient_sessions"


def test_canonical_signal_features_do_not_change_when_future_bars_change():
    from ai_trading.tools.train_replay_aligned_model import _feature_frame

    bars, _ = _bars()
    bars["close"] = bars["open"] + np.sin(np.arange(len(bars)) / 9) * 0.1
    bars["high"] = bars[["open", "close"]].max(axis=1) + 0.1
    bars["low"] = bars[["open", "close"]].min(axis=1) - 0.1
    bars["volume"] = 10000
    original = _feature_frame(bars, symbol="AAPL")
    changed = bars.copy()
    changed.iloc[400:, changed.columns.get_indexer(["open", "high", "low", "close"])] *= 2
    alternative = _feature_frame(changed, symbol="AAPL")
    pd.testing.assert_frame_equal(original.iloc[:400], alternative.iloc[:400])


def test_reversal_uses_prior_volatility_and_is_future_invariant():
    from ai_trading.research.focused_cycle import selected_signals

    bars, features = _bars()
    returns = np.tile([0.0001, -0.0001], len(bars) // 2)
    returns[386:401] = -0.001
    bars["close"] = 100 * np.exp(np.cumsum(returns))
    rule = "negative_15m_shock_2sigma_prior60"
    signals = selected_signals(bars, features, rule)
    assert signals.iloc[400]
    assert not signals.iloc[380]
    changed = bars.copy()
    changed.iloc[401:, changed.columns.get_loc("close")] *= 3
    pd.testing.assert_series_equal(signals.iloc[:401], selected_signals(changed, features, rule).iloc[:401])
    missing = bars.index[350]
    gapped = selected_signals(bars.drop(missing), features.drop(missing), rule)
    assert pd.isna(gapped.loc[bars.index[400]])
    flat = bars.assign(close=100.0)
    assert selected_signals(flat, features, rule).isna().all()
    with pytest.raises(ValueError, match="unregistered"):
        selected_signals(bars, features, "unknown")


def test_completed_family_blocks_renaming_and_preserves_output(tmp_path, monkeypatch):
    import json
    from ai_trading.research import focused_cycle

    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"completed_families": [{"signal_rule": "positive_sma_macd", "universe": ["AAPL"], "closed_horizons_minutes": [1, 5, 15, 60]}]}))
    monkeypatch.setattr(focused_cycle, "REGISTRY_PATH", registry)
    protocol = {"protocol_id": "renamed", "universe": ["AAPL"], "horizons_minutes": [15]}
    with pytest.raises(ValueError, match="closed"):
        focused_cycle.check_study_permission(protocol, tmp_path)
    protocol["signal_rule"] = "negative_15m_shock_2sigma_prior60"
    with pytest.raises(ValueError, match="feasibility_review_required"):
        focused_cycle.check_study_permission(protocol, tmp_path)
    (tmp_path / "study_report.json").write_text("original")
    with pytest.raises(ValueError, match="already exists"):
        focused_cycle.check_study_permission(protocol, tmp_path)
    assert (tmp_path / "study_report.json").read_text() == "original"
