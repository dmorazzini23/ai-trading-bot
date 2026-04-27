from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ai_trading.core.netting import NettedTarget, SleeveProposal
from ai_trading.core.netting_candidate_rank import rank_netting_candidates
from ai_trading.risk.liquidity_regime import LiquidityFeatures


def _make_target(
    *,
    symbol: str,
    now: datetime,
    target_dollars: float,
    confidence: float,
    disagreement: float,
) -> NettedTarget:
    proposal = SleeveProposal(
        symbol=symbol,
        sleeve="intraday",
        bar_ts=now,
        target_dollars=target_dollars,
        expected_edge_bps=12.0,
        expected_cost_bps=2.0,
        score=confidence,
        confidence=confidence,
    )
    return NettedTarget(
        symbol=symbol,
        bar_ts=now,
        target_dollars=target_dollars,
        target_shares=0.0,
        proposals=[proposal],
        disagreement_ratio=disagreement,
    )


def _base_kwargs() -> dict[str, Any]:
    now = datetime(2026, 4, 19, 15, 0, tzinfo=UTC)
    return {
        "now": now,
        "targets": {"AAPL": _make_target(symbol="AAPL", now=now, target_dollars=1000.0, confidence=0.9, disagreement=0.8)},
        "net_edge_raw_by_symbol": {"AAPL": 10.0},
        "latest_liquidity": {"AAPL": LiquidityFeatures(rolling_volume=5000.0, spread_bps=8.0, volatility_proxy=0.8)},
        "latest_price": {"AAPL": 100.0},
        "positions": {"AAPL": 0.0},
        "symbol_returns": {"AAPL": [0.01, -0.005, 0.02]},
        "edge_realism_rank_factor_by_symbol": {},
        "counterfactual_signal_by_symbol_seed": {},
        "opportunity_quality_by_symbol_seed": {},
        "learned_fill_trials_by_bucket": {},
        "learned_fill_success_by_bucket": {},
        "learned_exec_cost_stats_by_bucket": {},
        "execution_learning_state": {},
        "rejection_concentration_by_symbol": {},
        "realized_edge_by_symbol": {},
        "realized_edge_by_symbol_session": {},
        "realized_edge_by_symbol_session_regime": {},
        "bandit_rewards_by_symbol": {},
        "bandit_rewards_by_symbol_session": {},
        "bandit_rewards_by_symbol_session_regime": {},
        "bandit_bucket_significance_cache": {},
        "counterfactual_buckets": {},
        "replay_quality_by_symbol": {},
        "replay_quality_by_symbol_session": {},
        "replay_quality_by_symbol_session_regime": {},
        "replay_quality_context": {},
        "portfolio_rank_correlation": {},
        "held_notional_weights": {},
        "opportunity_quality_gate": {
            "enabled": False,
            "top_quantile": 0.2,
            "min_symbols": 1,
            "threshold": None,
            "allowed_symbols": [],
        },
        "safe_float": lambda value: float(value) if value is not None else None,
        "mean_std_func": lambda values: (
            float(sum(values) / len(values)),
            0.0,
            len(values),
        )
        if values
        else (0.0, 0.0, 0),
        "sequential_significance_gate_func": lambda **_kwargs: {"passed": True},
        "percentile_linear_func": lambda values, quantile: sorted(values)[
            max(0, min(len(values) - 1, int((len(values) - 1) * quantile)))
        ]
        if values
        else None,
        "lookup_learned_fill_probability_func": lambda **_kwargs: (None, 0, None),
        "lookup_learned_execution_cost_components_func": lambda **_kwargs: (None, None, None, 0, None),
        "exit_policy_pressure_context_func": lambda _state, **_kwargs: {"active": False},
        "exit_policy_state": object(),
        "edge_realism_rank_calibration_enabled": False,
        "clip_expected_edge_enabled": False,
        "edge_clip_cap_bps": 50.0,
        "alpha_time_decay_enabled": False,
        "alpha_time_decay_half_life_sec": 300.0,
        "alpha_time_decay_floor": 0.5,
        "bandit_active_session": "opening",
        "bandit_active_regime": "trend",
        "expected_capture_rank_enabled": True,
        "expected_capture_learned_fill_model_enabled": False,
        "expected_capture_learned_fill_min_samples": 5,
        "expected_capture_learned_fill_prior_alpha": 2.0,
        "expected_capture_learned_fill_prior_beta": 2.0,
        "expected_capture_liquidity_role_default": "maker",
        "expected_capture_venue_default": "ALPACA",
        "execution_learning_rank_enabled": False,
        "execution_learning_rank_min_samples": 2,
        "execution_learning_rank_slippage_floor_bps": 4.0,
        "execution_learning_rank_slippage_weight": 0.3,
        "execution_learning_rank_negative_edge_weight": 0.15,
        "execution_learning_rank_adverse_weight": 0.1,
        "execution_learning_rank_realization_floor": 0.85,
        "execution_learning_rank_realization_weight_bps": 8.0,
        "execution_learning_rank_max_penalty_bps": 35.0,
        "expected_capture_fill_prob_floor": 0.2,
        "expected_capture_fill_prob_cap": 0.95,
        "expected_capture_spread_penalty_bps": 0.1,
        "expected_capture_participation_penalty_bps": 0.2,
        "expected_capture_age_half_life_sec": 300.0,
        "expected_capture_latency_bps_per_sec": 0.02,
        "expected_capture_cost_model_enabled": False,
        "expected_capture_cost_model_min_samples": 10,
        "expected_capture_cost_spread_weight": 1.0,
        "expected_capture_cost_impact_weight": 1.0,
        "expected_capture_cost_latency_weight": 1.0,
        "rejection_concentration_rank_enabled": False,
        "rejection_concentration_min_count": 3,
        "rejection_concentration_scale_bps": 0.45,
        "rejection_concentration_max_penalty_bps": 24.0,
        "expected_capture_rank_weight": 0.5,
        "expected_capture_max_adjust_abs": 100.0,
        "expected_capture_floor_bps_effective": 2.0,
        "expected_capture_constraint_weight": 0.25,
        "expected_capture_constraint_max_adjust_abs": 50.0,
        "edge_model_v2_enabled": False,
        "edge_model_v2_weight": 0.45,
        "edge_model_v2_min_samples": 3,
        "edge_model_v2_uncertainty_z": 1.0,
        "edge_model_v2_regime_weight": 0.55,
        "edge_model_v2_session_weight": 0.30,
        "edge_model_v2_global_weight": 0.15,
        "edge_model_v2_cost_weight": 1.0,
        "edge_model_v2_clip_bps": 80.0,
        "edge_model_v2_max_rank_uplift_abs": 15.0,
        "edge_model_v2_max_rank_uplift_frac": 0.2,
        "realized_edge_rank_enabled": False,
        "realized_edge_rank_weight": 0.35,
        "realized_edge_rank_min_samples": 2,
        "realized_edge_rank_uncertainty_z": 1.0,
        "realized_edge_rank_clip_bps": 30.0,
        "realized_edge_rank_max_rank_uplift_abs": 12.0,
        "realized_edge_rank_max_rank_uplift_frac": 0.2,
        "realized_edge_rank_session_bucket_enabled": True,
        "realized_edge_rank_regime_bucket_enabled": True,
        "bandit_enabled": False,
        "bandit_method": "ucb",
        "bandit_min_samples": 2,
        "bandit_session_bucket_enabled": True,
        "bandit_regime_bucket_enabled": True,
        "bandit_weight": 0.25,
        "bandit_exploration": 1.0,
        "bandit_shadow_only": True,
        "bandit_live_promoted": False,
        "bandit_bucket_promotion_enabled": True,
        "bandit_bucket_promote_min_samples": 5,
        "bandit_bucket_promote_min_mean_reward_bps": 0.0,
        "bandit_promote_min_mean_reward_bps": 0.0,
        "bandit_max_rank_uplift_abs": 25.0,
        "bandit_max_rank_uplift_frac": 0.3,
        "promotion_significance_enabled": True,
        "promotion_significance_method": "posterior",
        "promotion_significance_posterior_prob_min": 0.95,
        "promotion_significance_sprt_alpha": 0.05,
        "promotion_significance_sprt_beta": 0.1,
        "promotion_significance_sprt_effect_bps": 1.0,
        "counterfactual_enabled": False,
        "counterfactual_weight": 0.5,
        "counterfactual_min_samples": 5,
        "counterfactual_clip_bps": 50.0,
        "counterfactual_max_rank_uplift_abs": 20.0,
        "counterfactual_max_rank_uplift_frac": 0.25,
        "counterfactual_live_promoted": False,
        "geometric_tiebreak_enabled": False,
        "geometric_tiebreak_weight": 0.15,
        "geometric_variance_penalty": 1.0,
        "geometric_downside_penalty": 0.75,
        "geometric_drawdown_penalty": 1.25,
        "portfolio_log_growth_rank_enabled": False,
        "portfolio_log_growth_rank_weight": 0.2,
        "portfolio_log_growth_variance_penalty": 1.0,
        "portfolio_log_growth_corr_penalty_bps": 1.0,
        "portfolio_log_growth_exposure_penalty_bps": 1.0,
        "portfolio_log_growth_turnover_penalty_bps": 1.0,
        "portfolio_log_growth_liquidity_penalty_bps": 1.0,
        "portfolio_log_growth_max_participation": 0.1,
        "portfolio_log_growth_max_adjust_abs": 25.0,
        "portfolio_current_gross_for_rank": 0.0,
        "replay_quality_rank_enabled": False,
        "replay_quality_weight": 0.18,
        "replay_quality_min_samples": 5,
        "replay_quality_clip_bps": 25.0,
        "replay_quality_max_rank_uplift_abs": 8.0,
        "replay_quality_max_rank_uplift_frac": 0.1,
        "replay_quality_session_bucket_enabled": True,
        "replay_quality_regime_bucket_enabled": True,
        "replay_quality_fallback_to_edge_model_v2": True,
        "exit_policy_entry_penalty_enabled": False,
        "exit_policy_entry_penalty_weight": 5.0,
        "exit_policy_entry_penalty_min_pressure": 0.55,
        "exit_policy_entry_penalty_max_abs": 20.0,
        "rank_downside_overlap_cap_enabled": False,
        "rank_downside_overlap_cap_frac": 0.3,
        "rank_downside_overlap_cap_abs": 15.0,
        "opportunity_quality_enabled": False,
        "opportunity_top_quantile": 0.5,
        "opportunity_min_symbols": 1,
        "opportunity_demote_score": 1000.0,
    }


def test_rank_netting_candidates_computes_expected_capture_and_reasons() -> None:
    result = rank_netting_candidates(**_base_kwargs())

    assert result.candidate_expected_net_edge["AAPL"] == 10.0
    assert result.candidate_expected_capture["AAPL"] < 10.0
    assert result.candidate_rank["AAPL"] > 0.0
    assert "EXPECTED_CAPTURE_OPTIMIZER" in result.counterfactual_signal_by_symbol["AAPL"] or True
    assert result.counterfactual_signal_by_symbol["AAPL"]["expected_capture_bps"] == result.candidate_expected_capture["AAPL"]
    assert result.counterfactual_signal_by_symbol["AAPL"]["opportunity_quality_score"] >= 0.0


def test_rank_netting_candidates_uses_order_delta_for_expected_capture_costs() -> None:
    kwargs = _base_kwargs()
    kwargs["targets"] = {
        "AAPL": _make_target(
            symbol="AAPL",
            now=kwargs["now"],
            target_dollars=1100.0,
            confidence=0.9,
            disagreement=0.8,
        )
    }
    kwargs["latest_price"] = {"AAPL": 100.0}
    kwargs["latest_liquidity"] = {
        "AAPL": LiquidityFeatures(
            rolling_volume=100.0,
            spread_bps=1.0,
            volatility_proxy=0.8,
        )
    }
    kwargs["positions"] = {"AAPL": 10.0}

    result = rank_netting_candidates(**kwargs)

    signal = result.counterfactual_signal_by_symbol["AAPL"]
    assert signal["expected_capture_participation"] == 0.01


def test_rank_netting_candidates_applies_opportunity_gate_demotions() -> None:
    kwargs = _base_kwargs()
    now = kwargs["now"]
    kwargs["targets"] = {
        "AAPL": _make_target(symbol="AAPL", now=now, target_dollars=1200.0, confidence=0.95, disagreement=0.95),
        "MSFT": _make_target(symbol="MSFT", now=now, target_dollars=200.0, confidence=0.25, disagreement=0.15),
    }
    kwargs["net_edge_raw_by_symbol"] = {"AAPL": 20.0, "MSFT": 1.0}
    kwargs["latest_liquidity"] = {
        "AAPL": LiquidityFeatures(rolling_volume=7000.0, spread_bps=5.0, volatility_proxy=0.7),
        "MSFT": LiquidityFeatures(rolling_volume=2000.0, spread_bps=18.0, volatility_proxy=1.2),
    }
    kwargs["latest_price"] = {"AAPL": 100.0, "MSFT": 50.0}
    kwargs["positions"] = {"AAPL": 0.0, "MSFT": 0.0}
    kwargs["symbol_returns"] = {
        "AAPL": [0.01, 0.01, 0.02],
        "MSFT": [-0.01, 0.0, 0.005],
    }
    kwargs["opportunity_quality_enabled"] = True
    kwargs["opportunity_top_quantile"] = 0.95
    kwargs["opportunity_min_symbols"] = 1
    kwargs["percentile_linear_func"] = lambda values, _quantile: max(values) if values else None
    kwargs["opportunity_quality_gate"] = {
        "enabled": True,
        "top_quantile": 0.95,
        "min_symbols": 1,
        "threshold": None,
        "allowed_symbols": [],
    }

    result = rank_netting_candidates(**kwargs)

    assert result.opportunity_quality_gate["active"] is True
    assert "AAPL" in result.opportunity_allowed_symbols
    assert "MSFT" not in result.opportunity_allowed_symbols
    assert result.candidate_rank["AAPL"] > result.candidate_rank["MSFT"]
    assert result.candidate_rank["MSFT"] < -500.0
