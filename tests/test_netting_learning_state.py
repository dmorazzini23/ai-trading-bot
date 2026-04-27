from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from ai_trading.core.netting_learning_state import (
    build_netting_learning_state,
    lookup_learned_execution_cost_components,
    lookup_learned_fill_probability,
)


def test_lookup_helpers_prefer_specific_bucket() -> None:
    learned_fill_trials_by_bucket = {
        "AAPL:opening:trend:maker:ALPACA:opening": 12.0,
        "AAPL": 50.0,
    }
    learned_fill_success_by_bucket = {
        "AAPL:opening:trend:maker:ALPACA:opening": 9.0,
        "AAPL": 20.0,
    }
    learned_exec_cost_stats_by_bucket = {
        "AAPL:opening:trend:maker:ALPACA:opening": {
            "samples": 12.0,
            "spread_sum_bps": 18.0,
            "impact_sum_bps": 24.0,
            "latency_sum_bps": 6.0,
        }
    }

    fill_probability, fill_samples, fill_key = lookup_learned_fill_probability(
        learned_fill_trials_by_bucket=learned_fill_trials_by_bucket,
        learned_fill_success_by_bucket=learned_fill_success_by_bucket,
        expected_capture_learned_fill_min_samples=10,
        expected_capture_learned_fill_prior_alpha=2.0,
        expected_capture_learned_fill_prior_beta=2.0,
        symbol="AAPL",
        session_token="opening",
        regime_token="trend",
        liquidity_role="maker",
        venue_session="ALPACA:opening",
    )
    assert fill_key == "AAPL:opening:trend:maker:ALPACA:opening"
    assert fill_samples == 12
    assert fill_probability is not None
    assert fill_probability == 0.6875

    spread_cost, impact_cost, latency_cost, cost_samples, cost_key = (
        lookup_learned_execution_cost_components(
            learned_exec_cost_stats_by_bucket=learned_exec_cost_stats_by_bucket,
            safe_float=lambda value: float(value) if value is not None else None,
            expected_capture_cost_model_min_samples=10,
            symbol="AAPL",
            session_token="opening",
            regime_token="trend",
            liquidity_role="maker",
            venue_session="ALPACA:opening",
        )
    )
    assert cost_key == "AAPL:opening:trend:maker:ALPACA:opening"
    assert cost_samples == 12
    assert spread_cost == 1.5
    assert impact_cost == 2.0
    assert latency_cost == 0.5


def _build_state_for_single_tca_row(
    *,
    now: datetime,
    target: SimpleNamespace,
    tca_row: dict[str, object],
    fill_model_enabled: bool,
    cost_model_enabled: bool,
):
    return build_netting_learning_state(
        now=now,
        targets={"AAPL": target},
        positions={"AAPL": 5.0},
        latest_price={"AAPL": 100.0},
        symbol_returns={"AAPL": [0.01, -0.005, 0.02]},
        get_env=lambda _key, default=None, cast=None: default,
        safe_float=lambda value: float(value) if value is not None else None,
        session_bucket_from_ts_func=lambda _ts: "opening",
        get_regime_signal_profile_func=lambda: "trend",
        load_counterfactual_learning_state_func=lambda: {"global": {}, "buckets": {}},
        load_execution_learning_state_func=lambda: {
            "global": {},
            "buckets": {},
            "symbol_buckets": {},
        },
        load_recent_rejection_concentration_by_symbol_func=lambda **_kwargs: {},
        resolved_tca_path_func=lambda: "runtime/tca.jsonl",
        read_jsonl_records_func=lambda _path, max_records=0: [tca_row],
        infer_tca_liquidity_role_func=lambda _row: "maker",
        extract_tca_fill_success_ratio_func=lambda _row: 1.0,
        mean_std_func=lambda values: (
            float(sum(values) / len(values)),
            0.0,
            len(values),
        )
        if values
        else (0.0, 0.0, 0),
        sequential_significance_gate_func=lambda **_kwargs: {"passed": True},
        build_symbol_return_correlation_matrix_func=lambda _returns: {
            "AAPL": {"AAPL": 1.0}
        },
        percentile_linear_func=lambda values, quantile: sorted(values)[
            max(0, min(len(values) - 1, int((len(values) - 1) * quantile)))
        ]
        if values
        else None,
        parse_iso_timestamp_func=lambda _value: now,
        bandit_enabled=True,
        bandit_method="ucb",
        bandit_min_samples=1,
        bandit_window_trades=10,
        bandit_session_bucket_enabled=True,
        bandit_regime_bucket_enabled=True,
        bandit_shadow_only=False,
        bandit_auto_promote=True,
        bandit_promote_min_samples=1,
        bandit_promote_min_mean_reward_bps=0.0,
        bandit_bucket_promotion_enabled=True,
        bandit_bucket_promote_min_samples=1,
        bandit_bucket_promote_min_mean_reward_bps=0.0,
        edge_realism_rank_calibration_enabled=True,
        edge_realism_rank_calibration_session_enabled=True,
        edge_realism_rank_calibration_min_samples=1,
        edge_realism_rank_calibration_window_trades=10,
        edge_realism_rank_calibration_prior_samples=0,
        edge_realism_rank_calibration_ratio_floor=0.1,
        edge_realism_rank_calibration_ratio_cap=2.0,
        counterfactual_enabled=False,
        counterfactual_shadow_only=False,
        counterfactual_auto_promote=False,
        counterfactual_min_samples=1,
        counterfactual_promote_min_events=1,
        counterfactual_promote_min_dr_mean_bps=0.0,
        realized_edge_rank_enabled=True,
        expected_capture_rank_enabled=True,
        expected_capture_learned_fill_model_enabled=fill_model_enabled,
        expected_capture_learned_fill_min_samples=1,
        expected_capture_learned_fill_prior_alpha=2.0,
        expected_capture_learned_fill_prior_beta=2.0,
        expected_capture_venue_default="ALPACA",
        expected_capture_cost_model_enabled=cost_model_enabled,
        expected_capture_cost_model_min_samples=1,
        expected_capture_latency_bps_per_sec=0.02,
        expected_capture_floor_bps=5.0,
        expected_capture_floor_adaptive_enabled=True,
        expected_capture_floor_adaptive_quantile=0.5,
        execution_learning_rank_enabled=False,
        rejection_concentration_rank_enabled=False,
        rejection_concentration_window_records=100,
        rejection_concentration_lookback_hours=18.0,
        promotion_significance_enabled=True,
        promotion_significance_method="posterior",
        promotion_significance_posterior_prob_min=0.95,
        promotion_significance_sprt_alpha=0.05,
        promotion_significance_sprt_beta=0.1,
        promotion_significance_sprt_effect_bps=1.0,
        opportunity_quality_enabled=False,
        opportunity_top_quantile=0.2,
        opportunity_min_symbols=1,
        opportunity_openings_only=False,
        portfolio_log_growth_rank_enabled=True,
    )


def test_build_netting_learning_state_collects_learned_buckets() -> None:
    now = datetime(2026, 4, 19, 15, 0, tzinfo=UTC)
    target = SimpleNamespace(target_dollars=1500.0)
    tca_row = {
        "symbol": "AAPL",
        "status": "filled",
        "session_regime": "opening",
        "regime_profile": "trend",
        "venue": "ALPACA",
        "expected_net_edge_bps": 12.0,
        "spread_paid_bps": 1.0,
        "is_bps": 3.0,
        "fill_latency_ms": 250.0,
        "execution_drift_bps": 0.5,
        "realized_net_edge_bps": 8.0,
    }

    state = _build_state_for_single_tca_row(
        now=now,
        target=target,
        tca_row=tca_row,
        fill_model_enabled=True,
        cost_model_enabled=True,
    )

    assert state.bandit_active_session == "opening"
    assert state.bandit_active_regime == "trend"
    assert state.bandit_rewards_by_symbol["AAPL"] == [8.0]
    assert state.realized_edge_by_symbol["AAPL"] == [8.0]
    assert state.learned_fill_trials_by_bucket["AAPL"] == 1.0
    assert state.learned_fill_success_by_bucket["AAPL"] == 1.0
    assert state.learned_exec_cost_stats_by_bucket["AAPL"]["samples"] == 1.0
    assert state.expected_capture_observed_values
    assert state.edge_realism_rank_factor_by_symbol["AAPL"] > 0.0
    assert state.bandit_live_promoted is True
    assert state.portfolio_rank_correlation["AAPL"]["AAPL"] == 1.0
    assert state.held_notional_weights["AAPL"] == 1.0
    assert state.expected_capture_floor_bps_effective <= 5.0


def test_build_netting_learning_state_collects_costs_when_fill_model_disabled() -> None:
    now = datetime(2026, 4, 19, 15, 0, tzinfo=UTC)
    target = SimpleNamespace(target_dollars=1500.0)
    tca_row = {
        "symbol": "AAPL",
        "status": "filled",
        "session_regime": "opening",
        "regime_profile": "trend",
        "venue": "ALPACA",
        "expected_net_edge_bps": 12.0,
        "spread_paid_bps": 1.0,
        "is_bps": 3.0,
        "fill_latency_ms": 250.0,
        "execution_drift_bps": 0.5,
        "realized_net_edge_bps": 8.0,
    }

    state = _build_state_for_single_tca_row(
        now=now,
        target=target,
        tca_row=tca_row,
        fill_model_enabled=False,
        cost_model_enabled=True,
    )

    assert state.learned_fill_trials_by_bucket == {}
    assert state.learned_exec_cost_stats_by_bucket["AAPL"]["samples"] == 1.0
