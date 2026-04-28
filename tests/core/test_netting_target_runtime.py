from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from types import SimpleNamespace

from ai_trading.core.netting import NettedTarget
from ai_trading.core.netting_candidate_rank import NettingCandidateRankingResult
from ai_trading.core.netting_target_runtime import (
    NettingExecutionRankRuntimeContext,
    apply_target_construction_controls,
    prepare_portfolio_optimizer_runtime,
    store_candidate_ranking_runtime_state,
)
from ai_trading.risk.portfolio_limits import PortfolioLimitsResult


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, object] | None]] = []

    def warning(self, event: str, extra: dict[str, object] | None = None, **_: object) -> None:
        self.warnings.append((event, extra))


def _target(symbol: str, dollars: float) -> NettedTarget:
    return NettedTarget(
        symbol=symbol,
        bar_ts=datetime(2026, 4, 20, 14, 30, tzinfo=UTC),
        target_dollars=float(dollars),
        target_shares=0.0,
    )


def test_store_candidate_ranking_runtime_state_builds_context_from_values() -> None:
    runtime = SimpleNamespace()
    bandit_sig = {"posterior": {"pass": True}}
    toggles = {"slice_a": True}
    ranking_result = NettingCandidateRankingResult(
        candidate_expected_net_edge={"AAPL": 12.5},
        candidate_expected_capture={"AAPL": 8.0},
        candidate_rank={"AAPL": 1.25},
        counterfactual_signal_by_symbol={"AAPL": {"score": 0.5}},
        opportunity_quality_by_symbol={"AAPL": 0.9},
        opportunity_allowed_symbols={"AAPL"},
        opportunity_quality_gate={"threshold": 0.8},
    )

    store_candidate_ranking_runtime_state(
        runtime=runtime,
        ranking_result=ranking_result,
        edge_realism_rank_factor_by_symbol={"AAPL": 0.95},
        context_values={
            "clip_expected_edge_enabled": True,
            "edge_clip_cap_bps": 15.0,
            "median_abs_edge_bps": 6.0,
            "bandit_enabled": True,
            "bandit_method": "ucb",
            "bandit_active_session": "rth",
            "bandit_active_regime": "trend",
            "bandit_session_bucket_enabled": True,
            "bandit_regime_bucket_enabled": False,
            "bandit_shadow_only": False,
            "bandit_live_promoted": True,
            "bandit_bucket_promotion_enabled": True,
            "bandit_bucket_promote_min_samples": 10,
            "bandit_bucket_promote_min_mean_reward_bps": 2.5,
            "bandit_promote_min_samples": 12,
            "bandit_promote_min_mean_reward_bps": 1.5,
            "bandit_global_samples": 50,
            "bandit_global_mean_reward_bps": 3.2,
            "bandit_max_rank_uplift_abs": 5.0,
            "bandit_max_rank_uplift_frac": 0.2,
            "promotion_significance_enabled": True,
            "promotion_significance_method": "sprt",
            "bandit_significance_context": bandit_sig,
            "counterfactual_enabled": True,
            "counterfactual_shadow_only": False,
            "counterfactual_live_promoted": True,
            "counterfactual_weight": 0.4,
            "counterfactual_min_samples": 9,
            "counterfactual_clip_bps": 7.0,
            "counterfactual_global_events": 30,
            "counterfactual_global_dr_mean_bps": 1.1,
            "counterfactual_significance_context": {"prob": 0.9},
            "realized_edge_rank_enabled": True,
            "realized_edge_rank_weight": 0.2,
            "realized_edge_rank_min_samples": 5,
            "realized_edge_rank_uncertainty_z": 1.2,
            "realized_edge_rank_clip_bps": 9.0,
            "edge_model_v2_enabled": True,
            "edge_model_v2_weight": 0.3,
            "edge_model_v2_min_samples": 7,
            "edge_model_v2_uncertainty_z": 1.0,
            "edge_model_v2_regime_weight": 0.1,
            "edge_model_v2_session_weight": 0.2,
            "edge_model_v2_global_weight": 0.3,
            "edge_model_v2_cost_weight": 0.4,
            "edge_model_v2_clip_bps": 6.0,
            "replay_quality_rank_enabled": True,
            "replay_quality_session_bucket_enabled": True,
            "replay_quality_regime_bucket_enabled": False,
            "replay_quality_auto_disable_if_stale": True,
            "replay_quality_fallback_to_edge_model_v2": True,
            "replay_quality_weight": 0.15,
            "replay_quality_min_samples": 11,
            "replay_quality_clip_bps": 5.5,
            "replay_quality_max_age_hours": 24.0,
            "replay_quality_by_symbol": {"AAPL": 0.8},
            "replay_quality_by_symbol_session": {"AAPL:rth": 0.7},
            "replay_quality_by_symbol_session_regime": {"AAPL:rth:trend": 0.6},
            "replay_quality_context": {"bucket": {"age_h": 2}},
            "expected_capture_rank_enabled": True,
            "expected_capture_rank_weight": 0.25,
            "expected_capture_floor_bps_effective": 3.0,
            "expected_capture_floor_bps": 4.0,
            "expected_capture_learned_fill_model_enabled": True,
            "expected_capture_learned_fill_min_samples": 8,
            "expected_capture_cost_model_enabled": True,
            "expected_capture_cost_model_min_samples": 6,
            "execution_learning_rank_enabled": True,
            "execution_learning_rank_min_samples": 4,
            "execution_learning_rank_slippage_floor_bps": 1.5,
            "execution_learning_rank_max_penalty_bps": 6.5,
            "rejection_concentration_rank_enabled": True,
            "rejection_concentration_lookback_hours": 48.0,
            "rejection_concentration_window_records": 100,
            "rejection_concentration_max_penalty_bps": 8.0,
            "expected_capture_fill_prob_floor": 0.2,
            "expected_capture_fill_prob_cap": 0.9,
            "expected_capture_spread_penalty_bps": 0.4,
            "expected_capture_participation_penalty_bps": 0.8,
            "expected_capture_age_half_life_sec": 120.0,
            "rank_downside_overlap_cap_enabled": True,
            "rank_downside_overlap_cap_frac": 0.5,
            "rank_downside_overlap_cap_abs": 1000.0,
            "alpha_time_decay_enabled": True,
            "alpha_time_decay_half_life_sec": 60.0,
            "alpha_time_decay_floor": 0.25,
            "alpha_stale_signal_sec": 300.0,
            "alpha_time_stop_enabled": True,
            "alpha_time_stop_sec": 180.0,
            "alpha_time_stop_max_expected_edge_bps": 2.0,
            "portfolio_log_growth_rank_enabled": True,
            "portfolio_log_growth_rank_weight": 0.1,
            "portfolio_log_growth_turnover_penalty_bps": 0.3,
            "portfolio_log_growth_liquidity_penalty_bps": 0.2,
            "portfolio_log_growth_max_participation": 0.05,
            "geometric_tiebreak_enabled": True,
            "geometric_tiebreak_weight": 0.05,
            "edge_realism_rank_calibration_enabled": True,
            "edge_realism_rank_calibration_session_enabled": True,
            "edge_realism_rank_calibration_min_samples": 5,
            "edge_realism_rank_calibration_prior_samples": 20,
            "edge_realism_rank_calibration_ratio_floor": 0.5,
            "edge_realism_rank_calibration_ratio_cap": 1.5,
            "edge_realism_apply_to_approval_enabled": True,
            "policy_rollback_disabled_slices": {"slice_b", "slice_a"},
            "policy_disabled_gate_roots": {"gate_b", "gate_a"},
            "policy_disabled_sleeves": {"sleeve_b", "sleeve_a"},
            "policy_runtime_payload": {
                "updated_at": "2026-04-20T12:00:00+00:00",
                "source_updated_at": "2026-04-20T11:59:00+00:00",
            },
            "toggles": toggles,
            "opportunity_quality_gate": {"threshold": 0.8},
        },
    )

    bandit_sig["posterior"]["pass"] = False
    toggles["slice_a"] = False

    assert runtime.execution_candidate_rank_expected_edge_bps == {"AAPL": 12.5}
    assert runtime.execution_candidate_rank_context["bandit_method"] == "ucb"
    assert runtime.execution_candidate_rank_context["bandit_promotion_significance"] == {
        "posterior": {"pass": True}
    }
    assert runtime.execution_candidate_rank_context["policy_runtime_toggles"] == {
        "slice_a": True
    }
    assert runtime.execution_candidate_rank_context["policy_disabled_gate_roots"] == [
        "gate_a",
        "gate_b",
    ]


def test_netting_execution_rank_runtime_context_defaults_and_shape() -> None:
    context = NettingExecutionRankRuntimeContext.from_values(
        {
            "bandit_enabled": True,
            "bandit_method": "ucb",
            "bandit_promote_min_samples": "bad",
            "expected_capture_rank_weight": "not-a-float",
            "execution_learning_rank_slippage_floor_bps": float("nan"),
            "bandit_significance_context": {"nested": {"passed": True}},
            "counterfactual_significance_context": {"passed": False},
            "policy_runtime_payload": "malformed",
            "policy_disabled_gate_roots": {"gate_b", "gate_a"},
            "policy_disabled_sleeves": "not-a-sequence",
            "toggles": {"rankers": {"bandit_enabled": True}},
        }
    )

    runtime_context = context.to_runtime_context()
    assert runtime_context["bandit_enabled"] is True
    assert runtime_context["counterfactual_enabled"] is False
    assert runtime_context["bandit_promote_min_samples"] == 0
    assert runtime_context["expected_capture_rank_weight"] == 0.0
    assert runtime_context["execution_learning_rank_slippage_floor_bps"] == 0.0
    assert runtime_context["bandit_method"] == "ucb"
    assert runtime_context["bandit_promotion_significance"] == {
        "nested": {"passed": True}
    }
    assert runtime_context["counterfactual_promotion_significance"] == {
        "passed": False
    }
    assert runtime_context["policy_runtime_toggles_updated_at"] is None
    assert runtime_context["policy_runtime_toggles_source_updated_at"] is None
    assert runtime_context["policy_disabled_gate_roots"] == ["gate_a", "gate_b"]
    assert runtime_context["policy_disabled_sleeves"] == []
    assert runtime_context["policy_runtime_toggles"] == {
        "rankers": {"bandit_enabled": True}
    }


def test_store_candidate_ranking_runtime_state_preserves_context_dict_shape() -> None:
    runtime = SimpleNamespace()
    captured_contexts: list[dict[str, object]] = []
    ranking_result = NettingCandidateRankingResult(
        candidate_expected_net_edge={"AAPL": 1.0},
        candidate_expected_capture={"AAPL": 0.5},
        candidate_rank={"AAPL": 2.0},
        counterfactual_signal_by_symbol={},
        opportunity_quality_by_symbol={},
        opportunity_allowed_symbols=set(),
        opportunity_quality_gate={},
    )

    def _store_runtime_state(
        runtime_arg: object,
        **kwargs: object,
    ) -> None:
        assert runtime_arg is runtime
        captured_contexts.append(kwargs["rank_context"])  # type: ignore[arg-type]
        runtime.execution_candidate_rank_context = kwargs["rank_context"]

    store_candidate_ranking_runtime_state(
        runtime=runtime,
        ranking_result=ranking_result,
        edge_realism_rank_factor_by_symbol={},
        context_values={
            "bandit_enabled": True,
            "bandit_global_samples": 12,
            "bandit_global_mean_reward_bps": 1.75,
            "policy_runtime_payload": {
                "updated_at": "2026-04-27T12:00:00+00:00",
                "source_updated_at": "2026-04-27T11:59:00+00:00",
            },
        },
        store_execution_candidate_ranking_runtime_state_func=_store_runtime_state,
    )

    assert isinstance(runtime.execution_candidate_rank_context, dict)
    assert captured_contexts == [runtime.execution_candidate_rank_context]
    assert runtime.execution_candidate_rank_context["bandit_enabled"] is True
    assert runtime.execution_candidate_rank_context["bandit_global_samples"] == 12
    assert runtime.execution_candidate_rank_context["bandit_global_mean_reward_bps"] == 1.75
    assert runtime.execution_candidate_rank_context["policy_runtime_toggles_updated_at"] == (
        "2026-04-27T12:00:00+00:00"
    )
    assert runtime.execution_candidate_rank_context["policy_runtime_toggles"] == {}


def test_prepare_portfolio_optimizer_runtime_normalizes_symbol_keys() -> None:
    logger = _Logger()
    captured_returns: list[dict[str, list[float]]] = []

    def _build_corr(
        returns: Mapping[str, Sequence[float]],
    ) -> dict[str, dict[str, float]]:
        captured_returns.append({symbol: list(values) for symbol, values in returns.items()})
        return {"AAPL": {"AAPL": 1.0}}

    runtime = prepare_portfolio_optimizer_runtime(
        latest_price={"aapl": 100.0, "AAPL": 101.0, "bad": float("nan")},
        symbol_returns={
            "aapl": [0.01, 0.02],
            "AAPL": [0.03],
            "msft": [0.04, "bad", 0.05],  # type: ignore[list-item]
        },
        rollout_toggle_func=lambda _name, default: True,
        get_env_func=lambda key, default=None, cast=None: {
            "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_OPENINGS_ONLY": True,
            "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_IMPROVEMENT_THRESHOLD": 0.02,
            "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_MAX_CORRELATION_PENALTY": 0.15,
            "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_REBALANCE_DRIFT_THRESHOLD": 0.05,
            "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_TURNOVER_PENALTY": 0.01,
            "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_MODE": "execution_live",
        }.get(key, default),
        build_symbol_return_correlation_matrix_func=_build_corr,
        logger=logger,
        create_portfolio_optimizer_func=lambda config: {"config": dict(config)},
    )

    assert runtime.enabled is True
    assert runtime.context["active"] is True
    assert runtime.market_data["prices"] == {"AAPL": 101.0}
    assert runtime.market_data["returns"] == {"AAPL": [0.01, 0.02, 0.03], "MSFT": [0.04, 0.05]}
    assert captured_returns[-1] == {"AAPL": [0.01, 0.02, 0.03], "MSFT": [0.04, 0.05]}


def test_prepare_portfolio_optimizer_runtime_keeps_init_failure_enabled_by_default() -> None:
    logger = _Logger()

    runtime = prepare_portfolio_optimizer_runtime(
        latest_price={"AAPL": 100.0},
        symbol_returns={},
        rollout_toggle_func=lambda _name, default: True,
        get_env_func=lambda key, default=None, cast=None: default,
        build_symbol_return_correlation_matrix_func=lambda _returns: {},
        logger=logger,
        create_portfolio_optimizer_func=lambda _config: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert runtime.enabled is True
    assert runtime.optimizer is None
    assert runtime.context["init_failed"] is True
    assert runtime.context["init_fail_open"] is False
    assert runtime.context["fail_open_applied"] is False
    assert logger.warnings[-1][0] == "PORTFOLIO_OPTIMIZER_INIT_FAILED"


def test_prepare_portfolio_optimizer_runtime_can_explicitly_fail_open_on_init_failure() -> None:
    runtime = prepare_portfolio_optimizer_runtime(
        latest_price={"AAPL": 100.0},
        symbol_returns={},
        rollout_toggle_func=lambda _name, default: True,
        get_env_func=lambda key, default=None, cast=None: (
            True
            if key == "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_INIT_FAIL_OPEN"
            else default
        ),
        build_symbol_return_correlation_matrix_func=lambda _returns: {},
        logger=_Logger(),
        create_portfolio_optimizer_func=lambda _config: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert runtime.enabled is False
    assert runtime.context["init_failed"] is True
    assert runtime.context["init_fail_open"] is True
    assert runtime.context["fail_open_applied"] is True


def test_apply_target_construction_controls_normalizes_returns_and_skips_invalid_targets() -> None:
    logger = _Logger()
    targets = {
        "AAPL": _target("AAPL", 1000.0),
        "MSFT": _target("MSFT", 500.0),
    }
    captured_limits_inputs: list[dict[str, object]] = []

    def _apply_limits(**kwargs: object) -> PortfolioLimitsResult:
        captured_limits_inputs.append(dict(kwargs))
        return PortfolioLimitsResult(
            scaled_targets={"aapl": 1500.0, "MSFT": float("nan")},
            scale=1.0,
            reasons=["VOL_TARGET_SCALE"],
        )

    apply_target_construction_controls(
        targets=targets,
        cfg=SimpleNamespace(
            global_max_symbol_dollars=10_000.0,
            global_max_gross_dollars=20_000.0,
            global_max_net_dollars=15_000.0,
        ),
        normalized_symbol_returns={"aapl": [0.01, 0.02], "MSFT": [0.03, 0.04]},
        apply_global_caps_func=lambda *_args, **_kwargs: None,
        apply_portfolio_limits_func=_apply_limits,
        get_env_func=lambda key, default=None, cast=None: {
            "AI_TRADING_PORTFOLIO_LIMITS_ENABLED": True,
            "AI_TRADING_VOL_TARGETING_ENABLED": True,
            "AI_TRADING_TARGET_ANNUAL_VOL": 0.18,
            "AI_TRADING_VOL_LOOKBACK_DAYS": 20,
            "AI_TRADING_VOL_MIN_SCALE": 0.25,
            "AI_TRADING_VOL_MAX_SCALE": 1.25,
            "AI_TRADING_CONCENTRATION_CAP_ENABLED": True,
            "AI_TRADING_MAX_SYMBOL_WEIGHT": 0.12,
            "AI_TRADING_MAX_CLUSTER_WEIGHT": 0.25,
            "AI_TRADING_CORR_CAP_ENABLED": True,
            "AI_TRADING_CORR_LOOKBACK_DAYS": 30,
            "AI_TRADING_CORR_THRESHOLD": 0.80,
            "AI_TRADING_CORR_GROUP_GROSS_CAP": 0.35,
        }.get(key, default),
        logger=logger,
    )

    assert captured_limits_inputs[-1]["symbol_returns"] == {
        "AAPL": [0.01, 0.02],
        "MSFT": [0.03, 0.04],
    }
    assert targets["AAPL"].target_dollars == 1500.0
    assert targets["AAPL"].reasons == ["VOL_TARGET_SCALE"]
    assert targets["MSFT"].target_dollars == 500.0
    assert logger.warnings[-1][0] == "PORTFOLIO_LIMIT_INVALID_TARGET_SKIP"
