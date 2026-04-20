"""Netting-cycle ranking/runtime helpers extracted from ``bot_engine.py``."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ai_trading.core.netting import apply_global_caps
from ai_trading.core.netting_candidate_rank import NettingCandidateRankingResult, rank_netting_candidates
from ai_trading.core.netting_learning_state import (
    build_netting_learning_state,
    lookup_learned_execution_cost_components,
    lookup_learned_fill_probability,
)
from ai_trading.core.netting_rank_prelude import (
    apply_policy_runtime_overrides,
    load_replay_quality_state,
)
from ai_trading.core.netting_target_runtime import (
    PortfolioOptimizerRuntime,
    apply_target_construction_controls,
    prepare_portfolio_optimizer_runtime,
    store_candidate_ranking_runtime_state,
)


def _bot_engine() -> Any:
    return importlib.import_module("ai_trading.core.bot_engine")


def _rollout_toggle(get_env_func: Any, primary_env: str, default: bool = False) -> bool:
    primary_value = get_env_func(primary_env, None, cast=bool)
    if primary_value is not None:
        return bool(primary_value)
    return bool(default)


def _normalize_disabled_slices(raw: Any) -> set[str]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return set()
    return {
        str(item).strip().upper()
        for item in raw
        if str(item).strip()
    }


@dataclass(slots=True)
class NettingRankRuntimeResult:
    ranking_result: NettingCandidateRankingResult
    portfolio_optimizer_runtime: PortfolioOptimizerRuntime
    edge_realism_rank_factor_by_symbol: dict[str, float]
    edge_realism_apply_to_approval_enabled: bool
    policy_runtime_payload: dict[str, Any]
    policy_disabled_gate_roots: set[str]
    policy_disabled_sleeves: set[str]
    policy_rollback_disabled_slices: set[str]
    opportunity_quality_enabled: bool
    opportunity_top_quantile: float
    opportunity_openings_only: bool
    alpha_time_decay_enabled: bool
    alpha_stale_signal_sec: float
    alpha_time_stop_enabled: bool
    alpha_time_stop_sec: float
    alpha_time_stop_max_expected_edge_bps: float


def build_netting_rank_runtime(
    *,
    state: Any,
    runtime: Any,
    cfg: Any,
    now: datetime,
    targets: dict[str, Any],
    net_edge_raw_by_symbol: Mapping[str, float],
    latest_liquidity: Mapping[str, Any],
    latest_price: Mapping[str, float],
    positions: Mapping[str, float],
    symbol_returns: Mapping[str, list[float]],
) -> NettingRankRuntimeResult:
    """Load ranking controls, compute learning state, and rank netting targets."""

    be = _bot_engine()
    get_env = be.get_env
    logger = be.logger

    clip_expected_edge_enabled = bool(
        get_env("AI_TRADING_ALLOC_EXPECTED_EDGE_CLIP_ENABLED", True, cast=bool)
    )
    clip_multiplier = max(
        0.5,
        float(
            get_env(
                "AI_TRADING_ALLOC_EXPECTED_EDGE_CLIP_MULTIPLIER",
                3.0,
                cast=float,
            )
        ),
    )
    clip_min_cap_bps = max(
        1.0,
        float(
            get_env(
                "AI_TRADING_ALLOC_EXPECTED_EDGE_CLIP_MIN_BPS",
                20.0,
                cast=float,
            )
        ),
    )
    clip_max_cap_bps = max(
        clip_min_cap_bps,
        float(
            get_env(
                "AI_TRADING_ALLOC_EXPECTED_EDGE_CLIP_MAX_BPS",
                250.0,
                cast=float,
            )
        ),
    )
    abs_edges = [
        abs(float(value))
        for value in net_edge_raw_by_symbol.values()
        if math.isfinite(float(value))
    ]
    median_abs_edge_bps = 0.0
    if abs_edges:
        sorted_abs = sorted(abs_edges)
        mid = len(sorted_abs) // 2
        if len(sorted_abs) % 2 == 0:
            median_abs_edge_bps = float((sorted_abs[mid - 1] + sorted_abs[mid]) / 2.0)
        else:
            median_abs_edge_bps = float(sorted_abs[mid])
    dynamic_cap_bps = max(
        clip_min_cap_bps,
        min(
            clip_max_cap_bps,
            float(median_abs_edge_bps) * float(clip_multiplier),
        ),
    )
    edge_clip_cap_bps = (
        float(dynamic_cap_bps) if clip_expected_edge_enabled else float(clip_max_cap_bps)
    )

    deprecated_short_knobs: list[str] = []
    for short_name in (
        "BANDIT",
        "GEOMETRIC",
        "META_LABEL",
        "PORTFOLIO_OPTIMIZER",
        "COUNTERFACTUAL",
        "PORTFOLIO_LOG_GROWTH",
    ):
        short_raw = get_env(short_name, "", cast=str)
        if str(short_raw or "").strip():
            deprecated_short_knobs.append(str(short_name))
    if deprecated_short_knobs:
        logger.info(
            "LEGACY_SHORT_ROLLOUT_KNOBS_IGNORED",
            extra={"knobs": sorted(deprecated_short_knobs)},
        )

    bandit_enabled = _rollout_toggle(
        get_env,
        "AI_TRADING_EXEC_BANDIT_ROUTING_ENABLED",
        False,
    )
    bandit_method = str(
        get_env("AI_TRADING_EXEC_BANDIT_METHOD", "ucb", cast=str) or "ucb"
    ).strip().lower()
    if bandit_method not in {"ucb", "thompson"}:
        bandit_method = "ucb"
    bandit_weight = max(
        0.0,
        min(
            5.0,
            float(get_env("AI_TRADING_EXEC_BANDIT_SCORE_WEIGHT", 0.25, cast=float)),
        ),
    )
    bandit_exploration = max(
        0.0,
        float(get_env("AI_TRADING_EXEC_BANDIT_EXPLORATION", 1.0, cast=float)),
    )
    bandit_min_samples = max(
        1,
        int(get_env("AI_TRADING_EXEC_BANDIT_MIN_SAMPLES", 40, cast=int)),
    )
    bandit_window_trades = max(
        bandit_min_samples,
        int(get_env("AI_TRADING_EXEC_BANDIT_WINDOW_TRADES", 200, cast=int)),
    )
    bandit_session_bucket_enabled = bool(
        get_env("AI_TRADING_EXEC_BANDIT_SESSION_BUCKET_ENABLED", True, cast=bool)
    )
    bandit_regime_bucket_enabled = bool(
        get_env("AI_TRADING_EXEC_BANDIT_REGIME_BUCKET_ENABLED", True, cast=bool)
    )
    bandit_shadow_only = bool(
        get_env("AI_TRADING_EXEC_BANDIT_SHADOW_ONLY", True, cast=bool)
    )
    bandit_auto_promote = bool(
        get_env("AI_TRADING_EXEC_BANDIT_AUTO_PROMOTE", False, cast=bool)
    )
    bandit_promote_min_samples = max(
        bandit_min_samples,
        int(get_env("AI_TRADING_EXEC_BANDIT_PROMOTE_MIN_SAMPLES", 120, cast=int)),
    )
    bandit_promote_min_mean_reward_bps = float(
        get_env("AI_TRADING_EXEC_BANDIT_PROMOTE_MIN_MEAN_REWARD_BPS", 0.0, cast=float)
    )
    bandit_max_rank_uplift_abs = max(
        0.0,
        float(get_env("AI_TRADING_EXEC_BANDIT_MAX_RANK_UPLIFT_ABS", 25.0, cast=float)),
    )
    bandit_max_rank_uplift_frac = max(
        0.0,
        min(
            1.0,
            float(
                get_env(
                    "AI_TRADING_EXEC_BANDIT_MAX_RANK_UPLIFT_FRAC",
                    0.30,
                    cast=float,
                )
            ),
        ),
    )
    bandit_bucket_promotion_enabled = bool(
        get_env(
            "AI_TRADING_EXEC_BANDIT_BUCKET_PROMOTION_ENABLED",
            True,
            cast=bool,
        )
    )
    bandit_bucket_promote_min_samples = max(
        bandit_min_samples,
        int(
            get_env(
                "AI_TRADING_EXEC_BANDIT_BUCKET_PROMOTE_MIN_SAMPLES",
                60,
                cast=int,
            )
        ),
    )
    bandit_bucket_promote_min_mean_reward_bps = float(
        get_env(
            "AI_TRADING_EXEC_BANDIT_BUCKET_PROMOTE_MIN_MEAN_REWARD_BPS",
            0.0,
            cast=float,
        )
    )
    geometric_tiebreak_enabled = _rollout_toggle(
        get_env,
        "AI_TRADING_EXEC_GEOMETRIC_TIEBREAK_ENABLED",
        False,
    )
    geometric_tiebreak_weight = max(
        0.0,
        min(
            1.0,
            float(
                get_env("AI_TRADING_EXEC_GEOMETRIC_TIEBREAK_WEIGHT", 0.15, cast=float)
            ),
        ),
    )
    geometric_variance_penalty = max(
        0.0,
        float(get_env("AI_TRADING_EXEC_GEOMETRIC_VARIANCE_PENALTY", 1.0, cast=float)),
    )
    geometric_downside_penalty = max(
        0.0,
        float(get_env("AI_TRADING_EXEC_GEOMETRIC_DOWNSIDE_PENALTY", 0.75, cast=float)),
    )
    geometric_drawdown_penalty = max(
        0.0,
        float(get_env("AI_TRADING_EXEC_GEOMETRIC_DRAWDOWN_PENALTY", 1.25, cast=float)),
    )
    edge_realism_rank_calibration_enabled = bool(
        get_env(
            "AI_TRADING_EXEC_EDGE_REALISM_RANK_CALIBRATION_ENABLED",
            False,
            cast=bool,
        )
    )
    edge_realism_rank_calibration_session_enabled = bool(
        get_env(
            "AI_TRADING_EXEC_EDGE_REALISM_RANK_CALIBRATION_SESSION_ENABLED",
            True,
            cast=bool,
        )
    )
    edge_realism_rank_calibration_min_samples = max(
        1,
        int(
            get_env(
                "AI_TRADING_EXEC_EDGE_REALISM_RANK_CALIBRATION_MIN_SAMPLES",
                20,
                cast=int,
            )
        ),
    )
    edge_realism_rank_calibration_window_trades = max(
        edge_realism_rank_calibration_min_samples,
        int(
            get_env(
                "AI_TRADING_EXEC_EDGE_REALISM_RANK_CALIBRATION_WINDOW_TRADES",
                300,
                cast=int,
            )
        ),
    )
    edge_realism_rank_calibration_prior_samples = max(
        0,
        int(
            get_env(
                "AI_TRADING_EXEC_EDGE_REALISM_RANK_CALIBRATION_PRIOR_SAMPLES",
                24,
                cast=int,
            )
        ),
    )
    edge_realism_rank_calibration_ratio_floor = max(
        0.01,
        min(
            1.0,
            float(
                get_env(
                    "AI_TRADING_EXEC_EDGE_REALISM_RANK_CALIBRATION_RATIO_FLOOR",
                    0.15,
                    cast=float,
                )
            ),
        ),
    )
    edge_realism_rank_calibration_ratio_cap = max(
        edge_realism_rank_calibration_ratio_floor,
        min(
            3.0,
            float(
                get_env(
                    "AI_TRADING_EXEC_EDGE_REALISM_RANK_CALIBRATION_RATIO_CAP",
                    1.0,
                    cast=float,
                )
            ),
        ),
    )
    edge_realism_apply_to_approval_enabled = bool(
        get_env(
            "AI_TRADING_EXEC_EDGE_REALISM_APPLY_TO_APPROVAL_ENABLED",
            True,
            cast=bool,
        )
    )
    counterfactual_enabled = _rollout_toggle(
        get_env,
        "AI_TRADING_EXEC_COUNTERFACTUAL_LEARNING_ENABLED",
        False,
    )
    counterfactual_shadow_only = bool(
        get_env("AI_TRADING_EXEC_COUNTERFACTUAL_SHADOW_ONLY", True, cast=bool)
    )
    counterfactual_auto_promote = bool(
        get_env("AI_TRADING_EXEC_COUNTERFACTUAL_AUTO_PROMOTE", False, cast=bool)
    )
    counterfactual_min_samples = max(
        1,
        int(get_env("AI_TRADING_EXEC_COUNTERFACTUAL_MIN_SAMPLES", 40, cast=int)),
    )
    counterfactual_weight = max(
        0.0,
        min(
            5.0,
            float(
                get_env("AI_TRADING_EXEC_COUNTERFACTUAL_SCORE_WEIGHT", 0.5, cast=float)
            ),
        ),
    )
    counterfactual_clip_bps = max(
        1.0,
        float(get_env("AI_TRADING_EXEC_COUNTERFACTUAL_CLIP_BPS", 50.0, cast=float)),
    )
    counterfactual_max_rank_uplift_abs = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_COUNTERFACTUAL_MAX_RANK_UPLIFT_ABS",
                20.0,
                cast=float,
            )
        ),
    )
    counterfactual_max_rank_uplift_frac = max(
        0.0,
        min(
            1.0,
            float(
                get_env(
                    "AI_TRADING_EXEC_COUNTERFACTUAL_MAX_RANK_UPLIFT_FRAC",
                    0.25,
                    cast=float,
                )
            ),
        ),
    )
    counterfactual_promote_min_events = max(
        counterfactual_min_samples,
        int(
            get_env(
                "AI_TRADING_EXEC_COUNTERFACTUAL_PROMOTE_MIN_EVENTS",
                500,
                cast=int,
            )
        ),
    )
    counterfactual_promote_min_dr_mean_bps = float(
        get_env(
            "AI_TRADING_EXEC_COUNTERFACTUAL_PROMOTE_MIN_DR_MEAN_BPS",
            0.0,
            cast=float,
        )
    )
    realized_edge_rank_enabled = bool(
        get_env("AI_TRADING_EXEC_REALIZED_EDGE_RANK_ENABLED", True, cast=bool)
    )
    realized_edge_rank_weight = max(
        0.0,
        min(
            5.0,
            float(
                get_env(
                    "AI_TRADING_EXEC_REALIZED_EDGE_SCORE_WEIGHT",
                    0.35,
                    cast=float,
                )
            ),
        ),
    )
    realized_edge_rank_min_samples = max(
        2,
        int(get_env("AI_TRADING_EXEC_REALIZED_EDGE_MIN_SAMPLES", 12, cast=int)),
    )
    realized_edge_rank_uncertainty_z = max(
        0.0,
        min(
            4.0,
            float(
                get_env(
                    "AI_TRADING_EXEC_REALIZED_EDGE_UNCERTAINTY_Z",
                    0.75,
                    cast=float,
                )
            ),
        ),
    )
    realized_edge_rank_clip_bps = max(
        1.0,
        float(get_env("AI_TRADING_EXEC_REALIZED_EDGE_CLIP_BPS", 60.0, cast=float)),
    )
    realized_edge_rank_max_rank_uplift_abs = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_REALIZED_EDGE_MAX_RANK_UPLIFT_ABS",
                18.0,
                cast=float,
            )
        ),
    )
    realized_edge_rank_max_rank_uplift_frac = max(
        0.0,
        min(
            1.0,
            float(
                get_env(
                    "AI_TRADING_EXEC_REALIZED_EDGE_MAX_RANK_UPLIFT_FRAC",
                    0.25,
                    cast=float,
                )
            ),
        ),
    )
    realized_edge_rank_session_bucket_enabled = bool(
        get_env(
            "AI_TRADING_EXEC_REALIZED_EDGE_SESSION_BUCKET_ENABLED",
            True,
            cast=bool,
        )
    )
    realized_edge_rank_regime_bucket_enabled = bool(
        get_env(
            "AI_TRADING_EXEC_REALIZED_EDGE_REGIME_BUCKET_ENABLED",
            True,
            cast=bool,
        )
    )
    edge_model_v2_enabled = bool(
        get_env("AI_TRADING_EXEC_EDGE_MODEL_V2_ENABLED", True, cast=bool)
    )
    edge_model_v2_weight = max(
        0.0,
        min(
            5.0,
            float(get_env("AI_TRADING_EXEC_EDGE_MODEL_V2_WEIGHT", 0.45, cast=float)),
        ),
    )
    edge_model_v2_min_samples = max(
        2,
        int(get_env("AI_TRADING_EXEC_EDGE_MODEL_V2_MIN_SAMPLES", 12, cast=int)),
    )
    edge_model_v2_uncertainty_z = max(
        0.0,
        min(
            5.0,
            float(
                get_env("AI_TRADING_EXEC_EDGE_MODEL_V2_UNCERTAINTY_Z", 1.0, cast=float)
            ),
        ),
    )
    edge_model_v2_regime_weight = max(
        0.0,
        float(get_env("AI_TRADING_EXEC_EDGE_MODEL_V2_REGIME_WEIGHT", 0.55, cast=float)),
    )
    edge_model_v2_session_weight = max(
        0.0,
        float(
            get_env("AI_TRADING_EXEC_EDGE_MODEL_V2_SESSION_WEIGHT", 0.30, cast=float)
        ),
    )
    edge_model_v2_global_weight = max(
        0.0,
        float(get_env("AI_TRADING_EXEC_EDGE_MODEL_V2_GLOBAL_WEIGHT", 0.15, cast=float)),
    )
    edge_model_v2_cost_weight = max(
        0.0,
        float(get_env("AI_TRADING_EXEC_EDGE_MODEL_V2_COST_WEIGHT", 1.0, cast=float)),
    )
    edge_model_v2_clip_bps = max(
        1.0,
        float(get_env("AI_TRADING_EXEC_EDGE_MODEL_V2_CLIP_BPS", 80.0, cast=float)),
    )
    edge_model_v2_max_rank_uplift_abs = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_EDGE_MODEL_V2_MAX_RANK_UPLIFT_ABS",
                22.0,
                cast=float,
            )
        ),
    )
    edge_model_v2_max_rank_uplift_frac = max(
        0.0,
        min(
            1.0,
            float(
                get_env(
                    "AI_TRADING_EXEC_EDGE_MODEL_V2_MAX_RANK_UPLIFT_FRAC",
                    0.30,
                    cast=float,
                )
            ),
        ),
    )
    promotion_significance_enabled = bool(
        get_env(
            "AI_TRADING_EXEC_PROMOTION_SEQUENTIAL_SIGNIFICANCE_ENABLED",
            True,
            cast=bool,
        )
    )
    promotion_significance_method = str(
        get_env(
            "AI_TRADING_EXEC_PROMOTION_SEQUENTIAL_METHOD",
            "either",
            cast=str,
        )
        or "either"
    ).strip().lower()
    if promotion_significance_method not in {"bayes", "sprt", "either", "both"}:
        promotion_significance_method = "either"
    promotion_significance_posterior_prob_min = max(
        0.5,
        min(
            0.999999,
            float(
                get_env(
                    "AI_TRADING_EXEC_PROMOTION_BAYES_POSTERIOR_MIN",
                    0.90,
                    cast=float,
                )
            ),
        ),
    )
    promotion_significance_sprt_alpha = max(
        1e-4,
        min(
            0.50,
            float(
                get_env(
                    "AI_TRADING_EXEC_PROMOTION_SPRT_ALPHA",
                    0.05,
                    cast=float,
                )
            ),
        ),
    )
    promotion_significance_sprt_beta = max(
        1e-4,
        min(
            0.50,
            float(
                get_env(
                    "AI_TRADING_EXEC_PROMOTION_SPRT_BETA",
                    0.10,
                    cast=float,
                )
            ),
        ),
    )
    promotion_significance_sprt_effect_bps = max(
        0.1,
        float(
            get_env(
                "AI_TRADING_EXEC_PROMOTION_SPRT_EFFECT_BPS",
                0.40,
                cast=float,
            )
        ),
    )
    opportunity_quality_enabled = bool(
        get_env("AI_TRADING_EXEC_OPPORTUNITY_QUALITY_ENABLED", True, cast=bool)
    )
    opportunity_top_quantile = max(
        0.05,
        min(
            0.99,
            float(
                get_env(
                    "AI_TRADING_EXEC_OPPORTUNITY_TOP_QUANTILE",
                    0.93,
                    cast=float,
                )
            ),
        ),
    )
    opportunity_min_symbols = max(
        1,
        int(get_env("AI_TRADING_EXEC_OPPORTUNITY_MIN_SYMBOLS", 5, cast=int)),
    )
    opportunity_openings_only = bool(
        get_env("AI_TRADING_EXEC_OPPORTUNITY_OPENINGS_ONLY", True, cast=bool)
    )
    alpha_time_decay_enabled = bool(
        get_env("AI_TRADING_EXEC_ALPHA_TIME_DECAY_ENABLED", True, cast=bool)
    )
    alpha_time_decay_half_life_sec = max(
        1.0,
        float(get_env("AI_TRADING_EXEC_ALPHA_HALF_LIFE_SEC", 900.0, cast=float)),
    )
    alpha_time_decay_floor = max(
        0.01,
        min(
            1.0,
            float(
                get_env(
                    "AI_TRADING_EXEC_ALPHA_TIME_DECAY_FLOOR",
                    0.10,
                    cast=float,
                )
            ),
        ),
    )
    alpha_stale_signal_sec = max(
        0.0,
        float(get_env("AI_TRADING_EXEC_ALPHA_STALE_SIGNAL_SEC", 3600.0, cast=float)),
    )
    alpha_time_stop_enabled = bool(
        get_env("AI_TRADING_EXEC_ALPHA_TIME_STOP_ENABLED", True, cast=bool)
    )
    alpha_time_stop_sec = max(
        0.0,
        float(get_env("AI_TRADING_EXEC_ALPHA_TIME_STOP_SEC", 14_400.0, cast=float)),
    )
    alpha_time_stop_max_expected_edge_bps = float(
        get_env(
            "AI_TRADING_EXEC_ALPHA_TIME_STOP_MAX_EXPECTED_EDGE_BPS",
            1.0,
            cast=float,
        )
    )
    exit_policy_entry_penalty_enabled = bool(
        get_env("AI_TRADING_EXEC_EXIT_POLICY_ENTRY_PENALTY_ENABLED", True, cast=bool)
    )
    exit_policy_entry_penalty_weight = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_EXIT_POLICY_ENTRY_PENALTY_WEIGHT",
                6.0,
                cast=float,
            )
        ),
    )
    exit_policy_entry_penalty_max_abs = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_EXIT_POLICY_ENTRY_PENALTY_MAX_ABS",
                12.0,
                cast=float,
            )
        ),
    )
    exit_policy_entry_penalty_min_pressure = max(
        0.0,
        min(
            1.0,
            float(
                get_env(
                    "AI_TRADING_EXEC_EXIT_POLICY_ENTRY_PENALTY_MIN_PRESSURE",
                    0.20,
                    cast=float,
                )
            ),
        ),
    )
    portfolio_log_growth_rank_enabled = _rollout_toggle(
        get_env,
        "AI_TRADING_EXEC_PORTFOLIO_LOG_GROWTH_RANK_ENABLED",
        False,
    )
    portfolio_log_growth_rank_weight = max(
        0.0,
        min(
            1.0,
            float(
                get_env(
                    "AI_TRADING_EXEC_PORTFOLIO_LOG_GROWTH_RANK_WEIGHT",
                    0.10,
                    cast=float,
                )
            ),
        ),
    )
    portfolio_log_growth_variance_penalty = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_PORTFOLIO_LOG_GROWTH_VARIANCE_PENALTY",
                1.0,
                cast=float,
            )
        ),
    )
    portfolio_log_growth_corr_penalty_bps = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_PORTFOLIO_LOG_GROWTH_CORR_PENALTY_BPS",
                8.0,
                cast=float,
            )
        ),
    )
    portfolio_log_growth_exposure_penalty_bps = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_PORTFOLIO_LOG_GROWTH_EXPOSURE_PENALTY_BPS",
                6.0,
                cast=float,
            )
        ),
    )
    portfolio_log_growth_turnover_penalty_bps = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_PORTFOLIO_LOG_GROWTH_TURNOVER_PENALTY_BPS",
                4.0,
                cast=float,
            )
        ),
    )
    portfolio_log_growth_liquidity_penalty_bps = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_PORTFOLIO_LOG_GROWTH_LIQUIDITY_IMPACT_PENALTY_BPS",
                6.0,
                cast=float,
            )
        ),
    )
    portfolio_log_growth_max_participation = max(
        0.005,
        min(
            0.5,
            float(
                get_env(
                    "AI_TRADING_EXEC_PORTFOLIO_LOG_GROWTH_MAX_PARTICIPATION",
                    0.05,
                    cast=float,
                )
            ),
        ),
    )
    portfolio_log_growth_max_adjust_abs = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_PORTFOLIO_LOG_GROWTH_MAX_ADJUST_ABS",
                20.0,
                cast=float,
            )
        ),
    )
    expected_capture_rank_enabled = bool(
        get_env("AI_TRADING_EXEC_EXPECTED_CAPTURE_RANK_ENABLED", True, cast=bool)
    )
    expected_capture_rank_weight = max(
        0.0,
        min(
            5.0,
            float(
                get_env(
                    "AI_TRADING_EXEC_EXPECTED_CAPTURE_RANK_WEIGHT",
                    0.45,
                    cast=float,
                )
            ),
        ),
    )
    expected_capture_spread_penalty_bps = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_SPREAD_PENALTY_BPS",
                0.40,
                cast=float,
            )
        ),
    )
    expected_capture_participation_penalty_bps = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_PARTICIPATION_PENALTY_BPS",
                12.0,
                cast=float,
            )
        ),
    )
    expected_capture_age_half_life_sec = max(
        10.0,
        float(
            get_env(
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_SIGNAL_HALF_LIFE_SEC",
                900.0,
                cast=float,
            )
        ),
    )
    expected_capture_fill_prob_floor = max(
        0.01,
        min(
            0.95,
            float(
                get_env(
                    "AI_TRADING_EXEC_EXPECTED_CAPTURE_FILL_PROB_FLOOR",
                    0.45,
                    cast=float,
                )
            ),
        ),
    )
    expected_capture_fill_prob_cap = max(
        expected_capture_fill_prob_floor,
        min(
            0.99,
            float(
                get_env(
                    "AI_TRADING_EXEC_EXPECTED_CAPTURE_FILL_PROB_CAP",
                    0.95,
                    cast=float,
                )
            ),
        ),
    )
    expected_capture_max_adjust_abs = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_MAX_ADJUST_ABS",
                10.0,
                cast=float,
            )
        ),
    )
    expected_capture_floor_bps = float(
        get_env(
            "AI_TRADING_EXEC_EXPECTED_CAPTURE_FLOOR_BPS",
            1.0,
            cast=float,
        )
    )
    expected_capture_constraint_weight = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_CONSTRAINT_WEIGHT",
                1.2,
                cast=float,
            )
        ),
    )
    expected_capture_constraint_max_adjust_abs = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_CONSTRAINT_MAX_ADJUST_ABS",
                12.0,
                cast=float,
            )
        ),
    )
    rank_downside_overlap_cap_enabled = bool(
        get_env("AI_TRADING_EXEC_RANK_DOWNSIDE_OVERLAP_CAP_ENABLED", True, cast=bool)
    )
    rank_downside_overlap_cap_frac = max(
        0.0,
        min(
            1.0,
            float(
                get_env(
                    "AI_TRADING_EXEC_RANK_DOWNSIDE_OVERLAP_CAP_FRAC",
                    0.55,
                    cast=float,
                )
            ),
        ),
    )
    rank_downside_overlap_cap_abs = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_RANK_DOWNSIDE_OVERLAP_CAP_ABS",
                18.0,
                cast=float,
            )
        ),
    )
    expected_capture_learned_fill_model_enabled = bool(
        get_env(
            "AI_TRADING_EXEC_EXPECTED_CAPTURE_LEARNED_FILL_MODEL_ENABLED",
            True,
            cast=bool,
        )
    )
    expected_capture_learned_fill_min_samples = max(
        1,
        int(
            get_env(
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_LEARNED_FILL_MIN_SAMPLES",
                40,
                cast=int,
            )
        ),
    )
    expected_capture_learned_fill_prior_alpha = max(
        0.01,
        float(
            get_env(
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_LEARNED_FILL_PRIOR_ALPHA",
                2.0,
                cast=float,
            )
        ),
    )
    expected_capture_learned_fill_prior_beta = max(
        0.01,
        float(
            get_env(
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_LEARNED_FILL_PRIOR_BETA",
                2.0,
                cast=float,
            )
        ),
    )
    expected_capture_liquidity_role_default = str(
        get_env(
            "AI_TRADING_EXEC_EXPECTED_CAPTURE_LIQUIDITY_ROLE_DEFAULT",
            "maker",
            cast=str,
        )
        or "maker"
    ).strip().lower()
    if expected_capture_liquidity_role_default not in {"maker", "taker", "mixed"}:
        expected_capture_liquidity_role_default = "maker"
    expected_capture_venue_default = str(
        get_env(
            "AI_TRADING_EXEC_EXPECTED_CAPTURE_VENUE_DEFAULT",
            "ALPACA",
            cast=str,
        )
        or "ALPACA"
    ).strip().upper() or "ALPACA"
    expected_capture_cost_model_enabled = bool(
        get_env(
            "AI_TRADING_EXEC_EXPECTED_CAPTURE_COST_MODEL_ENABLED",
            True,
            cast=bool,
        )
    )
    expected_capture_cost_model_min_samples = max(
        1,
        int(
            get_env(
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_COST_MODEL_MIN_SAMPLES",
                40,
                cast=int,
            )
        ),
    )
    expected_capture_cost_spread_weight = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_COST_SPREAD_WEIGHT",
                1.0,
                cast=float,
            )
        ),
    )
    expected_capture_cost_impact_weight = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_COST_IMPACT_WEIGHT",
                1.0,
                cast=float,
            )
        ),
    )
    expected_capture_cost_latency_weight = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_COST_LATENCY_WEIGHT",
                1.0,
                cast=float,
            )
        ),
    )
    expected_capture_latency_bps_per_sec = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_EXPECTED_CAPTURE_LATENCY_BPS_PER_SEC",
                0.02,
                cast=float,
            )
        ),
    )
    expected_capture_floor_adaptive_enabled = bool(
        get_env(
            "AI_TRADING_EXEC_EXPECTED_CAPTURE_FLOOR_ADAPTIVE_ENABLED",
            True,
            cast=bool,
        )
    )
    expected_capture_floor_adaptive_quantile = max(
        0.01,
        min(
            0.99,
            float(
                get_env(
                    "AI_TRADING_EXEC_EXPECTED_CAPTURE_FLOOR_ADAPTIVE_QUANTILE",
                    0.20,
                    cast=float,
                )
            ),
        ),
    )
    execution_learning_rank_enabled = bool(
        get_env(
            "AI_TRADING_EXECUTION_LEARNING_RANK_PENALTY_ENABLED",
            True,
            cast=bool,
        )
    )
    execution_learning_rank_min_samples = max(
        1,
        int(
            get_env(
                "AI_TRADING_EXECUTION_LEARNING_RANK_MIN_SAMPLES",
                2,
                cast=int,
            )
        ),
    )
    execution_learning_rank_slippage_floor_bps = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXECUTION_LEARNING_RANK_SLIPPAGE_FLOOR_BPS",
                4.0,
                cast=float,
            )
        ),
    )
    execution_learning_rank_slippage_weight = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXECUTION_LEARNING_RANK_SLIPPAGE_WEIGHT",
                0.30,
                cast=float,
            )
        ),
    )
    execution_learning_rank_negative_edge_weight = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXECUTION_LEARNING_RANK_NEGATIVE_EDGE_WEIGHT",
                0.15,
                cast=float,
            )
        ),
    )
    execution_learning_rank_adverse_weight = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXECUTION_LEARNING_RANK_ADVERSE_WEIGHT",
                0.10,
                cast=float,
            )
        ),
    )
    execution_learning_rank_realization_floor = max(
        0.0,
        min(
            2.0,
            float(
                get_env(
                    "AI_TRADING_EXECUTION_LEARNING_RANK_REALIZATION_FLOOR",
                    0.85,
                    cast=float,
                )
            ),
        ),
    )
    execution_learning_rank_realization_weight_bps = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXECUTION_LEARNING_RANK_REALIZATION_WEIGHT_BPS",
                8.0,
                cast=float,
            )
        ),
    )
    execution_learning_rank_max_penalty_bps = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXECUTION_LEARNING_RANK_MAX_PENALTY_BPS",
                35.0,
                cast=float,
            )
        ),
    )
    rejection_concentration_rank_enabled = bool(
        get_env(
            "AI_TRADING_EXEC_REJECTION_CONCENTRATION_RANK_ENABLED",
            True,
            cast=bool,
        )
    )
    rejection_concentration_window_records = max(
        100,
        int(
            get_env(
                "AI_TRADING_EXEC_REJECTION_CONCENTRATION_WINDOW_RECORDS",
                4000,
                cast=int,
            )
        ),
    )
    rejection_concentration_lookback_hours = max(
        1.0,
        float(
            get_env(
                "AI_TRADING_EXEC_REJECTION_CONCENTRATION_LOOKBACK_HOURS",
                18.0,
                cast=float,
            )
        ),
    )
    rejection_concentration_min_count = max(
        1,
        int(
            get_env(
                "AI_TRADING_EXEC_REJECTION_CONCENTRATION_MIN_COUNT",
                3,
                cast=int,
            )
        ),
    )
    rejection_concentration_scale_bps = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_REJECTION_CONCENTRATION_SCALE_BPS",
                0.45,
                cast=float,
            )
        ),
    )
    rejection_concentration_max_penalty_bps = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_REJECTION_CONCENTRATION_MAX_PENALTY_BPS",
                24.0,
                cast=float,
            )
        ),
    )
    replay_quality_rank_enabled = bool(
        get_env("AI_TRADING_EXEC_REPLAY_QUALITY_RANK_ENABLED", True, cast=bool)
    )
    replay_quality_session_bucket_enabled = bool(
        get_env(
            "AI_TRADING_EXEC_REPLAY_QUALITY_SESSION_BUCKET_ENABLED",
            True,
            cast=bool,
        )
    )
    replay_quality_regime_bucket_enabled = bool(
        get_env(
            "AI_TRADING_EXEC_REPLAY_QUALITY_REGIME_BUCKET_ENABLED",
            True,
            cast=bool,
        )
    )
    replay_quality_min_samples = max(
        1,
        int(get_env("AI_TRADING_EXEC_REPLAY_QUALITY_MIN_SAMPLES", 20, cast=int)),
    )
    replay_quality_weight = max(
        0.0,
        min(
            5.0,
            float(get_env("AI_TRADING_EXEC_REPLAY_QUALITY_WEIGHT", 0.18, cast=float)),
        ),
    )
    replay_quality_clip_bps = max(
        1.0,
        float(get_env("AI_TRADING_EXEC_REPLAY_QUALITY_CLIP_BPS", 25.0, cast=float)),
    )
    replay_quality_max_rank_uplift_abs = max(
        0.0,
        float(
            get_env(
                "AI_TRADING_EXEC_REPLAY_QUALITY_MAX_RANK_UPLIFT_ABS",
                8.0,
                cast=float,
            )
        ),
    )
    replay_quality_max_rank_uplift_frac = max(
        0.0,
        min(
            1.0,
            float(
                get_env(
                    "AI_TRADING_EXEC_REPLAY_QUALITY_MAX_RANK_UPLIFT_FRAC",
                    0.10,
                    cast=float,
                )
            ),
        ),
    )
    replay_quality_max_age_hours = max(
        1.0,
        float(
            get_env(
                "AI_TRADING_EXEC_REPLAY_QUALITY_MAX_AGE_HOURS",
                24.0,
                cast=float,
            )
        ),
    )
    replay_quality_auto_disable_if_stale = bool(
        get_env(
            "AI_TRADING_EXEC_REPLAY_QUALITY_AUTO_DISABLE_IF_STALE",
            True,
            cast=bool,
        )
    )
    replay_quality_fallback_to_edge_model_v2 = bool(
        get_env(
            "AI_TRADING_EXEC_REPLAY_QUALITY_FALLBACK_TO_EDGE_MODEL_V2",
            True,
            cast=bool,
        )
    )

    replay_quality_state = load_replay_quality_state(
        state=state,
        now=now,
        enabled=bool(replay_quality_rank_enabled),
        weight=float(replay_quality_weight),
        max_age_hours=float(replay_quality_max_age_hours),
        auto_disable_if_stale=bool(replay_quality_auto_disable_if_stale),
        get_env=get_env,
        safe_float=be._safe_float,
        parse_iso_timestamp=be._parse_iso_timestamp,
        resolve_runtime_artifact_path_func=be.resolve_runtime_artifact_path,
        load_latest_replay_quality_summaries_func=be._load_latest_replay_quality_summaries,
    )
    replay_quality_by_symbol = replay_quality_state.by_symbol
    replay_quality_by_symbol_session = replay_quality_state.by_symbol_session
    replay_quality_by_symbol_session_regime = replay_quality_state.by_symbol_session_regime
    replay_quality_context = replay_quality_state.context
    replay_quality_weight = replay_quality_state.effective_weight

    policy_override_state = apply_policy_runtime_overrides(
        load_policy_runtime_toggles_func=be._load_policy_runtime_toggles,
        bandit_enabled=bool(bandit_enabled),
        counterfactual_enabled=bool(counterfactual_enabled),
        geometric_tiebreak_enabled=bool(geometric_tiebreak_enabled),
        portfolio_log_growth_rank_enabled=bool(portfolio_log_growth_rank_enabled),
    )
    bandit_enabled = policy_override_state.bandit_enabled
    counterfactual_enabled = policy_override_state.counterfactual_enabled
    geometric_tiebreak_enabled = policy_override_state.geometric_tiebreak_enabled
    portfolio_log_growth_rank_enabled = (
        policy_override_state.portfolio_log_growth_rank_enabled
    )
    policy_runtime_payload = dict(policy_override_state.payload)
    toggles_raw = policy_runtime_payload.get("toggles")
    toggles = dict(toggles_raw) if isinstance(toggles_raw, Mapping) else {}
    policy_rollback_disabled_slices = _normalize_disabled_slices(
        policy_runtime_payload.get("disabled_slices")
    )
    state.policy_rollback_disabled_slices = sorted(policy_rollback_disabled_slices)
    policy_disabled_gate_roots = set(policy_override_state.disabled_gate_roots)
    policy_disabled_sleeves = set(policy_override_state.disabled_sleeves)

    learning_state = build_netting_learning_state(
        now=now,
        targets=targets,
        positions=positions,
        latest_price=latest_price,
        symbol_returns=symbol_returns,
        get_env=get_env,
        safe_float=be._safe_float,
        session_bucket_from_ts_func=be._session_bucket_from_ts,
        get_regime_signal_profile_func=be.get_regime_signal_profile,
        load_counterfactual_learning_state_func=be._load_counterfactual_learning_state,
        load_execution_learning_state_func=be._load_execution_learning_state,
        load_recent_rejection_concentration_by_symbol_func=be._load_recent_rejection_concentration_by_symbol,
        resolved_tca_path_func=be._resolved_tca_path,
        read_jsonl_records_func=be._read_jsonl_records,
        infer_tca_liquidity_role_func=be._infer_tca_liquidity_role,
        extract_tca_fill_success_ratio_func=be._extract_tca_fill_success_ratio,
        mean_std_func=be._mean_std,
        sequential_significance_gate_func=be._sequential_significance_gate,
        build_symbol_return_correlation_matrix_func=be._build_symbol_return_correlation_matrix,
        percentile_linear_func=be._percentile_linear,
        parse_iso_timestamp_func=be._parse_iso_timestamp,
        bandit_enabled=bool(bandit_enabled),
        bandit_method=str(bandit_method),
        bandit_min_samples=int(bandit_min_samples),
        bandit_window_trades=int(bandit_window_trades),
        bandit_session_bucket_enabled=bool(bandit_session_bucket_enabled),
        bandit_regime_bucket_enabled=bool(bandit_regime_bucket_enabled),
        bandit_shadow_only=bool(bandit_shadow_only),
        bandit_auto_promote=bool(bandit_auto_promote),
        bandit_promote_min_samples=int(bandit_promote_min_samples),
        bandit_promote_min_mean_reward_bps=float(bandit_promote_min_mean_reward_bps),
        bandit_bucket_promotion_enabled=bool(bandit_bucket_promotion_enabled),
        bandit_bucket_promote_min_samples=int(bandit_bucket_promote_min_samples),
        bandit_bucket_promote_min_mean_reward_bps=float(
            bandit_bucket_promote_min_mean_reward_bps
        ),
        edge_realism_rank_calibration_enabled=bool(edge_realism_rank_calibration_enabled),
        edge_realism_rank_calibration_session_enabled=bool(
            edge_realism_rank_calibration_session_enabled
        ),
        edge_realism_rank_calibration_min_samples=int(
            edge_realism_rank_calibration_min_samples
        ),
        edge_realism_rank_calibration_window_trades=int(
            edge_realism_rank_calibration_window_trades
        ),
        edge_realism_rank_calibration_prior_samples=int(
            edge_realism_rank_calibration_prior_samples
        ),
        edge_realism_rank_calibration_ratio_floor=float(
            edge_realism_rank_calibration_ratio_floor
        ),
        edge_realism_rank_calibration_ratio_cap=float(
            edge_realism_rank_calibration_ratio_cap
        ),
        counterfactual_enabled=bool(counterfactual_enabled),
        counterfactual_shadow_only=bool(counterfactual_shadow_only),
        counterfactual_auto_promote=bool(counterfactual_auto_promote),
        counterfactual_min_samples=int(counterfactual_min_samples),
        counterfactual_promote_min_events=int(counterfactual_promote_min_events),
        counterfactual_promote_min_dr_mean_bps=float(
            counterfactual_promote_min_dr_mean_bps
        ),
        realized_edge_rank_enabled=bool(realized_edge_rank_enabled),
        expected_capture_rank_enabled=bool(expected_capture_rank_enabled),
        expected_capture_learned_fill_model_enabled=bool(
            expected_capture_learned_fill_model_enabled
        ),
        expected_capture_learned_fill_min_samples=int(
            expected_capture_learned_fill_min_samples
        ),
        expected_capture_learned_fill_prior_alpha=float(
            expected_capture_learned_fill_prior_alpha
        ),
        expected_capture_learned_fill_prior_beta=float(
            expected_capture_learned_fill_prior_beta
        ),
        expected_capture_venue_default=str(expected_capture_venue_default),
        expected_capture_cost_model_enabled=bool(expected_capture_cost_model_enabled),
        expected_capture_cost_model_min_samples=int(
            expected_capture_cost_model_min_samples
        ),
        expected_capture_latency_bps_per_sec=float(expected_capture_latency_bps_per_sec),
        expected_capture_floor_bps=float(expected_capture_floor_bps),
        expected_capture_floor_adaptive_enabled=bool(
            expected_capture_floor_adaptive_enabled
        ),
        expected_capture_floor_adaptive_quantile=float(
            expected_capture_floor_adaptive_quantile
        ),
        execution_learning_rank_enabled=bool(execution_learning_rank_enabled),
        rejection_concentration_rank_enabled=bool(rejection_concentration_rank_enabled),
        rejection_concentration_window_records=int(rejection_concentration_window_records),
        rejection_concentration_lookback_hours=float(
            rejection_concentration_lookback_hours
        ),
        promotion_significance_enabled=bool(promotion_significance_enabled),
        promotion_significance_method=str(promotion_significance_method),
        promotion_significance_posterior_prob_min=float(
            promotion_significance_posterior_prob_min
        ),
        promotion_significance_sprt_alpha=float(promotion_significance_sprt_alpha),
        promotion_significance_sprt_beta=float(promotion_significance_sprt_beta),
        promotion_significance_sprt_effect_bps=float(
            promotion_significance_sprt_effect_bps
        ),
        opportunity_quality_enabled=bool(opportunity_quality_enabled),
        opportunity_top_quantile=float(opportunity_top_quantile),
        opportunity_min_symbols=int(opportunity_min_symbols),
        opportunity_openings_only=bool(opportunity_openings_only),
        portfolio_log_growth_rank_enabled=bool(portfolio_log_growth_rank_enabled),
    )
    edge_realism_rank_factor_by_symbol = learning_state.edge_realism_rank_factor_by_symbol
    counterfactual_signal_by_symbol = learning_state.counterfactual_signal_by_symbol
    opportunity_quality_by_symbol = learning_state.opportunity_quality_by_symbol
    opportunity_quality_gate = learning_state.opportunity_quality_gate

    ranking_result = rank_netting_candidates(
        now=now,
        targets=targets,
        net_edge_raw_by_symbol=net_edge_raw_by_symbol,
        latest_liquidity=latest_liquidity,
        latest_price=latest_price,
        positions=positions,
        symbol_returns=symbol_returns,
        edge_realism_rank_factor_by_symbol=edge_realism_rank_factor_by_symbol,
        counterfactual_signal_by_symbol_seed=counterfactual_signal_by_symbol,
        opportunity_quality_by_symbol_seed=opportunity_quality_by_symbol,
        learned_fill_trials_by_bucket=learning_state.learned_fill_trials_by_bucket,
        learned_fill_success_by_bucket=learning_state.learned_fill_success_by_bucket,
        learned_exec_cost_stats_by_bucket=learning_state.learned_exec_cost_stats_by_bucket,
        execution_learning_state=learning_state.execution_learning_state,
        rejection_concentration_by_symbol=learning_state.rejection_concentration_by_symbol,
        realized_edge_by_symbol=learning_state.realized_edge_by_symbol,
        realized_edge_by_symbol_session=learning_state.realized_edge_by_symbol_session,
        realized_edge_by_symbol_session_regime=learning_state.realized_edge_by_symbol_session_regime,
        bandit_rewards_by_symbol=learning_state.bandit_rewards_by_symbol,
        bandit_rewards_by_symbol_session=learning_state.bandit_rewards_by_symbol_session,
        bandit_rewards_by_symbol_session_regime=learning_state.bandit_rewards_by_symbol_session_regime,
        bandit_bucket_significance_cache=learning_state.bandit_bucket_significance_cache,
        counterfactual_buckets=learning_state.counterfactual_buckets,
        replay_quality_by_symbol=replay_quality_by_symbol,
        replay_quality_by_symbol_session=replay_quality_by_symbol_session,
        replay_quality_by_symbol_session_regime=replay_quality_by_symbol_session_regime,
        replay_quality_context=replay_quality_context,
        portfolio_rank_correlation=learning_state.portfolio_rank_correlation,
        held_notional_weights=learning_state.held_notional_weights,
        opportunity_quality_gate=opportunity_quality_gate,
        safe_float=be._safe_float,
        mean_std_func=be._mean_std,
        sequential_significance_gate_func=be._sequential_significance_gate,
        percentile_linear_func=be._percentile_linear,
        lookup_learned_fill_probability_func=lookup_learned_fill_probability,
        lookup_learned_execution_cost_components_func=lookup_learned_execution_cost_components,
        exit_policy_pressure_context_func=be._exit_policy_pressure_context,
        exit_policy_state=state,
        edge_realism_rank_calibration_enabled=bool(
            edge_realism_rank_calibration_enabled
        ),
        clip_expected_edge_enabled=bool(clip_expected_edge_enabled),
        edge_clip_cap_bps=float(edge_clip_cap_bps),
        alpha_time_decay_enabled=bool(alpha_time_decay_enabled),
        alpha_time_decay_half_life_sec=float(alpha_time_decay_half_life_sec),
        alpha_time_decay_floor=float(alpha_time_decay_floor),
        bandit_active_session=str(learning_state.bandit_active_session),
        bandit_active_regime=str(learning_state.bandit_active_regime),
        expected_capture_rank_enabled=bool(expected_capture_rank_enabled),
        expected_capture_learned_fill_model_enabled=bool(
            expected_capture_learned_fill_model_enabled
        ),
        expected_capture_learned_fill_min_samples=int(
            expected_capture_learned_fill_min_samples
        ),
        expected_capture_learned_fill_prior_alpha=float(
            expected_capture_learned_fill_prior_alpha
        ),
        expected_capture_learned_fill_prior_beta=float(
            expected_capture_learned_fill_prior_beta
        ),
        expected_capture_liquidity_role_default=str(
            expected_capture_liquidity_role_default
        ),
        expected_capture_venue_default=str(expected_capture_venue_default),
        execution_learning_rank_enabled=bool(execution_learning_rank_enabled),
        execution_learning_rank_min_samples=int(execution_learning_rank_min_samples),
        execution_learning_rank_slippage_floor_bps=float(
            execution_learning_rank_slippage_floor_bps
        ),
        execution_learning_rank_slippage_weight=float(
            execution_learning_rank_slippage_weight
        ),
        execution_learning_rank_negative_edge_weight=float(
            execution_learning_rank_negative_edge_weight
        ),
        execution_learning_rank_adverse_weight=float(
            execution_learning_rank_adverse_weight
        ),
        execution_learning_rank_realization_floor=float(
            execution_learning_rank_realization_floor
        ),
        execution_learning_rank_realization_weight_bps=float(
            execution_learning_rank_realization_weight_bps
        ),
        execution_learning_rank_max_penalty_bps=float(
            execution_learning_rank_max_penalty_bps
        ),
        expected_capture_fill_prob_floor=float(expected_capture_fill_prob_floor),
        expected_capture_fill_prob_cap=float(expected_capture_fill_prob_cap),
        expected_capture_spread_penalty_bps=float(
            expected_capture_spread_penalty_bps
        ),
        expected_capture_participation_penalty_bps=float(
            expected_capture_participation_penalty_bps
        ),
        expected_capture_age_half_life_sec=float(expected_capture_age_half_life_sec),
        expected_capture_latency_bps_per_sec=float(
            expected_capture_latency_bps_per_sec
        ),
        expected_capture_cost_model_enabled=bool(expected_capture_cost_model_enabled),
        expected_capture_cost_model_min_samples=int(
            expected_capture_cost_model_min_samples
        ),
        expected_capture_cost_spread_weight=float(
            expected_capture_cost_spread_weight
        ),
        expected_capture_cost_impact_weight=float(
            expected_capture_cost_impact_weight
        ),
        expected_capture_cost_latency_weight=float(
            expected_capture_cost_latency_weight
        ),
        rejection_concentration_rank_enabled=bool(
            rejection_concentration_rank_enabled
        ),
        rejection_concentration_min_count=int(rejection_concentration_min_count),
        rejection_concentration_scale_bps=float(rejection_concentration_scale_bps),
        rejection_concentration_max_penalty_bps=float(
            rejection_concentration_max_penalty_bps
        ),
        expected_capture_rank_weight=float(expected_capture_rank_weight),
        expected_capture_max_adjust_abs=float(expected_capture_max_adjust_abs),
        expected_capture_floor_bps_effective=float(
            learning_state.expected_capture_floor_bps_effective
        ),
        expected_capture_constraint_weight=float(expected_capture_constraint_weight),
        expected_capture_constraint_max_adjust_abs=float(
            expected_capture_constraint_max_adjust_abs
        ),
        edge_model_v2_enabled=bool(edge_model_v2_enabled),
        edge_model_v2_weight=float(edge_model_v2_weight),
        edge_model_v2_min_samples=int(edge_model_v2_min_samples),
        edge_model_v2_uncertainty_z=float(edge_model_v2_uncertainty_z),
        edge_model_v2_regime_weight=float(edge_model_v2_regime_weight),
        edge_model_v2_session_weight=float(edge_model_v2_session_weight),
        edge_model_v2_global_weight=float(edge_model_v2_global_weight),
        edge_model_v2_cost_weight=float(edge_model_v2_cost_weight),
        edge_model_v2_clip_bps=float(edge_model_v2_clip_bps),
        edge_model_v2_max_rank_uplift_abs=float(edge_model_v2_max_rank_uplift_abs),
        edge_model_v2_max_rank_uplift_frac=float(edge_model_v2_max_rank_uplift_frac),
        realized_edge_rank_enabled=bool(realized_edge_rank_enabled),
        realized_edge_rank_weight=float(realized_edge_rank_weight),
        realized_edge_rank_min_samples=int(realized_edge_rank_min_samples),
        realized_edge_rank_uncertainty_z=float(realized_edge_rank_uncertainty_z),
        realized_edge_rank_clip_bps=float(realized_edge_rank_clip_bps),
        realized_edge_rank_max_rank_uplift_abs=float(
            realized_edge_rank_max_rank_uplift_abs
        ),
        realized_edge_rank_max_rank_uplift_frac=float(
            realized_edge_rank_max_rank_uplift_frac
        ),
        realized_edge_rank_session_bucket_enabled=bool(
            realized_edge_rank_session_bucket_enabled
        ),
        realized_edge_rank_regime_bucket_enabled=bool(
            realized_edge_rank_regime_bucket_enabled
        ),
        bandit_enabled=bool(bandit_enabled),
        bandit_method=str(bandit_method),
        bandit_min_samples=int(bandit_min_samples),
        bandit_session_bucket_enabled=bool(bandit_session_bucket_enabled),
        bandit_regime_bucket_enabled=bool(bandit_regime_bucket_enabled),
        bandit_weight=float(bandit_weight),
        bandit_exploration=float(bandit_exploration),
        bandit_shadow_only=bool(bandit_shadow_only),
        bandit_live_promoted=bool(learning_state.bandit_live_promoted),
        bandit_bucket_promotion_enabled=bool(bandit_bucket_promotion_enabled),
        bandit_bucket_promote_min_samples=int(bandit_bucket_promote_min_samples),
        bandit_bucket_promote_min_mean_reward_bps=float(
            bandit_bucket_promote_min_mean_reward_bps
        ),
        bandit_promote_min_mean_reward_bps=float(
            bandit_promote_min_mean_reward_bps
        ),
        bandit_max_rank_uplift_abs=float(bandit_max_rank_uplift_abs),
        bandit_max_rank_uplift_frac=float(bandit_max_rank_uplift_frac),
        promotion_significance_enabled=bool(promotion_significance_enabled),
        promotion_significance_method=str(promotion_significance_method),
        promotion_significance_posterior_prob_min=float(
            promotion_significance_posterior_prob_min
        ),
        promotion_significance_sprt_alpha=float(promotion_significance_sprt_alpha),
        promotion_significance_sprt_beta=float(promotion_significance_sprt_beta),
        promotion_significance_sprt_effect_bps=float(
            promotion_significance_sprt_effect_bps
        ),
        counterfactual_enabled=bool(counterfactual_enabled),
        counterfactual_weight=float(counterfactual_weight),
        counterfactual_min_samples=int(counterfactual_min_samples),
        counterfactual_clip_bps=float(counterfactual_clip_bps),
        counterfactual_max_rank_uplift_abs=float(counterfactual_max_rank_uplift_abs),
        counterfactual_max_rank_uplift_frac=float(counterfactual_max_rank_uplift_frac),
        counterfactual_live_promoted=bool(learning_state.counterfactual_live_promoted),
        geometric_tiebreak_enabled=bool(geometric_tiebreak_enabled),
        geometric_tiebreak_weight=float(geometric_tiebreak_weight),
        geometric_variance_penalty=float(geometric_variance_penalty),
        geometric_downside_penalty=float(geometric_downside_penalty),
        geometric_drawdown_penalty=float(geometric_drawdown_penalty),
        portfolio_log_growth_rank_enabled=bool(portfolio_log_growth_rank_enabled),
        portfolio_log_growth_rank_weight=float(portfolio_log_growth_rank_weight),
        portfolio_log_growth_variance_penalty=float(
            portfolio_log_growth_variance_penalty
        ),
        portfolio_log_growth_corr_penalty_bps=float(
            portfolio_log_growth_corr_penalty_bps
        ),
        portfolio_log_growth_exposure_penalty_bps=float(
            portfolio_log_growth_exposure_penalty_bps
        ),
        portfolio_log_growth_turnover_penalty_bps=float(
            portfolio_log_growth_turnover_penalty_bps
        ),
        portfolio_log_growth_liquidity_penalty_bps=float(
            portfolio_log_growth_liquidity_penalty_bps
        ),
        portfolio_log_growth_max_participation=float(
            portfolio_log_growth_max_participation
        ),
        portfolio_log_growth_max_adjust_abs=float(
            portfolio_log_growth_max_adjust_abs
        ),
        portfolio_current_gross_for_rank=float(
            learning_state.portfolio_current_gross_for_rank
        ),
        replay_quality_rank_enabled=bool(replay_quality_rank_enabled),
        replay_quality_weight=float(replay_quality_weight),
        replay_quality_min_samples=int(replay_quality_min_samples),
        replay_quality_clip_bps=float(replay_quality_clip_bps),
        replay_quality_max_rank_uplift_abs=float(replay_quality_max_rank_uplift_abs),
        replay_quality_max_rank_uplift_frac=float(replay_quality_max_rank_uplift_frac),
        replay_quality_session_bucket_enabled=bool(
            replay_quality_session_bucket_enabled
        ),
        replay_quality_regime_bucket_enabled=bool(replay_quality_regime_bucket_enabled),
        replay_quality_fallback_to_edge_model_v2=bool(
            replay_quality_fallback_to_edge_model_v2
        ),
        exit_policy_entry_penalty_enabled=bool(
            exit_policy_entry_penalty_enabled
        ),
        exit_policy_entry_penalty_weight=float(exit_policy_entry_penalty_weight),
        exit_policy_entry_penalty_min_pressure=float(
            exit_policy_entry_penalty_min_pressure
        ),
        exit_policy_entry_penalty_max_abs=float(exit_policy_entry_penalty_max_abs),
        rank_downside_overlap_cap_enabled=bool(rank_downside_overlap_cap_enabled),
        rank_downside_overlap_cap_frac=float(rank_downside_overlap_cap_frac),
        rank_downside_overlap_cap_abs=float(rank_downside_overlap_cap_abs),
        opportunity_quality_enabled=bool(opportunity_quality_enabled),
        opportunity_top_quantile=float(opportunity_top_quantile),
        opportunity_min_symbols=int(opportunity_min_symbols),
        opportunity_demote_score=float(
            get_env(
                "AI_TRADING_EXEC_OPPORTUNITY_DEMOTE_SCORE",
                1_000_000.0,
                cast=float,
            )
        ),
    )

    store_candidate_ranking_runtime_state(
        runtime=runtime,
        ranking_result=ranking_result,
        edge_realism_rank_factor_by_symbol=edge_realism_rank_factor_by_symbol,
        context_values=locals(),
    )
    def _rank_rollout_toggle(primary_env: str, default: bool = False) -> bool:
        return _rollout_toggle(get_env, primary_env, default)

    portfolio_optimizer_runtime = prepare_portfolio_optimizer_runtime(
        latest_price=latest_price,
        symbol_returns=symbol_returns,
        rollout_toggle_func=_rank_rollout_toggle,
        get_env_func=get_env,
        build_symbol_return_correlation_matrix_func=be._build_symbol_return_correlation_matrix,
        logger=logger,
    )
    apply_target_construction_controls(
        targets=targets,
        cfg=cfg,
        normalized_symbol_returns=portfolio_optimizer_runtime.market_data["returns"],
        apply_global_caps_func=apply_global_caps,
        apply_portfolio_limits_func=be.apply_portfolio_limits,
        get_env_func=get_env,
        logger=logger,
    )
    return NettingRankRuntimeResult(
        ranking_result=ranking_result,
        portfolio_optimizer_runtime=portfolio_optimizer_runtime,
        edge_realism_rank_factor_by_symbol=dict(edge_realism_rank_factor_by_symbol),
        edge_realism_apply_to_approval_enabled=bool(
            edge_realism_apply_to_approval_enabled
        ),
        policy_runtime_payload=policy_runtime_payload,
        policy_disabled_gate_roots=policy_disabled_gate_roots,
        policy_disabled_sleeves=policy_disabled_sleeves,
        policy_rollback_disabled_slices=policy_rollback_disabled_slices,
        opportunity_quality_enabled=bool(opportunity_quality_enabled),
        opportunity_top_quantile=float(opportunity_top_quantile),
        opportunity_openings_only=bool(opportunity_openings_only),
        alpha_time_decay_enabled=bool(alpha_time_decay_enabled),
        alpha_stale_signal_sec=float(alpha_stale_signal_sec),
        alpha_time_stop_enabled=bool(alpha_time_stop_enabled),
        alpha_time_stop_sec=float(alpha_time_stop_sec),
        alpha_time_stop_max_expected_edge_bps=float(
            alpha_time_stop_max_expected_edge_bps
        ),
    )


__all__ = [
    "NettingRankRuntimeResult",
    "build_netting_rank_runtime",
]
