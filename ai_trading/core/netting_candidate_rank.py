"""Candidate ranking helpers for the live netting cycle."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
from statistics import NormalDist
from typing import Any, Callable, Mapping, Sequence

from ai_trading.core.netting import NettedTarget
from ai_trading.risk.liquidity_regime import LiquidityFeatures


@dataclass(slots=True)
class NettingCandidateRankingResult:
    candidate_expected_net_edge: dict[str, float]
    candidate_expected_capture: dict[str, float]
    candidate_rank: dict[str, float]
    counterfactual_signal_by_symbol: dict[str, dict[str, Any]]
    opportunity_quality_by_symbol: dict[str, float]
    opportunity_allowed_symbols: set[str]
    opportunity_quality_gate: dict[str, Any]


def _bandit_ucb_score(
    *,
    mean_reward_bps: float,
    samples: int,
    total_samples: int,
    exploration: float,
) -> float:
    if samples <= 0:
        return float(mean_reward_bps)
    safe_total = max(int(total_samples), int(samples), 1)
    exploration_term = float(exploration) * math.sqrt(
        max(math.log(float(safe_total)), 0.0) / float(max(samples, 1))
    )
    return float(mean_reward_bps) + float(exploration_term)


def _geometric_growth_tiebreak_score(
    *,
    expected_edge_bps: float,
    returns_window: Sequence[float],
    drawdown: float,
    variance_penalty: float,
    downside_penalty: float,
    drawdown_penalty: float,
) -> float:
    expected_return = float(expected_edge_bps) / 10000.0
    clean_returns = [
        float(value)
        for value in returns_window
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if clean_returns:
        sample_count = float(len(clean_returns))
        mean_return = float(sum(clean_returns) / sample_count)
        variance = float(
            sum((value - mean_return) ** 2 for value in clean_returns) / sample_count
        )
        downside = float(
            math.sqrt(
                sum((min(value, 0.0) ** 2) for value in clean_returns) / sample_count
            )
        )
    else:
        mean_return = 0.0
        variance = 0.0
        downside = 0.0
    drawdown_term = max(float(drawdown), 0.0) * abs(expected_return)
    growth_score = (
        expected_return
        + mean_return
        - (max(float(variance_penalty), 0.0) * variance)
        - (max(float(downside_penalty), 0.0) * downside)
        - (max(float(drawdown_penalty), 0.0) * drawdown_term)
    )
    return float(growth_score * 10000.0)


def _execution_learning_bucket_keys(
    *,
    symbol: str,
    session_token: str,
    regime_token: str,
    liquidity_role: str,
    side: str,
) -> tuple[str, ...]:
    symbol_token = str(symbol or "").strip().upper() or "UNKNOWN"
    session_value = str(session_token or "").strip().lower() or "offhours"
    regime_value = str(regime_token or "").strip().lower() or "unknown"
    liquidity_value = str(liquidity_role or "").strip().lower() or "mixed"
    side_value = str(side or "").strip().lower() or "buy"
    role_values = [liquidity_value]
    for profile_value in ("balanced", "passive", "urgent", "protective"):
        if profile_value not in role_values:
            role_values.append(profile_value)
    keys: list[str] = []
    for role_value in role_values:
        keys.extend(
            (
                f"{symbol_token}:{session_value}:{regime_value}:{role_value}:{side_value}",
                f"{symbol_token}:{session_value}:{role_value}:{side_value}",
                f"{session_value}:{role_value}:{side_value}",
            )
        )
    keys.extend(
        (
            f"{symbol_token}:{session_value}:{regime_value}:{side_value}",
            f"{symbol_token}:{session_value}:{side_value}",
            f"{symbol_token}:{session_value}",
            f"{symbol_token}",
            "global",
        )
    )
    return tuple(dict.fromkeys(keys))


def _lookup_execution_learning_bucket_entry(
    *,
    state: Mapping[str, Any],
    symbol: str,
    session_token: str,
    regime_token: str,
    liquidity_role: str,
    side: str,
    min_samples: int,
    safe_float: Callable[[Any], float | None],
) -> tuple[dict[str, Any] | None, str | None]:
    buckets_raw = state.get("buckets")
    buckets = dict(buckets_raw) if isinstance(buckets_raw, Mapping) else {}
    symbol_buckets_raw = state.get("symbol_buckets")
    symbol_buckets = (
        dict(symbol_buckets_raw) if isinstance(symbol_buckets_raw, Mapping) else {}
    )
    for key in _execution_learning_bucket_keys(
        symbol=symbol,
        session_token=session_token,
        regime_token=regime_token,
        liquidity_role=liquidity_role,
        side=side,
    ):
        candidates = (
            symbol_buckets.get(str(key)),
            symbol_buckets.get(str(key).lower()),
            buckets.get(str(key)),
            buckets.get(str(key).lower()),
        )
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            samples = int(safe_float(candidate.get("samples")) or 0.0)
            if samples < int(min_samples):
                continue
            return (dict(candidate), str(key))
    return (None, None)


def _compute_execution_learning_rank_penalty(
    *,
    entry: Mapping[str, Any] | None,
    min_samples: int,
    slippage_floor_bps: float,
    slippage_weight: float,
    negative_edge_weight: float,
    adverse_weight: float,
    realization_floor: float,
    realization_weight_bps: float,
    max_penalty_bps: float,
    safe_float: Callable[[Any], float | None],
) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        return {
            "samples": 0,
            "penalty_bps": 0.0,
            "blend": 0.0,
            "slippage_penalty_bps": 0.0,
            "negative_edge_penalty_bps": 0.0,
            "adverse_penalty_bps": 0.0,
            "realization_penalty_bps": 0.0,
            "mean_fill_probability": None,
            "fill_rate": None,
        }
    samples = max(0, int(safe_float(entry.get("samples")) or 0.0))
    blend = min(1.0, float(samples) / float(max(1, 2 * int(min_samples))))
    mean_slippage = max(0.0, float(safe_float(entry.get("mean_slippage_bps")) or 0.0))
    mean_net_edge = float(safe_float(entry.get("mean_net_edge_bps")) or 0.0)
    mean_adverse = max(
        0.0,
        float(safe_float(entry.get("mean_adverse_selection_risk_bps")) or 0.0),
    )
    realization_ratio = max(
        0.0,
        float(safe_float(entry.get("mean_realization_ratio")) or 1.0),
    )
    mean_fill_probability = safe_float(entry.get("mean_fill_probability"))
    fill_rate = safe_float(entry.get("fill_rate"))
    slippage_penalty = max(0.0, mean_slippage - float(slippage_floor_bps)) * float(
        slippage_weight
    )
    negative_edge_penalty = max(0.0, -mean_net_edge) * float(negative_edge_weight)
    adverse_penalty = max(0.0, mean_adverse) * float(adverse_weight)
    realization_penalty = max(
        0.0,
        float(realization_floor) - float(realization_ratio),
    ) * float(realization_weight_bps)
    penalty = float(blend) * (
        float(slippage_penalty)
        + float(negative_edge_penalty)
        + float(adverse_penalty)
        + float(realization_penalty)
    )
    penalty = max(0.0, min(float(max_penalty_bps), float(penalty)))
    return {
        "samples": int(samples),
        "blend": float(blend),
        "penalty_bps": float(penalty),
        "slippage_penalty_bps": float(slippage_penalty * float(blend)),
        "negative_edge_penalty_bps": float(negative_edge_penalty * float(blend)),
        "adverse_penalty_bps": float(adverse_penalty * float(blend)),
        "realization_penalty_bps": float(realization_penalty * float(blend)),
        "mean_fill_probability": (
            float(mean_fill_probability) if mean_fill_probability is not None else None
        ),
        "fill_rate": float(fill_rate) if fill_rate is not None else None,
    }


def _compute_rejection_concentration_penalty_bps(
    *,
    counts: Mapping[str, Any] | None,
    min_count: int,
    scale_bps: float,
    max_penalty_bps: float,
    safe_float: Callable[[Any], float | None],
) -> dict[str, float]:
    if not isinstance(counts, Mapping):
        return {
            "total": 0.0,
            "weighted_count": 0.0,
            "penalty_bps": 0.0,
            "pre_execution": 0.0,
            "slippage": 0.0,
            "capacity": 0.0,
            "portfolio": 0.0,
        }
    total = max(0.0, float(safe_float(counts.get("total")) or 0.0))
    pre_execution = max(
        0.0, float(safe_float(counts.get("pre_execution")) or 0.0)
    )
    slippage = max(0.0, float(safe_float(counts.get("slippage")) or 0.0))
    capacity = max(0.0, float(safe_float(counts.get("capacity")) or 0.0))
    portfolio = max(0.0, float(safe_float(counts.get("portfolio")) or 0.0))
    weighted_count = (
        (0.25 * float(total))
        + (1.50 * float(pre_execution))
        + (2.00 * float(slippage))
        + (1.00 * float(capacity))
        + (0.50 * float(portfolio))
    )
    effective_excess = max(0.0, float(weighted_count) - float(max(0, int(min_count))))
    penalty = max(0.0, min(float(max_penalty_bps), effective_excess * float(scale_bps)))
    return {
        "total": float(total),
        "weighted_count": float(weighted_count),
        "penalty_bps": float(penalty),
        "pre_execution": float(pre_execution),
        "slippage": float(slippage),
        "capacity": float(capacity),
        "portfolio": float(portfolio),
    }


def rank_netting_candidates(  # noqa: PLR0913, C901
    *,
    now: datetime,
    targets: Mapping[str, NettedTarget],
    net_edge_raw_by_symbol: Mapping[str, float],
    latest_liquidity: Mapping[str, LiquidityFeatures],
    latest_price: Mapping[str, float],
    positions: Mapping[str, float],
    symbol_returns: Mapping[str, Sequence[float]],
    edge_realism_rank_factor_by_symbol: Mapping[str, float],
    counterfactual_signal_by_symbol_seed: Mapping[str, Mapping[str, Any]],
    opportunity_quality_by_symbol_seed: Mapping[str, float],
    learned_fill_trials_by_bucket: Mapping[str, float],
    learned_fill_success_by_bucket: Mapping[str, float],
    learned_exec_cost_stats_by_bucket: Mapping[str, Mapping[str, float]],
    execution_learning_state: Mapping[str, Any],
    rejection_concentration_by_symbol: Mapping[str, Any],
    realized_edge_by_symbol: Mapping[str, Sequence[float]],
    realized_edge_by_symbol_session: Mapping[str, Sequence[float]],
    realized_edge_by_symbol_session_regime: Mapping[str, Sequence[float]],
    bandit_rewards_by_symbol: Mapping[str, Sequence[float]],
    bandit_rewards_by_symbol_session: Mapping[str, Sequence[float]],
    bandit_rewards_by_symbol_session_regime: Mapping[str, Sequence[float]],
    bandit_bucket_significance_cache: dict[str, dict[str, Any]],
    counterfactual_buckets: Mapping[str, Any],
    replay_quality_by_symbol: Mapping[str, Mapping[str, Any]],
    replay_quality_by_symbol_session: Mapping[str, Mapping[str, Any]],
    replay_quality_by_symbol_session_regime: Mapping[str, Mapping[str, Any]],
    replay_quality_context: Mapping[str, Any],
    portfolio_rank_correlation: Mapping[str, Mapping[str, float]],
    held_notional_weights: Mapping[str, float],
    opportunity_quality_gate: Mapping[str, Any],
    safe_float: Callable[[Any], float | None],
    mean_std_func: Callable[[Sequence[float]], tuple[float, float, int]],
    sequential_significance_gate_func: Callable[..., dict[str, Any]],
    percentile_linear_func: Callable[[Sequence[float], float], float | None],
    lookup_learned_fill_probability_func: Callable[..., tuple[float | None, int, str | None]],
    lookup_learned_execution_cost_components_func: Callable[..., tuple[float | None, float | None, float | None, int, str | None]],
    exit_policy_pressure_context_func: Callable[..., dict[str, Any]],
    exit_policy_state: Any,
    edge_realism_rank_calibration_enabled: bool,
    clip_expected_edge_enabled: bool,
    edge_clip_cap_bps: float,
    alpha_time_decay_enabled: bool,
    alpha_time_decay_half_life_sec: float,
    alpha_time_decay_floor: float,
    bandit_active_session: str,
    bandit_active_regime: str,
    expected_capture_rank_enabled: bool,
    expected_capture_learned_fill_model_enabled: bool,
    expected_capture_learned_fill_min_samples: int,
    expected_capture_learned_fill_prior_alpha: float,
    expected_capture_learned_fill_prior_beta: float,
    expected_capture_liquidity_role_default: str,
    expected_capture_venue_default: str,
    execution_learning_rank_enabled: bool,
    execution_learning_rank_min_samples: int,
    execution_learning_rank_slippage_floor_bps: float,
    execution_learning_rank_slippage_weight: float,
    execution_learning_rank_negative_edge_weight: float,
    execution_learning_rank_adverse_weight: float,
    execution_learning_rank_realization_floor: float,
    execution_learning_rank_realization_weight_bps: float,
    execution_learning_rank_max_penalty_bps: float,
    expected_capture_fill_prob_floor: float,
    expected_capture_fill_prob_cap: float,
    expected_capture_spread_penalty_bps: float,
    expected_capture_participation_penalty_bps: float,
    expected_capture_age_half_life_sec: float,
    expected_capture_latency_bps_per_sec: float,
    expected_capture_cost_model_enabled: bool,
    expected_capture_cost_model_min_samples: int,
    expected_capture_cost_spread_weight: float,
    expected_capture_cost_impact_weight: float,
    expected_capture_cost_latency_weight: float,
    rejection_concentration_rank_enabled: bool,
    rejection_concentration_min_count: int,
    rejection_concentration_scale_bps: float,
    rejection_concentration_max_penalty_bps: float,
    expected_capture_rank_weight: float,
    expected_capture_max_adjust_abs: float,
    expected_capture_floor_bps_effective: float,
    expected_capture_constraint_weight: float,
    expected_capture_constraint_max_adjust_abs: float,
    edge_model_v2_enabled: bool,
    edge_model_v2_weight: float,
    edge_model_v2_min_samples: int,
    edge_model_v2_uncertainty_z: float,
    edge_model_v2_regime_weight: float,
    edge_model_v2_session_weight: float,
    edge_model_v2_global_weight: float,
    edge_model_v2_cost_weight: float,
    edge_model_v2_clip_bps: float,
    edge_model_v2_max_rank_uplift_abs: float,
    edge_model_v2_max_rank_uplift_frac: float,
    realized_edge_rank_enabled: bool,
    realized_edge_rank_weight: float,
    realized_edge_rank_min_samples: int,
    realized_edge_rank_uncertainty_z: float,
    realized_edge_rank_clip_bps: float,
    realized_edge_rank_max_rank_uplift_abs: float,
    realized_edge_rank_max_rank_uplift_frac: float,
    realized_edge_rank_session_bucket_enabled: bool,
    realized_edge_rank_regime_bucket_enabled: bool,
    bandit_enabled: bool,
    bandit_method: str,
    bandit_min_samples: int,
    bandit_session_bucket_enabled: bool,
    bandit_regime_bucket_enabled: bool,
    bandit_weight: float,
    bandit_exploration: float,
    bandit_shadow_only: bool,
    bandit_live_promoted: bool,
    bandit_bucket_promotion_enabled: bool,
    bandit_bucket_promote_min_samples: int,
    bandit_bucket_promote_min_mean_reward_bps: float,
    bandit_promote_min_mean_reward_bps: float,
    bandit_max_rank_uplift_abs: float,
    bandit_max_rank_uplift_frac: float,
    promotion_significance_enabled: bool,
    promotion_significance_method: str,
    promotion_significance_posterior_prob_min: float,
    promotion_significance_sprt_alpha: float,
    promotion_significance_sprt_beta: float,
    promotion_significance_sprt_effect_bps: float,
    counterfactual_enabled: bool,
    counterfactual_weight: float,
    counterfactual_min_samples: int,
    counterfactual_clip_bps: float,
    counterfactual_max_rank_uplift_abs: float,
    counterfactual_max_rank_uplift_frac: float,
    counterfactual_live_promoted: bool,
    geometric_tiebreak_enabled: bool,
    geometric_tiebreak_weight: float,
    geometric_variance_penalty: float,
    geometric_downside_penalty: float,
    geometric_drawdown_penalty: float,
    portfolio_log_growth_rank_enabled: bool,
    portfolio_log_growth_rank_weight: float,
    portfolio_log_growth_variance_penalty: float,
    portfolio_log_growth_corr_penalty_bps: float,
    portfolio_log_growth_exposure_penalty_bps: float,
    portfolio_log_growth_turnover_penalty_bps: float,
    portfolio_log_growth_liquidity_penalty_bps: float,
    portfolio_log_growth_max_participation: float,
    portfolio_log_growth_max_adjust_abs: float,
    portfolio_current_gross_for_rank: float,
    replay_quality_rank_enabled: bool,
    replay_quality_weight: float,
    replay_quality_min_samples: int,
    replay_quality_clip_bps: float,
    replay_quality_max_rank_uplift_abs: float,
    replay_quality_max_rank_uplift_frac: float,
    replay_quality_session_bucket_enabled: bool,
    replay_quality_regime_bucket_enabled: bool,
    replay_quality_fallback_to_edge_model_v2: bool,
    exit_policy_entry_penalty_enabled: bool,
    exit_policy_entry_penalty_weight: float,
    exit_policy_entry_penalty_min_pressure: float,
    exit_policy_entry_penalty_max_abs: float,
    rank_downside_overlap_cap_enabled: bool,
    rank_downside_overlap_cap_frac: float,
    rank_downside_overlap_cap_abs: float,
    opportunity_quality_enabled: bool,
    opportunity_top_quantile: float,
    opportunity_min_symbols: int,
    opportunity_demote_score: float,
) -> NettingCandidateRankingResult:
    counterfactual_signal_by_symbol = {
        str(symbol): dict(payload) if isinstance(payload, Mapping) else {}
        for symbol, payload in counterfactual_signal_by_symbol_seed.items()
    }
    opportunity_quality_by_symbol = {
        str(symbol): float(score)
        for symbol, score in opportunity_quality_by_symbol_seed.items()
    }
    opportunity_quality_gate_value = dict(opportunity_quality_gate)
    candidate_expected_net_edge: dict[str, float] = {}
    candidate_expected_capture: dict[str, float] = {}
    candidate_rank: dict[str, float] = {}

    for symbol, target in targets.items():
        raw_net_edge_total = float(net_edge_raw_by_symbol.get(symbol, 0.0) or 0.0)
        px_for_qty = safe_float(latest_price.get(symbol))
        current_shares = safe_float(positions.get(symbol))
        current_notional = 0.0
        if px_for_qty is not None and px_for_qty > 0.0 and current_shares is not None:
            current_notional = float(current_shares) * float(px_for_qty)
        candidate_trade_notional = float(target.target_dollars) - float(current_notional)
        candidate_side = "buy" if float(candidate_trade_notional) >= 0.0 else "sell_short"
        current_notional_abs = abs(float(current_notional))
        target_notional_abs = abs(float(target.target_dollars))
        trade_notional_abs = abs(float(candidate_trade_notional))
        current_position_side = (
            1
            if float(current_notional) > 1e-9
            else (-1 if float(current_notional) < -1e-9 else 0)
        )
        target_position_side = (
            1
            if float(target.target_dollars) > 1e-9
            else (-1 if float(target.target_dollars) < -1e-9 else 0)
        )
        candidate_expands_exposure = (
            target_position_side != 0
            and (
                current_position_side == 0
                or current_position_side != target_position_side
                or float(target_notional_abs) > float(current_notional_abs)
            )
        )
        edge_realism_rank_factor = float(
            edge_realism_rank_factor_by_symbol.get(symbol, 1.0)
        )
        adjusted_net_edge_total = float(raw_net_edge_total)
        if edge_realism_rank_calibration_enabled and adjusted_net_edge_total > 0.0:
            adjusted_net_edge_total = float(adjusted_net_edge_total) * float(
                edge_realism_rank_factor
            )
        clipped_net_edge_total = (
            max(
                -float(edge_clip_cap_bps),
                min(float(edge_clip_cap_bps), adjusted_net_edge_total),
            )
            if clip_expected_edge_enabled
            else adjusted_net_edge_total
        )
        candidate_expected_net_edge[symbol] = float(clipped_net_edge_total)
        max_conf = max(
            (float(proposal.confidence) for proposal in target.proposals),
            default=0.0,
        )
        disagreement = (
            float(target.disagreement_ratio)
            if target.disagreement_ratio is not None
            else 1.0
        )
        if not math.isfinite(disagreement):
            disagreement = 1.0
        disagreement = max(0.05, min(disagreement, 1.0))
        rank_score = (
            max(clipped_net_edge_total, -1000.0)
            * max(max_conf, 0.05)
            * disagreement
        )
        signal_age_seconds = max(
            0.0,
            (now - target.bar_ts).total_seconds()
            if isinstance(target.bar_ts, datetime)
            else 0.0,
        )
        time_decay_multiplier = 1.0
        if alpha_time_decay_enabled and alpha_time_decay_half_life_sec > 0.0:
            try:
                decay = math.exp(
                    -math.log(2.0)
                    * (float(signal_age_seconds) / float(alpha_time_decay_half_life_sec))
                )
            except (ValueError, OverflowError):
                decay = 1.0
            time_decay_multiplier = max(
                float(alpha_time_decay_floor),
                min(1.0, float(decay)),
            )
            rank_score *= float(time_decay_multiplier)
        rank_score_baseline = float(rank_score)
        expected_capture_bps = float(clipped_net_edge_total)
        expected_capture_fill_probability = 0.5
        expected_capture_spread_bps = 0.0
        expected_capture_participation = 0.0
        expected_capture_impact_cost_bps = 0.0
        expected_capture_latency_drift_bps = 0.0
        expected_capture_fill_model_source = "heuristic"
        expected_capture_fill_model_samples = 0
        expected_capture_cost_model_source = "heuristic"
        expected_capture_cost_model_samples = 0
        execution_learning_penalty_bps = 0.0
        execution_learning_penalty_context: dict[str, Any] = {}
        execution_learning_source = "none"
        rejection_penalty_bps = 0.0
        rejection_penalty_context: dict[str, Any] = {}
        if expected_capture_rank_enabled:
            liq_features = latest_liquidity.get(symbol)
            spread_bps_value = max(
                0.0,
                safe_float(getattr(liq_features, "spread_bps", 0.0)) or 0.0,
            )
            rolling_volume = max(
                0.0,
                safe_float(getattr(liq_features, "rolling_volume", 0.0)) or 0.0,
            )
            trade_qty = 0.0
            if px_for_qty is not None and px_for_qty > 0.0:
                trade_qty = abs(float(candidate_trade_notional)) / float(px_for_qty)
            participation = 0.0
            if rolling_volume > 0.0 and trade_qty > 0.0:
                participation = min(
                    2.0,
                    float(trade_qty) / float(max(rolling_volume, 1.0)),
                )
            fill_prob_proxy = 0.65
            fill_prob_proxy -= min(float(spread_bps_value) / 35.0, 0.50)
            fill_prob_proxy -= min(float(participation) / 0.08, 0.55)
            if expected_capture_age_half_life_sec > 0.0:
                try:
                    age_decay = math.exp(
                        -math.log(2.0)
                        * (
                            float(signal_age_seconds)
                            / float(expected_capture_age_half_life_sec)
                        )
                    )
                except (ValueError, OverflowError):
                    age_decay = 1.0
                fill_prob_proxy *= max(0.20, min(1.0, float(age_decay)))
            assumed_liquidity_role = str(expected_capture_liquidity_role_default)
            venue_session = f"{expected_capture_venue_default}:{bandit_active_session}"
            if expected_capture_learned_fill_model_enabled:
                learned_fill_prob, learned_fill_samples, learned_fill_key = (
                    lookup_learned_fill_probability_func(
                        learned_fill_trials_by_bucket=learned_fill_trials_by_bucket,
                        learned_fill_success_by_bucket=learned_fill_success_by_bucket,
                        expected_capture_learned_fill_min_samples=int(
                            expected_capture_learned_fill_min_samples
                        ),
                        expected_capture_learned_fill_prior_alpha=float(
                            expected_capture_learned_fill_prior_alpha
                        ),
                        expected_capture_learned_fill_prior_beta=float(
                            expected_capture_learned_fill_prior_beta
                        ),
                        symbol=str(symbol),
                        session_token=str(bandit_active_session),
                        regime_token=str(bandit_active_regime),
                        liquidity_role=str(assumed_liquidity_role),
                        venue_session=str(venue_session),
                    )
                )
                if learned_fill_prob is not None:
                    fill_prob_proxy = float(learned_fill_prob)
                    expected_capture_fill_model_source = str(
                        learned_fill_key or "learned"
                    )
                    expected_capture_fill_model_samples = int(learned_fill_samples)
            if execution_learning_rank_enabled:
                execution_learning_entry, execution_learning_source_key = (
                    _lookup_execution_learning_bucket_entry(
                        state=execution_learning_state,
                        symbol=str(symbol),
                        session_token=str(bandit_active_session),
                        regime_token=str(bandit_active_regime),
                        liquidity_role=str(assumed_liquidity_role),
                        side=str(candidate_side),
                        min_samples=int(execution_learning_rank_min_samples),
                        safe_float=safe_float,
                    )
                )
                execution_learning_source = str(
                    execution_learning_source_key or "none"
                )
                execution_learning_penalty_context = dict(
                    _compute_execution_learning_rank_penalty(
                        entry=execution_learning_entry,
                        min_samples=int(execution_learning_rank_min_samples),
                        slippage_floor_bps=float(
                            execution_learning_rank_slippage_floor_bps
                        ),
                        slippage_weight=float(
                            execution_learning_rank_slippage_weight
                        ),
                        negative_edge_weight=float(
                            execution_learning_rank_negative_edge_weight
                        ),
                        adverse_weight=float(
                            execution_learning_rank_adverse_weight
                        ),
                        realization_floor=float(
                            execution_learning_rank_realization_floor
                        ),
                        realization_weight_bps=float(
                            execution_learning_rank_realization_weight_bps
                        ),
                        max_penalty_bps=float(
                            execution_learning_rank_max_penalty_bps
                        ),
                        safe_float=safe_float,
                    )
                )
                execution_fill_cap = safe_float(
                    execution_learning_penalty_context.get("mean_fill_probability")
                )
                if execution_fill_cap is None or execution_fill_cap <= 0.0:
                    execution_fill_cap = safe_float(
                        execution_learning_penalty_context.get("fill_rate")
                    )
                if execution_fill_cap is not None and execution_fill_cap > 0.0:
                    fill_prob_proxy = min(
                        float(fill_prob_proxy),
                        max(
                            float(expected_capture_fill_prob_floor),
                            min(1.0, float(execution_fill_cap)),
                        ),
                    )
            expected_capture_fill_probability = max(
                float(expected_capture_fill_prob_floor),
                min(float(expected_capture_fill_prob_cap), float(fill_prob_proxy)),
            )
            spread_cost_bps = float(expected_capture_spread_penalty_bps) * float(
                spread_bps_value
            )
            impact_cost_bps = float(
                expected_capture_participation_penalty_bps
            ) * float(participation)
            latency_drift_cost_bps = (
                max(
                    0.0,
                    float(signal_age_seconds)
                    / max(float(expected_capture_age_half_life_sec), 1.0),
                )
                * float(expected_capture_latency_bps_per_sec)
            )
            if expected_capture_cost_model_enabled:
                (
                    learned_spread_cost,
                    learned_impact_cost,
                    learned_latency_cost,
                    learned_cost_samples,
                    learned_cost_key,
                ) = lookup_learned_execution_cost_components_func(
                    learned_exec_cost_stats_by_bucket=learned_exec_cost_stats_by_bucket,
                    safe_float=safe_float,
                    expected_capture_cost_model_min_samples=int(
                        expected_capture_cost_model_min_samples
                    ),
                    symbol=str(symbol),
                    session_token=str(bandit_active_session),
                    regime_token=str(bandit_active_regime),
                    liquidity_role=str(assumed_liquidity_role),
                    venue_session=str(venue_session),
                )
                if learned_cost_samples > 0:
                    blend = min(
                        1.0,
                        float(learned_cost_samples)
                        / float(max(1, 2 * expected_capture_cost_model_min_samples)),
                    )
                    if learned_spread_cost is not None:
                        spread_cost_bps = (
                            (1.0 - float(blend)) * float(spread_cost_bps)
                        ) + (float(blend) * float(learned_spread_cost))
                    if learned_impact_cost is not None:
                        impact_cost_bps = (
                            (1.0 - float(blend)) * float(impact_cost_bps)
                        ) + (float(blend) * float(learned_impact_cost))
                    if learned_latency_cost is not None:
                        latency_drift_cost_bps = (
                            (1.0 - float(blend))
                            * float(latency_drift_cost_bps)
                        ) + (float(blend) * float(learned_latency_cost))
                    expected_capture_cost_model_source = str(
                        learned_cost_key or "learned"
                    )
                    expected_capture_cost_model_samples = int(learned_cost_samples)
            modeled_execution_cost_bps = (
                (float(expected_capture_cost_spread_weight) * float(spread_cost_bps))
                + (float(expected_capture_cost_impact_weight) * float(impact_cost_bps))
                + (
                    float(expected_capture_cost_latency_weight)
                    * float(latency_drift_cost_bps)
                )
            )
            execution_learning_penalty_bps = float(
                execution_learning_penalty_context.get("penalty_bps", 0.0) or 0.0
            )
            modeled_execution_cost_bps += float(execution_learning_penalty_bps)
            if rejection_concentration_rank_enabled:
                rejection_penalty_context = dict(
                    _compute_rejection_concentration_penalty_bps(
                        counts=rejection_concentration_by_symbol.get(str(symbol)),
                        min_count=int(rejection_concentration_min_count),
                        scale_bps=float(rejection_concentration_scale_bps),
                        max_penalty_bps=float(
                            rejection_concentration_max_penalty_bps
                        ),
                        safe_float=safe_float,
                    )
                )
                rejection_penalty_bps = float(
                    rejection_penalty_context.get("penalty_bps", 0.0) or 0.0
                )
                modeled_execution_cost_bps += float(rejection_penalty_bps)
            expected_capture_bps = (
                float(clipped_net_edge_total) * float(expected_capture_fill_probability)
            ) - float(modeled_execution_cost_bps)
            capture_bonus = (
                float(expected_capture_rank_weight)
                * float(expected_capture_bps)
                * max(max_conf, 0.05)
                * disagreement
            )
            if expected_capture_max_adjust_abs > 0.0:
                capture_bonus = max(
                    -float(expected_capture_max_adjust_abs),
                    min(float(expected_capture_max_adjust_abs), float(capture_bonus)),
                )
            rank_score += float(capture_bonus)
            capture_constraint_penalty = max(
                0.0,
                float(expected_capture_floor_bps_effective)
                - float(expected_capture_bps),
            )
            if (
                capture_constraint_penalty > 0.0
                and expected_capture_constraint_weight > 0.0
            ):
                constraint_adjust = (
                    float(capture_constraint_penalty)
                    * float(expected_capture_constraint_weight)
                    * max(max_conf, 0.05)
                    * disagreement
                )
                if expected_capture_constraint_max_adjust_abs > 0.0:
                    constraint_adjust = min(
                        float(expected_capture_constraint_max_adjust_abs),
                        float(constraint_adjust),
                    )
                rank_score -= float(constraint_adjust)
            target.reasons.append("EXPECTED_CAPTURE_OPTIMIZER")
            if (
                expected_capture_fill_model_source != "heuristic"
                or expected_capture_cost_model_source != "heuristic"
            ):
                target.reasons.append("EXPECTED_CAPTURE_MODEL_LEARNED")
            if execution_learning_penalty_bps > 0.0:
                target.reasons.append("EXECUTION_LEARNING_DEWEIGHT")
            if rejection_penalty_bps > 0.0:
                target.reasons.append("REJECTION_CONCENTRATION_DEWEIGHT")
            expected_capture_spread_bps = float(spread_bps_value)
            expected_capture_participation = float(participation)
            expected_capture_impact_cost_bps = float(impact_cost_bps)
            expected_capture_latency_drift_bps = float(latency_drift_cost_bps)
            counterfactual_signal_by_symbol.setdefault(symbol, {}).update(
                {
                    "expected_capture_bps": float(expected_capture_bps),
                    "expected_capture_fill_probability": float(
                        expected_capture_fill_probability
                    ),
                    "expected_capture_spread_bps": float(
                        expected_capture_spread_bps
                    ),
                    "expected_capture_participation": float(
                        expected_capture_participation
                    ),
                    "expected_capture_impact_cost_bps": float(
                        expected_capture_impact_cost_bps
                    ),
                    "expected_capture_latency_drift_bps": float(
                        expected_capture_latency_drift_bps
                    ),
                    "expected_capture_cost_model_source": str(
                        expected_capture_cost_model_source
                    ),
                    "expected_capture_cost_model_samples": int(
                        expected_capture_cost_model_samples
                    ),
                    "expected_capture_fill_model_source": str(
                        expected_capture_fill_model_source
                    ),
                    "expected_capture_fill_model_samples": int(
                        expected_capture_fill_model_samples
                    ),
                    "expected_capture_floor_bps": float(
                        expected_capture_floor_bps_effective
                    ),
                    "execution_learning_source": str(execution_learning_source),
                    "execution_learning_penalty_bps": float(
                        execution_learning_penalty_bps
                    ),
                    "execution_learning_penalty_context": dict(
                        execution_learning_penalty_context
                    ),
                    "rejection_concentration_penalty_bps": float(
                        rejection_penalty_bps
                    ),
                    "rejection_concentration_penalty_context": dict(
                        rejection_penalty_context
                    ),
                }
            )
        candidate_expected_capture[symbol] = float(expected_capture_bps)
        edge_model_v2_source = "none"
        edge_model_v2_samples = 0
        edge_model_v2_expected_bps = 0.0
        edge_model_v2_uncertainty_bps = 0.0
        edge_model_v2_cost_penalty_bps = 0.0
        edge_model_v2_target_bps = 0.0
        if edge_model_v2_enabled and edge_model_v2_weight > 0.0:
            global_series = realized_edge_by_symbol.get(symbol, [])
            session_series = realized_edge_by_symbol_session.get(
                f"{symbol}:{bandit_active_session}",
                [],
            )
            regime_series = realized_edge_by_symbol_session_regime.get(
                f"{symbol}:{bandit_active_session}:{bandit_active_regime}",
                [],
            )
            global_mean, global_std, global_n = mean_std_func(global_series)
            session_mean, session_std, session_n = mean_std_func(session_series)
            regime_mean, regime_std, regime_n = mean_std_func(regime_series)
            blend_terms: list[tuple[str, float, float, int, float]] = []
            if regime_n >= edge_model_v2_min_samples:
                blend_terms.append(
                    (
                        "regime",
                        float(edge_model_v2_regime_weight),
                        float(regime_mean),
                        int(regime_n),
                        float(regime_std),
                    )
                )
            if session_n >= edge_model_v2_min_samples:
                blend_terms.append(
                    (
                        "session",
                        float(edge_model_v2_session_weight),
                        float(session_mean),
                        int(session_n),
                        float(session_std),
                    )
                )
            if global_n >= edge_model_v2_min_samples:
                blend_terms.append(
                    (
                        "global",
                        float(edge_model_v2_global_weight),
                        float(global_mean),
                        int(global_n),
                        float(global_std),
                    )
                )
            total_weight = float(sum(max(term[1], 0.0) for term in blend_terms))
            if blend_terms and total_weight > 0.0:
                blended_mean = float(
                    sum(
                        (max(weight, 0.0) / total_weight) * mean
                        for _, weight, mean, _n, _std in blend_terms
                    )
                )
                blended_stderr = float(
                    sum(
                        (max(weight, 0.0) / total_weight)
                        * (max(float(std), 0.0) / math.sqrt(float(max(samples, 1))))
                        for _, weight, _mean, samples, std in blend_terms
                    )
                )
                edge_model_v2_expected_bps = float(blended_mean)
                edge_model_v2_uncertainty_bps = float(
                    float(edge_model_v2_uncertainty_z) * float(blended_stderr)
                )
                edge_model_v2_cost_penalty_bps = float(
                    float(edge_model_v2_cost_weight)
                    * (
                        max(0.0, float(expected_capture_impact_cost_bps))
                        + max(0.0, float(expected_capture_latency_drift_bps))
                        + (0.50 * max(0.0, float(expected_capture_spread_bps)))
                    )
                )
                edge_model_v2_target_bps = max(
                    -float(edge_model_v2_clip_bps),
                    min(
                        float(edge_model_v2_clip_bps),
                        float(edge_model_v2_expected_bps)
                        - float(edge_model_v2_uncertainty_bps)
                        - float(edge_model_v2_cost_penalty_bps),
                    ),
                )
                edge_model_v2_samples = int(max(regime_n, session_n, global_n))
                edge_model_v2_source = "+".join(term[0] for term in blend_terms)
                edge_model_v2_bonus = (
                    float(edge_model_v2_weight)
                    * float(edge_model_v2_target_bps)
                    * max(max_conf, 0.05)
                    * disagreement
                )
                edge_model_v2_bonus_cap = float(edge_model_v2_max_rank_uplift_abs)
                if edge_model_v2_max_rank_uplift_frac > 0.0:
                    edge_model_v2_bonus_cap = min(
                        edge_model_v2_bonus_cap,
                        abs(float(rank_score))
                        * float(edge_model_v2_max_rank_uplift_frac),
                    )
                if edge_model_v2_bonus_cap > 0.0:
                    edge_model_v2_bonus = max(
                        -float(edge_model_v2_bonus_cap),
                        min(
                            float(edge_model_v2_bonus_cap),
                            float(edge_model_v2_bonus),
                        ),
                    )
                rank_score += float(edge_model_v2_bonus)
                target.reasons.append("EDGE_MODEL_V2")
                if "regime" in edge_model_v2_source or "session" in edge_model_v2_source:
                    target.reasons.append("EDGE_MODEL_V2_REGIME_BLEND")
                counterfactual_signal_by_symbol.setdefault(symbol, {}).update(
                    {
                        "edge_model_v2_source": str(edge_model_v2_source),
                        "edge_model_v2_samples": int(edge_model_v2_samples),
                        "edge_model_v2_expected_bps": float(edge_model_v2_expected_bps),
                        "edge_model_v2_uncertainty_bps": float(
                            edge_model_v2_uncertainty_bps
                        ),
                        "edge_model_v2_cost_penalty_bps": float(
                            edge_model_v2_cost_penalty_bps
                        ),
                        "edge_model_v2_target_bps": float(edge_model_v2_target_bps),
                        "edge_model_v2_weight": float(edge_model_v2_weight),
                    }
                )
        realized_rank_source = "none"
        realized_rank_samples = 0
        realized_rank_mean_bps = 0.0
        realized_rank_uncertainty_penalty_bps = 0.0
        realized_rank_target_bps = 0.0
        replay_quality_source = "none"
        replay_quality_samples = 0
        replay_quality_net_edge_bps = 0.0
        replay_quality_target_bps = 0.0
        replay_quality_bonus = 0.0
        exit_policy_entry_penalty_bps = 0.0
        exit_policy_entry_context: dict[str, Any] = {"enabled": False}
        if realized_edge_rank_enabled and realized_edge_rank_weight > 0.0:
            realized_series = realized_edge_by_symbol.get(symbol, [])
            realized_rank_source = "symbol"
            if realized_edge_rank_regime_bucket_enabled:
                regime_series = realized_edge_by_symbol_session_regime.get(
                    f"{symbol}:{bandit_active_session}:{bandit_active_regime}",
                    [],
                )
                if len(regime_series) >= realized_edge_rank_min_samples:
                    realized_series = regime_series
                    realized_rank_source = "symbol_session_regime"
            if realized_edge_rank_session_bucket_enabled:
                session_series = realized_edge_by_symbol_session.get(
                    f"{symbol}:{bandit_active_session}",
                    [],
                )
                if len(session_series) >= realized_edge_rank_min_samples:
                    realized_series = session_series
                    realized_rank_source = "symbol_session"
            realized_mean, realized_std, realized_samples = mean_std_func(
                realized_series
            )
            if realized_samples >= realized_edge_rank_min_samples:
                realized_rank_samples = int(realized_samples)
                realized_rank_mean_bps = float(realized_mean)
                realized_rank_uncertainty_penalty_bps = float(
                    float(realized_edge_rank_uncertainty_z)
                    * (
                        float(realized_std)
                        / math.sqrt(float(max(realized_samples, 1)))
                    )
                )
                realized_rank_target_bps = max(
                    -float(realized_edge_rank_clip_bps),
                    min(
                        float(realized_edge_rank_clip_bps),
                        float(realized_rank_mean_bps)
                        - float(realized_rank_uncertainty_penalty_bps),
                    ),
                )
                realized_bonus = (
                    float(realized_edge_rank_weight)
                    * float(realized_rank_target_bps)
                    * max(max_conf, 0.05)
                    * disagreement
                )
                realized_bonus_cap = float(realized_edge_rank_max_rank_uplift_abs)
                if realized_edge_rank_max_rank_uplift_frac > 0.0:
                    realized_bonus_cap = min(
                        realized_bonus_cap,
                        abs(float(rank_score))
                        * float(realized_edge_rank_max_rank_uplift_frac),
                    )
                if realized_bonus_cap > 0.0:
                    realized_bonus = max(
                        -realized_bonus_cap,
                        min(realized_bonus_cap, float(realized_bonus)),
                    )
                rank_score += float(realized_bonus)
                counterfactual_signal_by_symbol.setdefault(symbol, {}).update(
                    {
                        "realized_rank_enabled": True,
                        "realized_rank_source": str(realized_rank_source),
                        "realized_rank_samples": int(realized_rank_samples),
                        "realized_rank_mean_bps": float(realized_rank_mean_bps),
                        "realized_rank_uncertainty_penalty_bps": float(
                            realized_rank_uncertainty_penalty_bps
                        ),
                        "realized_rank_target_bps": float(realized_rank_target_bps),
                        "realized_rank_weight": float(realized_edge_rank_weight),
                    }
                )
        if bandit_enabled:
            reward_series = bandit_rewards_by_symbol.get(symbol, [])
            bandit_source = "symbol"
            if bandit_regime_bucket_enabled:
                session_regime_series = bandit_rewards_by_symbol_session_regime.get(
                    f"{symbol}:{bandit_active_session}:{bandit_active_regime}",
                    [],
                )
                if len(session_regime_series) >= bandit_min_samples:
                    reward_series = session_regime_series
                    bandit_source = "symbol_session_regime"
            if bandit_session_bucket_enabled:
                session_series = bandit_rewards_by_symbol_session.get(
                    f"{symbol}:{bandit_active_session}",
                    [],
                )
                if len(session_series) >= bandit_min_samples:
                    reward_series = session_series
                    bandit_source = "symbol_session"
            if len(reward_series) >= bandit_min_samples:
                sample_count = int(len(reward_series))
                mean_reward = float(sum(reward_series) / max(sample_count, 1))
                bucket_gate_key = (
                    f"{symbol}:{bandit_source}:{bandit_active_session}:"
                    f"{bandit_active_regime}"
                )
                bucket_significance_context = bandit_bucket_significance_cache.get(
                    bucket_gate_key
                )
                if bucket_significance_context is None:
                    bucket_mean, bucket_std, bucket_samples = mean_std_func(
                        reward_series
                    )
                    bucket_significance_context = sequential_significance_gate_func(
                        mean_reward_bps=float(bucket_mean),
                        std_reward_bps=float(bucket_std),
                        samples=int(bucket_samples),
                        min_samples=int(
                            max(
                                bandit_min_samples,
                                bandit_bucket_promote_min_samples,
                            )
                        ),
                        target_mean_bps=float(
                            bandit_bucket_promote_min_mean_reward_bps
                        ),
                        method=str(promotion_significance_method),
                        posterior_prob_min=float(
                            promotion_significance_posterior_prob_min
                        ),
                        sprt_alpha=float(promotion_significance_sprt_alpha),
                        sprt_beta=float(promotion_significance_sprt_beta),
                        sprt_effect_bps=float(promotion_significance_sprt_effect_bps),
                    )
                    bandit_bucket_significance_cache[bucket_gate_key] = dict(
                        bucket_significance_context
                    )
                bucket_significance_pass = (
                    bool(bucket_significance_context.get("passed", False))
                    if promotion_significance_enabled
                    else True
                )
                bandit_bucket_live_promoted = bool(
                    bandit_live_promoted
                    and (
                        not bandit_bucket_promotion_enabled
                        or (
                            sample_count >= bandit_bucket_promote_min_samples
                            and mean_reward
                            >= bandit_bucket_promote_min_mean_reward_bps
                            and bucket_significance_pass
                        )
                    )
                )
                if bandit_method == "thompson":
                    variance = float(
                        sum((value - mean_reward) ** 2 for value in reward_series)
                        / max(sample_count, 1)
                    )
                    std_error = math.sqrt(max(variance, 0.0) / max(sample_count, 1))
                    digest = hashlib.blake2b(
                        f"{symbol}:{bandit_active_session}:{now.isoformat()}".encode(
                            "utf-8"
                        ),
                        digest_size=8,
                    ).digest()
                    u01 = int.from_bytes(
                        digest, byteorder="big", signed=False
                    ) / float(2**64)
                    u01 = min(max(u01, 1e-6), 1.0 - 1e-6)
                    sampled_z = float(NormalDist().inv_cdf(u01))
                    sampled_z = max(-3.0, min(3.0, sampled_z))
                    bandit_score = float(
                        mean_reward
                        + sampled_z * float(bandit_exploration) * float(std_error)
                    )
                else:
                    total_samples = int(
                        sum(
                            len(values)
                            for key, values in bandit_rewards_by_symbol_session.items()
                            if key.endswith(f":{bandit_active_session}")
                        )
                    )
                    if total_samples <= 0:
                        total_samples = int(
                            sum(
                                len(values)
                                for values in bandit_rewards_by_symbol.values()
                            )
                        )
                    bandit_score = _bandit_ucb_score(
                        mean_reward_bps=mean_reward,
                        samples=sample_count,
                        total_samples=total_samples,
                        exploration=bandit_exploration,
                    )
                bandit_bonus = (
                    float(bandit_weight)
                    * float(bandit_score)
                    * max(max_conf, 0.05)
                    * disagreement
                )
                bandit_bonus_cap = float(bandit_max_rank_uplift_abs)
                if bandit_max_rank_uplift_frac > 0.0:
                    bandit_bonus_cap = min(
                        bandit_bonus_cap,
                        abs(float(rank_score)) * float(bandit_max_rank_uplift_frac),
                    )
                if bandit_bonus_cap > 0.0:
                    bandit_bonus = max(
                        -bandit_bonus_cap,
                        min(bandit_bonus_cap, bandit_bonus),
                    )
                if bandit_bucket_live_promoted:
                    rank_score += float(bandit_bonus)
                    target.reasons.append(
                        f"BANDIT_{bandit_method.upper()}_{bandit_source.upper()}"
                    )
                else:
                    target.reasons.append(
                        f"BANDIT_SHADOW_{bandit_method.upper()}_{bandit_source.upper()}"
                    )
                counterfactual_signal_by_symbol.setdefault(symbol, {}).update(
                    {
                        "bandit_source": str(bandit_source),
                        "bandit_samples": int(sample_count),
                        "bandit_mean_reward_bps": float(mean_reward),
                        "bandit_score_bps": float(bandit_score),
                        "bandit_bonus": float(bandit_bonus),
                        "bandit_live_promoted": bool(bandit_live_promoted),
                        "bandit_bucket_live_promoted": bool(
                            bandit_bucket_live_promoted
                        ),
                        "bandit_bucket_significance": dict(
                            bucket_significance_context
                        ),
                    }
                )
        if counterfactual_enabled and counterfactual_weight > 0.0:
            counterfactual_bucket_key = f"{symbol}:{bandit_active_session}"
            counterfactual_bucket_raw = counterfactual_buckets.get(
                counterfactual_bucket_key
            )
            if not isinstance(counterfactual_bucket_raw, Mapping):
                counterfactual_bucket_raw = counterfactual_buckets.get(
                    f"{symbol}:offhours"
                )
            if not isinstance(counterfactual_bucket_raw, Mapping):
                counterfactual_bucket_raw = {}
            cf_events = int(counterfactual_bucket_raw.get("events", 0) or 0)
            cf_dr_mean_bps = safe_float(counterfactual_bucket_raw.get("dr_mean_bps"))
            if cf_dr_mean_bps is None:
                cf_sum = safe_float(counterfactual_bucket_raw.get("dr_sum_bps"))
                if cf_sum is not None and cf_events > 0:
                    cf_dr_mean_bps = float(cf_sum) / float(cf_events)
            if cf_dr_mean_bps is not None and cf_events >= counterfactual_min_samples:
                bounded_cf = max(
                    -float(counterfactual_clip_bps),
                    min(float(counterfactual_clip_bps), float(cf_dr_mean_bps)),
                )
                cf_bonus = (
                    float(counterfactual_weight)
                    * float(bounded_cf)
                    * max(max_conf, 0.05)
                    * disagreement
                )
                cf_bonus_cap = float(counterfactual_max_rank_uplift_abs)
                if counterfactual_max_rank_uplift_frac > 0.0:
                    cf_bonus_cap = min(
                        cf_bonus_cap,
                        abs(float(rank_score))
                        * float(counterfactual_max_rank_uplift_frac),
                    )
                if cf_bonus_cap > 0.0:
                    cf_bonus = max(-cf_bonus_cap, min(cf_bonus_cap, cf_bonus))
                counterfactual_signal_by_symbol.setdefault(symbol, {}).update(
                    {
                        "counterfactual_bucket": str(counterfactual_bucket_key),
                        "counterfactual_events": int(cf_events),
                        "counterfactual_dr_mean_bps": float(cf_dr_mean_bps),
                        "counterfactual_bonus": float(cf_bonus),
                        "counterfactual_live_promoted": bool(
                            counterfactual_live_promoted
                        ),
                    }
                )
                if counterfactual_live_promoted:
                    rank_score += float(cf_bonus)
                    target.reasons.append("COUNTERFACTUAL_DR")
                else:
                    target.reasons.append("COUNTERFACTUAL_DR_SHADOW")
        if geometric_tiebreak_enabled and geometric_tiebreak_weight > 0.0:
            returns_window = symbol_returns.get(symbol, [])
            cumulative = 0.0
            peak = 0.0
            max_drawdown = 0.0
            for value in returns_window:
                cumulative += float(value)
                peak = max(peak, cumulative)
                max_drawdown = max(max_drawdown, peak - cumulative)
            growth_score = _geometric_growth_tiebreak_score(
                expected_edge_bps=float(clipped_net_edge_total),
                returns_window=returns_window,
                drawdown=float(max_drawdown),
                variance_penalty=float(geometric_variance_penalty),
                downside_penalty=float(geometric_downside_penalty),
                drawdown_penalty=float(geometric_drawdown_penalty),
            )
            rank_score += (
                float(geometric_tiebreak_weight)
                * float(growth_score)
                * max(max_conf, 0.05)
                * disagreement
            )
        if portfolio_log_growth_rank_enabled and portfolio_log_growth_rank_weight > 0.0:
            returns_window = symbol_returns.get(symbol, [])
            if returns_window:
                mean_return = float(sum(float(value) for value in returns_window)) / float(
                    max(1, len(returns_window))
                )
                variance = float(
                    sum((float(value) - mean_return) ** 2 for value in returns_window)
                    / float(max(1, len(returns_window)))
                )
            else:
                variance = 0.0
            corr_penalty = 0.0
            corr_row = portfolio_rank_correlation.get(symbol, {})
            if isinstance(corr_row, Mapping) and held_notional_weights:
                for held_symbol, held_weight in held_notional_weights.items():
                    corr_value = safe_float(corr_row.get(held_symbol))
                    if corr_value is None:
                        continue
                    corr_penalty += abs(float(corr_value)) * float(held_weight)
            exposure_penalty = 0.0
            if portfolio_current_gross_for_rank > 0.0:
                incremental_exposure = max(
                    0.0,
                    float(target_notional_abs) - float(current_notional_abs),
                )
                exposure_penalty = float(incremental_exposure) / float(
                    max(portfolio_current_gross_for_rank, 1.0)
                )
            turnover_penalty = 0.0
            if portfolio_current_gross_for_rank > 0.0:
                turnover_penalty = float(trade_notional_abs) / float(
                    max(portfolio_current_gross_for_rank, 1.0)
                )
            liquidity_impact_penalty = 0.0
            liq_features = latest_liquidity.get(symbol)
            spread_bps_for_impact = max(
                safe_float(getattr(liq_features, "spread_bps", 0.0)) or 0.0,
                0.0,
            )
            rolling_volume_for_impact = max(
                safe_float(getattr(liq_features, "rolling_volume", 0.0)) or 0.0,
                0.0,
            )
            if rolling_volume_for_impact > 0.0:
                trade_qty = abs(
                    float(candidate_trade_notional)
                    / max(float(latest_price.get(symbol, 0.0) or 0.0), 1e-9)
                )
                participation = float(trade_qty) / float(
                    max(rolling_volume_for_impact, 1.0)
                )
                participation_ratio = min(
                    2.0,
                    float(participation)
                    / float(portfolio_log_growth_max_participation),
                )
                liquidity_impact_penalty = (
                    (float(spread_bps_for_impact) / 10.0)
                    * float(participation_ratio)
                )
            portfolio_log_growth_base_bps = (
                min(float(clipped_net_edge_total), float(expected_capture_bps))
                if expected_capture_rank_enabled
                else float(clipped_net_edge_total)
            )
            marginal_log_growth_bps = float(portfolio_log_growth_base_bps) - (
                float(portfolio_log_growth_variance_penalty) * float(variance) * 10_000.0
            )
            marginal_log_growth_bps -= (
                float(portfolio_log_growth_corr_penalty_bps) * float(corr_penalty)
            )
            marginal_log_growth_bps -= (
                float(portfolio_log_growth_exposure_penalty_bps)
                * float(exposure_penalty)
            )
            marginal_log_growth_bps -= (
                float(portfolio_log_growth_turnover_penalty_bps)
                * float(turnover_penalty)
            )
            marginal_log_growth_bps -= (
                float(portfolio_log_growth_liquidity_penalty_bps)
                * float(liquidity_impact_penalty)
            )
            portfolio_bonus = (
                float(portfolio_log_growth_rank_weight)
                * float(marginal_log_growth_bps)
                * max(max_conf, 0.05)
                * disagreement
            )
            if portfolio_log_growth_max_adjust_abs > 0.0:
                portfolio_bonus = max(
                    -float(portfolio_log_growth_max_adjust_abs),
                    min(
                        float(portfolio_log_growth_max_adjust_abs),
                        float(portfolio_bonus),
                    ),
                )
            rank_score += float(portfolio_bonus)
            target.reasons.append("PORTFOLIO_LOG_GROWTH")
            counterfactual_signal_by_symbol.setdefault(symbol, {}).update(
                {
                    "portfolio_log_growth_base_bps": float(
                        portfolio_log_growth_base_bps
                    ),
                    "portfolio_log_growth_bps": float(marginal_log_growth_bps),
                    "portfolio_corr_penalty": float(corr_penalty),
                    "portfolio_exposure_penalty": float(exposure_penalty),
                    "portfolio_turnover_penalty": float(turnover_penalty),
                    "portfolio_liquidity_impact_penalty": float(
                        liquidity_impact_penalty
                    ),
                    "portfolio_rank_bonus": float(portfolio_bonus),
                }
            )
        if replay_quality_rank_enabled and replay_quality_weight > 0.0:
            replay_metrics = replay_quality_by_symbol.get(symbol)
            replay_bucket_source = "symbol"
            replay_session_key = f"{symbol}:{bandit_active_session}"
            replay_regime_key = f"{symbol}:{bandit_active_session}:{bandit_active_regime}"
            if replay_quality_regime_bucket_enabled:
                replay_regime_metrics = replay_quality_by_symbol_session_regime.get(
                    replay_regime_key
                )
                if isinstance(replay_regime_metrics, Mapping):
                    replay_regime_samples = int(
                        safe_float(replay_regime_metrics.get("sample_count")) or 0.0
                    )
                    if replay_regime_samples >= replay_quality_min_samples:
                        replay_metrics = replay_regime_metrics
                        replay_bucket_source = "symbol_session_regime"
            if replay_bucket_source == "symbol" and replay_quality_session_bucket_enabled:
                replay_session_metrics = replay_quality_by_symbol_session.get(
                    replay_session_key
                )
                if isinstance(replay_session_metrics, Mapping):
                    replay_session_samples = int(
                        safe_float(replay_session_metrics.get("sample_count")) or 0.0
                    )
                    if replay_session_samples >= replay_quality_min_samples:
                        replay_metrics = replay_session_metrics
                        replay_bucket_source = "symbol_session"
            if (
                not isinstance(replay_metrics, Mapping)
                and replay_quality_fallback_to_edge_model_v2
                and edge_model_v2_samples >= edge_model_v2_min_samples
            ):
                replay_metrics = {
                    "sample_count": float(edge_model_v2_samples),
                    "net_edge_bps": float(edge_model_v2_target_bps),
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                }
                replay_bucket_source = "edge_model_v2_fallback"
            if isinstance(replay_metrics, Mapping):
                replay_quality_samples = int(
                    safe_float(replay_metrics.get("sample_count")) or 0.0
                )
                replay_net_edge = safe_float(replay_metrics.get("net_edge_bps"))
                if (
                    replay_quality_samples >= replay_quality_min_samples
                    and replay_net_edge is not None
                ):
                    replay_quality_source = str(
                        f"{replay_quality_context.get('source') or 'unknown'}:{replay_bucket_source}"
                    )
                    replay_quality_net_edge_bps = float(replay_net_edge)
                    replay_quality_target_bps = max(
                        -float(replay_quality_clip_bps),
                        min(
                            float(replay_quality_clip_bps),
                            float(replay_quality_net_edge_bps),
                        ),
                    )
                    replay_quality_bonus = (
                        float(replay_quality_weight)
                        * float(replay_quality_target_bps)
                        * max(max_conf, 0.05)
                        * disagreement
                    )
                    replay_quality_bonus_cap = float(
                        replay_quality_max_rank_uplift_abs
                    )
                    if replay_quality_max_rank_uplift_frac > 0.0:
                        replay_quality_bonus_cap = min(
                            replay_quality_bonus_cap,
                            abs(float(rank_score))
                            * float(replay_quality_max_rank_uplift_frac),
                        )
                    if replay_quality_bonus_cap > 0.0:
                        replay_quality_bonus = max(
                            -float(replay_quality_bonus_cap),
                            min(
                                float(replay_quality_bonus_cap),
                                float(replay_quality_bonus),
                            ),
                        )
                    rank_score += float(replay_quality_bonus)
                    if replay_quality_bonus >= 0.0:
                        target.reasons.append("REPLAY_QUALITY_UPLIFT")
                    else:
                        target.reasons.append("REPLAY_QUALITY_DEWEIGHT")
                    counterfactual_signal_by_symbol.setdefault(symbol, {}).update(
                        {
                            "replay_quality_source": str(replay_quality_source),
                            "replay_quality_samples": int(replay_quality_samples),
                            "replay_quality_net_edge_bps": float(
                                replay_quality_net_edge_bps
                            ),
                            "replay_quality_target_bps": float(
                                replay_quality_target_bps
                            ),
                            "replay_quality_bonus": float(replay_quality_bonus),
                        }
                    )
        if exit_policy_entry_penalty_enabled and exit_policy_entry_penalty_weight > 0.0:
            target_side = "long" if target_position_side >= 0 else "short"
            if candidate_expands_exposure:
                exit_policy_entry_context = exit_policy_pressure_context_func(
                    exit_policy_state,
                    symbol=symbol,
                    regime=bandit_active_regime,
                    side=target_side,
                    now=now,
                    position_age_seconds=0.0,
                    expected_edge_bps=float(clipped_net_edge_total),
                )
            else:
                exit_policy_entry_context = {
                    "active": False,
                    "reason": "not_expanding_exposure",
                    "side": target_side,
                }
            pressure_score = safe_float(exit_policy_entry_context.get("pressure_score"))
            if (
                bool(exit_policy_entry_context.get("active", False))
                and pressure_score is not None
                and float(pressure_score)
                >= float(exit_policy_entry_penalty_min_pressure)
            ):
                exit_policy_entry_penalty_bps = (
                    float(exit_policy_entry_penalty_weight)
                    * float(pressure_score)
                    * max(max_conf, 0.05)
                    * disagreement
                )
                if exit_policy_entry_penalty_max_abs > 0.0:
                    exit_policy_entry_penalty_bps = min(
                        float(exit_policy_entry_penalty_max_abs),
                        float(exit_policy_entry_penalty_bps),
                    )
                rank_score -= float(exit_policy_entry_penalty_bps)
                target.reasons.append("EXIT_POLICY_ENTRY_PENALTY")
                counterfactual_signal_by_symbol.setdefault(symbol, {}).update(
                    {
                        "exit_policy_entry_penalty_bps": float(
                            exit_policy_entry_penalty_bps
                        ),
                        "exit_policy_entry_pressure": float(pressure_score),
                        "exit_policy_entry_context": dict(
                            exit_policy_entry_context
                        ),
                    }
                )
            else:
                counterfactual_signal_by_symbol.setdefault(symbol, {}).update(
                    {
                        "exit_policy_entry_penalty_bps": 0.0,
                        "exit_policy_entry_pressure": (
                            float(pressure_score)
                            if pressure_score is not None
                            else None
                        ),
                        "exit_policy_entry_context": dict(
                            exit_policy_entry_context
                        ),
                    }
                )
        rank_downside_before_cap = 0.0
        rank_downside_cap_applied = False
        rank_downside_cap_limit: float | None = None
        if bool(rank_downside_overlap_cap_enabled) and float(rank_score_baseline) > 0.0:
            rank_downside_before_cap = max(
                float(rank_score_baseline) - float(rank_score),
                0.0,
            )
            cap_candidates: list[float] = []
            if float(rank_downside_overlap_cap_abs) > 0.0:
                cap_candidates.append(float(rank_downside_overlap_cap_abs))
            if float(rank_downside_overlap_cap_frac) > 0.0:
                cap_candidates.append(
                    abs(float(rank_score_baseline))
                    * float(rank_downside_overlap_cap_frac)
                )
            if cap_candidates:
                rank_downside_cap_limit = min(float(value) for value in cap_candidates)
            if (
                rank_downside_cap_limit is not None
                and math.isfinite(float(rank_downside_cap_limit))
                and float(rank_downside_cap_limit) > 0.0
                and float(rank_downside_before_cap) > float(rank_downside_cap_limit)
            ):
                rank_score = float(rank_score_baseline) - float(
                    rank_downside_cap_limit
                )
                rank_downside_cap_applied = True
                target.reasons.append("RANK_DOWNSIDE_OVERLAP_CAP")
        counterfactual_signal_by_symbol.setdefault(symbol, {}).update(
            {
                "rank_score_baseline": float(rank_score_baseline),
                "rank_score_post_adjustments": float(rank_score),
                "rank_downside_before_cap": float(rank_downside_before_cap),
                "rank_downside_overlap_cap_applied": bool(
                    rank_downside_cap_applied
                ),
                "rank_downside_overlap_cap_limit": (
                    float(rank_downside_cap_limit)
                    if rank_downside_cap_limit is not None
                    else None
                ),
            }
        )
        quality_edge_denom = max(float(edge_clip_cap_bps), 1.0)
        quality_realized_denom = max(float(realized_edge_rank_clip_bps), 1.0)
        quality_edge = 0.5 + (
            0.5 * math.tanh(float(clipped_net_edge_total) / float(quality_edge_denom))
        )
        quality_realized = 0.5 + (
            0.5 * math.tanh(
                float(realized_rank_target_bps) / float(quality_realized_denom)
            )
        )
        opportunity_quality = max(
            0.0,
            min(
                1.0,
                (
                    (0.38 * float(quality_edge))
                    + (0.24 * float(quality_realized))
                    + (0.22 * float(max_conf))
                    + (0.16 * float(disagreement))
                ),
            ),
        )
        opportunity_quality_by_symbol[symbol] = float(opportunity_quality)
        counterfactual_signal_by_symbol.setdefault(symbol, {}).update(
            {
                "opportunity_quality_score": float(opportunity_quality),
                "signal_age_seconds": float(signal_age_seconds),
                "time_decay_multiplier": float(time_decay_multiplier),
            }
        )
        candidate_rank[symbol] = rank_score

    opportunity_allowed_symbols: set[str] = set()
    if (
        opportunity_quality_enabled
        and len(opportunity_quality_by_symbol) >= max(2, opportunity_min_symbols)
    ):
        quality_values = list(opportunity_quality_by_symbol.values())
        quality_threshold = percentile_linear_func(
            quality_values,
            float(opportunity_top_quantile),
        )
        if quality_threshold is not None:
            opportunity_allowed_symbols = {
                symbol
                for symbol, score in opportunity_quality_by_symbol.items()
                if float(score) >= float(quality_threshold)
            }
            if len(opportunity_allowed_symbols) < int(opportunity_min_symbols):
                ranked_quality = sorted(
                    opportunity_quality_by_symbol.items(),
                    key=lambda item: (-float(item[1]), item[0]),
                )
                for symbol, _score in ranked_quality[:opportunity_min_symbols]:
                    opportunity_allowed_symbols.add(str(symbol))
            demote_value = float(opportunity_demote_score)
            for symbol in list(candidate_rank.keys()):
                if symbol not in opportunity_allowed_symbols:
                    candidate_rank[symbol] = float(candidate_rank[symbol]) - abs(
                        float(demote_value)
                    )
            opportunity_quality_gate_value.update(
                {
                    "threshold": float(quality_threshold),
                    "allowed_symbols": sorted(opportunity_allowed_symbols),
                    "active": True,
                    "demote_score": float(abs(float(demote_value))),
                }
            )

    return NettingCandidateRankingResult(
        candidate_expected_net_edge=candidate_expected_net_edge,
        candidate_expected_capture=candidate_expected_capture,
        candidate_rank=candidate_rank,
        counterfactual_signal_by_symbol=counterfactual_signal_by_symbol,
        opportunity_quality_by_symbol=opportunity_quality_by_symbol,
        opportunity_allowed_symbols=opportunity_allowed_symbols,
        opportunity_quality_gate=opportunity_quality_gate_value,
    )


__all__ = [
    "NettingCandidateRankingResult",
    "rank_netting_candidates",
]
