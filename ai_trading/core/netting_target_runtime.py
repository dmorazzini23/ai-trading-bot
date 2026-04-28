"""Netting target/runtime orchestration helpers extracted from ``bot_engine.py``."""

from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ai_trading.core.netting_candidate_rank import NettingCandidateRankingResult
from ai_trading.risk.portfolio_limits import PortfolioLimitsResult
def _copy_value(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return value


def _normalize_symbol(symbol: Any) -> str:
    return str(symbol or "").strip().upper()


def _normalize_symbol_prices(raw_prices: Mapping[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for raw_symbol, raw_price in raw_prices.items():
        symbol = _normalize_symbol(raw_symbol)
        if not symbol:
            continue
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(price) or price <= 0.0:
            continue
        normalized[symbol] = float(price)
    return normalized


def _normalize_symbol_returns(
    raw_returns: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    normalized: dict[str, list[float]] = {}
    for raw_symbol, raw_values in raw_returns.items():
        symbol = _normalize_symbol(raw_symbol)
        if not symbol or not isinstance(raw_values, Sequence):
            continue
        cleaned = normalized.setdefault(symbol, [])
        for raw_value in raw_values:
            if not isinstance(raw_value, (int, float)):
                continue
            value = float(raw_value)
            if math.isfinite(value):
                cleaned.append(value)
    return normalized


@dataclass(slots=True)
class PortfolioOptimizerRuntime:
    enabled: bool
    optimizer: Any | None
    openings_only: bool
    context: dict[str, Any]
    market_data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NettingExecutionRankRuntimeContext:
    bool_values: dict[str, bool]
    int_values: dict[str, int]
    float_values: dict[str, float]
    bandit_method: str
    bandit_active_session: str
    bandit_active_regime: str
    promotion_significance_method: str
    bandit_promotion_significance: Any
    counterfactual_promotion_significance: Any
    replay_quality_context: Any
    opportunity_quality_gate: Any
    policy_rollback_disabled_slices: list[Any]
    policy_disabled_gate_roots: list[Any]
    policy_disabled_sleeves: list[Any]
    policy_runtime_toggles_updated_at: Any
    policy_runtime_toggles_source_updated_at: Any
    policy_runtime_toggles: Any

    @classmethod
    def from_values(
        cls,
        context_values: Mapping[str, Any],
    ) -> "NettingExecutionRankRuntimeContext":
        bool_values = {
            key: bool(context_values.get(key))
            for key in _BOOL_CONTEXT_KEYS
        }
        int_values: dict[str, int] = {}
        for key in _INT_CONTEXT_KEYS:
            try:
                int_values[key] = int(context_values.get(key, 0))
            except (TypeError, ValueError):
                int_values[key] = 0

        float_values: dict[str, float] = {}
        for key in _FLOAT_CONTEXT_KEYS:
            try:
                parsed = float(context_values.get(key, 0.0))
            except (TypeError, ValueError):
                parsed = 0.0
            if not math.isfinite(parsed):
                parsed = 0.0
            float_values[key] = float(parsed)

        policy_runtime_payload = context_values.get("policy_runtime_payload", {})
        if isinstance(policy_runtime_payload, Mapping):
            toggles_updated_at = _copy_value(policy_runtime_payload.get("updated_at"))
            toggles_source_updated_at = _copy_value(
                policy_runtime_payload.get("source_updated_at")
            )
        else:
            toggles_updated_at = None
            toggles_source_updated_at = None

        return cls(
            bool_values=bool_values,
            int_values=int_values,
            float_values=float_values,
            bandit_method=str(context_values.get("bandit_method", "") or ""),
            bandit_active_session=str(
                context_values.get("bandit_active_session", "") or ""
            ),
            bandit_active_regime=str(
                context_values.get("bandit_active_regime", "") or ""
            ),
            promotion_significance_method=str(
                context_values.get("promotion_significance_method", "") or ""
            ),
            bandit_promotion_significance=_copy_value(
                context_values.get("bandit_significance_context", {})
            ),
            counterfactual_promotion_significance=_copy_value(
                context_values.get("counterfactual_significance_context", {})
            ),
            replay_quality_context=_copy_value(
                context_values.get("replay_quality_context", {})
            ),
            opportunity_quality_gate=_copy_value(
                context_values.get("opportunity_quality_gate", {})
            ),
            policy_rollback_disabled_slices=_sorted_values(
                context_values.get("policy_rollback_disabled_slices", [])
            ),
            policy_disabled_gate_roots=_sorted_values(
                context_values.get("policy_disabled_gate_roots", [])
            ),
            policy_disabled_sleeves=_sorted_values(
                context_values.get("policy_disabled_sleeves", [])
            ),
            policy_runtime_toggles_updated_at=toggles_updated_at,
            policy_runtime_toggles_source_updated_at=toggles_source_updated_at,
            policy_runtime_toggles=_copy_value(context_values.get("toggles", {})),
        )

    def to_runtime_context(self) -> dict[str, Any]:
        context: dict[str, Any] = {}
        context.update(self.bool_values)
        context.update(self.int_values)
        context.update(self.float_values)
        context["bandit_method"] = self.bandit_method
        context["bandit_active_session"] = self.bandit_active_session
        context["bandit_active_regime"] = self.bandit_active_regime
        context["promotion_significance_method"] = self.promotion_significance_method
        context["bandit_promotion_significance"] = _copy_value(
            self.bandit_promotion_significance
        )
        context["counterfactual_promotion_significance"] = _copy_value(
            self.counterfactual_promotion_significance
        )
        context["replay_quality_context"] = _copy_value(self.replay_quality_context)
        context["opportunity_quality_gate"] = _copy_value(self.opportunity_quality_gate)
        context["policy_rollback_disabled_slices"] = list(
            self.policy_rollback_disabled_slices
        )
        context["policy_disabled_gate_roots"] = list(self.policy_disabled_gate_roots)
        context["policy_disabled_sleeves"] = list(self.policy_disabled_sleeves)
        context["policy_runtime_toggles_updated_at"] = _copy_value(
            self.policy_runtime_toggles_updated_at
        )
        context["policy_runtime_toggles_source_updated_at"] = _copy_value(
            self.policy_runtime_toggles_source_updated_at
        )
        context["policy_runtime_toggles"] = _copy_value(self.policy_runtime_toggles)
        return context


def _sorted_values(raw: Any) -> list[Any]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        if isinstance(raw, set):
            return sorted(raw)
        return []
    try:
        return sorted(raw)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return sorted(str(item) for item in raw)


_BOOL_CONTEXT_KEYS = {
    "expected_edge_clip_enabled",
    "bandit_enabled",
    "bandit_session_bucket_enabled",
    "bandit_regime_bucket_enabled",
    "bandit_shadow_only",
    "bandit_live_promoted",
    "bandit_bucket_promotion_enabled",
    "promotion_significance_enabled",
    "counterfactual_enabled",
    "counterfactual_shadow_only",
    "counterfactual_live_promoted",
    "realized_edge_rank_enabled",
    "edge_model_v2_enabled",
    "replay_quality_rank_enabled",
    "replay_quality_session_bucket_enabled",
    "replay_quality_regime_bucket_enabled",
    "replay_quality_auto_disable_if_stale",
    "replay_quality_fallback_to_edge_model_v2",
    "expected_capture_rank_enabled",
    "expected_capture_learned_fill_model_enabled",
    "expected_capture_cost_model_enabled",
    "execution_learning_rank_enabled",
    "rejection_concentration_rank_enabled",
    "rank_downside_overlap_cap_enabled",
    "alpha_time_decay_enabled",
    "alpha_time_stop_enabled",
    "portfolio_log_growth_rank_enabled",
    "geometric_tiebreak_enabled",
    "edge_realism_rank_calibration_enabled",
    "edge_realism_rank_calibration_session_enabled",
    "edge_realism_apply_to_approval_enabled",
}

_INT_CONTEXT_KEYS = {
    "bandit_bucket_promote_min_samples",
    "bandit_promote_min_samples",
    "bandit_global_samples",
    "counterfactual_min_samples",
    "counterfactual_global_events",
    "realized_edge_rank_min_samples",
    "edge_model_v2_min_samples",
    "replay_quality_min_samples",
    "replay_quality_symbol_count",
    "replay_quality_symbol_session_count",
    "replay_quality_symbol_session_regime_count",
    "expected_capture_learned_fill_min_samples",
    "expected_capture_cost_model_min_samples",
    "execution_learning_rank_min_samples",
    "rejection_concentration_window_records",
    "edge_realism_rank_calibration_min_samples",
    "edge_realism_rank_calibration_prior_samples",
}

_FLOAT_CONTEXT_KEYS = {
    "expected_edge_clip_cap_bps",
    "expected_edge_clip_median_abs_bps",
    "bandit_bucket_promote_min_mean_reward_bps",
    "bandit_promote_min_mean_reward_bps",
    "bandit_global_mean_reward_bps",
    "bandit_max_rank_uplift_abs",
    "bandit_max_rank_uplift_frac",
    "counterfactual_weight",
    "counterfactual_clip_bps",
    "counterfactual_global_dr_mean_bps",
    "realized_edge_rank_weight",
    "realized_edge_rank_uncertainty_z",
    "realized_edge_rank_clip_bps",
    "edge_model_v2_weight",
    "edge_model_v2_uncertainty_z",
    "edge_model_v2_regime_weight",
    "edge_model_v2_session_weight",
    "edge_model_v2_global_weight",
    "edge_model_v2_cost_weight",
    "edge_model_v2_clip_bps",
    "replay_quality_weight",
    "replay_quality_clip_bps",
    "replay_quality_max_age_hours",
    "expected_capture_rank_weight",
    "expected_capture_floor_bps",
    "expected_capture_floor_bps_configured",
    "execution_learning_rank_slippage_floor_bps",
    "execution_learning_rank_max_penalty_bps",
    "rejection_concentration_lookback_hours",
    "rejection_concentration_max_penalty_bps",
    "expected_capture_fill_prob_floor",
    "expected_capture_fill_prob_cap",
    "expected_capture_spread_penalty_bps",
    "expected_capture_participation_penalty_bps",
    "expected_capture_age_half_life_sec",
    "rank_downside_overlap_cap_frac",
    "rank_downside_overlap_cap_abs",
    "alpha_time_decay_half_life_sec",
    "alpha_time_decay_floor",
    "alpha_stale_signal_sec",
    "alpha_time_stop_sec",
    "alpha_time_stop_max_expected_edge_bps",
    "portfolio_log_growth_rank_weight",
    "portfolio_log_growth_turnover_penalty_bps",
    "portfolio_log_growth_liquidity_penalty_bps",
    "portfolio_log_growth_max_participation",
    "geometric_tiebreak_weight",
    "edge_realism_rank_calibration_ratio_floor",
    "edge_realism_rank_calibration_ratio_cap",
}


def _build_execution_rank_context(context_values: Mapping[str, Any]) -> dict[str, Any]:
    return NettingExecutionRankRuntimeContext.from_values(
        context_values
    ).to_runtime_context()


def store_candidate_ranking_runtime_state(
    *,
    runtime: Any,
    ranking_result: NettingCandidateRankingResult,
    edge_realism_rank_factor_by_symbol: Mapping[str, Any],
    context_values: Mapping[str, Any],
    store_execution_candidate_ranking_runtime_state_func: Callable[..., None] | None = None,
) -> None:
    """Persist ranking outputs and assembled context onto runtime."""

    if store_execution_candidate_ranking_runtime_state_func is None:
        from ai_trading.core.execution_runtime_metadata import (
            store_execution_candidate_ranking_runtime_state,
        )

        store_execution_candidate_ranking_runtime_state_func = (
            store_execution_candidate_ranking_runtime_state
        )

    rank_context = _build_execution_rank_context(context_values)
    store_execution_candidate_ranking_runtime_state_func(
        runtime,
        opportunity_quality_by_symbol=ranking_result.opportunity_quality_by_symbol,
        opportunity_allowed_symbols=ranking_result.opportunity_allowed_symbols,
        opportunity_quality_gate=ranking_result.opportunity_quality_gate,
        candidate_rank=ranking_result.candidate_rank,
        candidate_expected_net_edge=ranking_result.candidate_expected_net_edge,
        candidate_expected_capture=ranking_result.candidate_expected_capture,
        edge_realism_rank_factor_by_symbol=edge_realism_rank_factor_by_symbol,
        counterfactual_signal_by_symbol=ranking_result.counterfactual_signal_by_symbol,
        rank_context=rank_context,
    )


def prepare_portfolio_optimizer_runtime(
    *,
    latest_price: Mapping[str, Any],
    symbol_returns: Mapping[str, Sequence[float]],
    rollout_toggle_func: Callable[[str, bool], bool],
    get_env_func: Callable[..., Any],
    build_symbol_return_correlation_matrix_func: Callable[
        [Mapping[str, Sequence[float]]],
        dict[str, dict[str, float]],
    ],
    logger: Any,
    create_portfolio_optimizer_func: Callable[[dict[str, Any]], Any] | None = None,
) -> PortfolioOptimizerRuntime:
    """Build portfolio-optimizer runtime state and normalized market data."""

    enabled = bool(
        rollout_toggle_func(
            "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_ENABLED",
            False,
        )
    )
    openings_only = bool(
        get_env_func(
            "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_OPENINGS_ONLY",
            True,
            cast=bool,
        )
    )
    optimizer: Any | None = None
    context: dict[str, Any] = {
        "enabled": bool(enabled),
        "openings_only": bool(openings_only),
    }
    init_fail_open = bool(
        get_env_func(
            "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_INIT_FAIL_OPEN",
            False,
            cast=bool,
        )
    )
    context["init_fail_open"] = bool(init_fail_open)

    if create_portfolio_optimizer_func is None:
        from ai_trading.portfolio import create_portfolio_optimizer

        create_portfolio_optimizer_func = create_portfolio_optimizer

    if enabled:
        try:
            optimizer = create_portfolio_optimizer_func(
                {
                    "improvement_threshold": float(
                        get_env_func(
                            "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_IMPROVEMENT_THRESHOLD",
                            0.02,
                            cast=float,
                        )
                    ),
                    "max_correlation_penalty": float(
                        get_env_func(
                            "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_MAX_CORRELATION_PENALTY",
                            0.15,
                            cast=float,
                        )
                    ),
                    "rebalance_drift_threshold": float(
                        get_env_func(
                            "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_REBALANCE_DRIFT_THRESHOLD",
                            0.05,
                            cast=float,
                        )
                    ),
                    "turnover_penalty": float(
                        get_env_func(
                            "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_TURNOVER_PENALTY",
                            0.01,
                            cast=float,
                        )
                    ),
                    "mode": str(
                        get_env_func(
                            "AI_TRADING_EXEC_PORTFOLIO_OPTIMIZER_MODE",
                            "execution_live",
                            cast=str,
                        )
                        or "execution_live"
                    ).strip(),
                }
            )
            context["active"] = True
        except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
            optimizer = None
            enabled = bool(not init_fail_open)
            context["active"] = False
            context["init_failed"] = True
            context["fail_open_applied"] = bool(init_fail_open)
            context["error_type"] = exc.__class__.__name__
            context["error"] = str(exc)
            logger.warning(
                "PORTFOLIO_OPTIMIZER_INIT_FAILED",
                extra=dict(context),
            )

    normalized_prices = _normalize_symbol_prices(latest_price)
    normalized_returns = _normalize_symbol_returns(symbol_returns)
    market_data: dict[str, Any] = {
        "prices": normalized_prices,
        "returns": normalized_returns,
    }
    market_data["correlations"] = build_symbol_return_correlation_matrix_func(
        normalized_returns
    )
    return PortfolioOptimizerRuntime(
        enabled=bool(enabled),
        optimizer=optimizer,
        openings_only=bool(openings_only),
        context=context,
        market_data=market_data,
    )


def apply_target_construction_controls(
    *,
    targets: dict[str, Any],
    cfg: Any,
    normalized_symbol_returns: Mapping[str, Sequence[float]],
    apply_global_caps_func: Callable[..., Any],
    apply_portfolio_limits_func: Callable[..., PortfolioLimitsResult],
    get_env_func: Callable[..., Any],
    logger: Any,
) -> None:
    """Apply global caps and portfolio-level limits to target construction."""

    apply_global_caps_func(
        targets,
        float(getattr(cfg, "global_max_symbol_dollars", 0.0)),
        float(getattr(cfg, "global_max_gross_dollars", 0.0)),
        float(getattr(cfg, "global_max_net_dollars", 0.0)),
    )
    if not bool(get_env_func("AI_TRADING_PORTFOLIO_LIMITS_ENABLED", False, cast=bool)):
        return

    limits_symbol_returns = _normalize_symbol_returns(normalized_symbol_returns)
    limits = apply_portfolio_limits_func(
        targets={
            str(symbol).strip().upper(): float(getattr(target, "target_dollars", 0.0))
            for symbol, target in targets.items()
            if str(symbol).strip()
        },
        symbol_returns=limits_symbol_returns,
        vol_targeting_enabled=bool(
            get_env_func("AI_TRADING_VOL_TARGETING_ENABLED", True, cast=bool)
        ),
        target_annual_vol=float(
            get_env_func("AI_TRADING_TARGET_ANNUAL_VOL", 0.18, cast=float)
        ),
        vol_lookback_days=int(
            get_env_func("AI_TRADING_VOL_LOOKBACK_DAYS", 20, cast=int)
        ),
        vol_min_scale=float(get_env_func("AI_TRADING_VOL_MIN_SCALE", 0.25, cast=float)),
        vol_max_scale=float(get_env_func("AI_TRADING_VOL_MAX_SCALE", 1.25, cast=float)),
        concentration_cap_enabled=bool(
            get_env_func("AI_TRADING_CONCENTRATION_CAP_ENABLED", True, cast=bool)
        ),
        max_symbol_weight=float(
            get_env_func("AI_TRADING_MAX_SYMBOL_WEIGHT", 0.12, cast=float)
        ),
        max_cluster_weight=float(
            get_env_func("AI_TRADING_MAX_CLUSTER_WEIGHT", 0.25, cast=float)
        ),
        corr_cap_enabled=bool(get_env_func("AI_TRADING_CORR_CAP_ENABLED", True, cast=bool)),
        corr_lookback_days=int(
            get_env_func("AI_TRADING_CORR_LOOKBACK_DAYS", 30, cast=int)
        ),
        corr_threshold=float(get_env_func("AI_TRADING_CORR_THRESHOLD", 0.80, cast=float)),
        corr_group_gross_cap=float(
            get_env_func("AI_TRADING_CORR_GROUP_GROSS_CAP", 0.35, cast=float)
        ),
    )
    for raw_symbol, raw_dollars in limits.scaled_targets.items():
        symbol = _normalize_symbol(raw_symbol)
        if symbol not in targets:
            continue
        try:
            dollars = float(raw_dollars)
        except (TypeError, ValueError):
            logger.warning(
                "PORTFOLIO_LIMIT_INVALID_TARGET_SKIP",
                extra={"symbol": symbol, "raw_target": raw_dollars},
            )
            continue
        if not math.isfinite(dollars):
            logger.warning(
                "PORTFOLIO_LIMIT_INVALID_TARGET_SKIP",
                extra={"symbol": symbol, "raw_target": raw_dollars},
            )
            continue
        targets[symbol].target_dollars = dollars
        for reason in limits.reasons:
            if reason not in targets[symbol].reasons:
                targets[symbol].reasons.append(reason)


__all__ = [
    "NettingExecutionRankRuntimeContext",
    "PortfolioOptimizerRuntime",
    "apply_target_construction_controls",
    "prepare_portfolio_optimizer_runtime",
    "store_candidate_ranking_runtime_state",
]
