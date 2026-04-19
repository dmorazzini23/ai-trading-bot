"""Learned ranking state helpers for the live netting cycle."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Callable, Mapping, Sequence


@dataclass(slots=True)
class NettingLearningState:
    bandit_rewards_by_symbol: dict[str, list[float]]
    bandit_rewards_by_symbol_session: dict[str, list[float]]
    bandit_rewards_by_symbol_session_regime: dict[str, list[float]]
    realized_edge_by_symbol: dict[str, list[float]]
    realized_edge_by_symbol_session: dict[str, list[float]]
    realized_edge_by_symbol_session_regime: dict[str, list[float]]
    edge_realism_ratio_by_symbol: dict[str, list[float]]
    edge_realism_ratio_by_symbol_session: dict[str, list[float]]
    edge_realism_rank_factor_by_symbol: dict[str, float]
    counterfactual_signal_by_symbol: dict[str, dict[str, Any]]
    opportunity_quality_by_symbol: dict[str, float]
    learned_fill_trials_by_bucket: dict[str, float]
    learned_fill_success_by_bucket: dict[str, float]
    learned_exec_cost_stats_by_bucket: dict[str, dict[str, float]]
    expected_capture_observed_values: list[float]
    opportunity_quality_gate: dict[str, Any]
    bandit_active_session: str
    bandit_active_regime: str
    counterfactual_global: dict[str, Any]
    counterfactual_buckets: dict[str, Any]
    execution_learning_state: dict[str, Any]
    rejection_concentration_by_symbol: dict[str, Any]
    bandit_global_samples: int
    bandit_global_mean_reward_bps: float
    bandit_significance_context: dict[str, Any]
    bandit_significance_pass: bool
    bandit_live_promoted: bool
    bandit_bucket_significance_cache: dict[str, dict[str, Any]]
    counterfactual_global_events: int
    counterfactual_global_dr_mean_bps: float
    counterfactual_significance_context: dict[str, Any]
    counterfactual_significance_pass: bool
    counterfactual_live_promoted: bool
    portfolio_rank_correlation: dict[str, dict[str, float]]
    portfolio_current_gross_for_rank: float
    held_notional_weights: dict[str, float]
    expected_capture_floor_bps_effective: float


def expected_capture_bucket_keys(
    *,
    symbol: str,
    session_token: str,
    regime_token: str,
    liquidity_role: str,
    venue_session: str,
) -> tuple[str, ...]:
    return (
        f"{symbol}:{session_token}:{regime_token}:{liquidity_role}:{venue_session}",
        f"{symbol}:{session_token}:{regime_token}:{liquidity_role}",
        f"{symbol}:{session_token}:{regime_token}",
        f"{symbol}:{session_token}",
        f"{symbol}",
        "__global__",
    )


def lookup_learned_fill_probability(
    *,
    learned_fill_trials_by_bucket: Mapping[str, float],
    learned_fill_success_by_bucket: Mapping[str, float],
    expected_capture_learned_fill_min_samples: int,
    expected_capture_learned_fill_prior_alpha: float,
    expected_capture_learned_fill_prior_beta: float,
    symbol: str,
    session_token: str,
    regime_token: str,
    liquidity_role: str,
    venue_session: str,
) -> tuple[float | None, int, str | None]:
    for key in expected_capture_bucket_keys(
        symbol=symbol,
        session_token=session_token,
        regime_token=regime_token,
        liquidity_role=liquidity_role,
        venue_session=venue_session,
    ):
        trials = float(learned_fill_trials_by_bucket.get(str(key), 0.0) or 0.0)
        if trials < float(expected_capture_learned_fill_min_samples):
            continue
        successes = float(learned_fill_success_by_bucket.get(str(key), 0.0) or 0.0)
        posterior_prob = (
            float(successes) + float(expected_capture_learned_fill_prior_alpha)
        ) / (
            float(trials)
            + float(expected_capture_learned_fill_prior_alpha)
            + float(expected_capture_learned_fill_prior_beta)
        )
        return (
            max(0.0, min(1.0, float(posterior_prob))),
            int(round(trials)),
            str(key),
        )
    return (None, 0, None)


def lookup_learned_execution_cost_components(
    *,
    learned_exec_cost_stats_by_bucket: Mapping[str, Mapping[str, float]],
    safe_float: Callable[[Any], float | None],
    expected_capture_cost_model_min_samples: int,
    symbol: str,
    session_token: str,
    regime_token: str,
    liquidity_role: str,
    venue_session: str,
) -> tuple[float | None, float | None, float | None, int, str | None]:
    for key in expected_capture_bucket_keys(
        symbol=symbol,
        session_token=session_token,
        regime_token=regime_token,
        liquidity_role=liquidity_role,
        venue_session=venue_session,
    ):
        stats_item = learned_exec_cost_stats_by_bucket.get(str(key))
        if not isinstance(stats_item, Mapping):
            continue
        samples = float(safe_float(stats_item.get("samples")) or 0.0)
        if samples < float(expected_capture_cost_model_min_samples):
            continue
        spread_mean = float((safe_float(stats_item.get("spread_sum_bps")) or 0.0) / float(samples))
        impact_mean = float((safe_float(stats_item.get("impact_sum_bps")) or 0.0) / float(samples))
        latency_mean = float((safe_float(stats_item.get("latency_sum_bps")) or 0.0) / float(samples))
        return (
            max(0.0, spread_mean),
            max(0.0, impact_mean),
            max(0.0, latency_mean),
            int(round(samples)),
            str(key),
        )
    return (None, None, None, 0, None)


def build_netting_learning_state(
    *,
    now: datetime,
    targets: Mapping[str, Any],
    positions: Mapping[str, float],
    latest_price: Mapping[str, float],
    symbol_returns: Mapping[str, list[float]],
    get_env: Callable[..., Any],
    safe_float: Callable[[Any], float | None],
    session_bucket_from_ts_func: Callable[[datetime | None], str],
    get_regime_signal_profile_func: Callable[[], str | None],
    load_counterfactual_learning_state_func: Callable[[], dict[str, Any]],
    load_execution_learning_state_func: Callable[[], dict[str, Any]],
    load_recent_rejection_concentration_by_symbol_func: Callable[..., dict[str, Any]],
    resolved_tca_path_func: Callable[[], Any],
    read_jsonl_records_func: Callable[..., list[dict[str, Any]]],
    infer_tca_liquidity_role_func: Callable[[Mapping[str, Any]], str],
    extract_tca_fill_success_ratio_func: Callable[[Mapping[str, Any]], float | None],
    mean_std_func: Callable[[Sequence[float]], tuple[float, float, int]],
    sequential_significance_gate_func: Callable[..., dict[str, Any]],
    build_symbol_return_correlation_matrix_func: Callable[[Mapping[str, list[float]]], dict[str, dict[str, float]]],
    percentile_linear_func: Callable[[Sequence[float], float], float | None],
    parse_iso_timestamp_func: Callable[[Any], datetime | None],
    bandit_enabled: bool,
    bandit_method: str,
    bandit_min_samples: int,
    bandit_window_trades: int,
    bandit_session_bucket_enabled: bool,
    bandit_regime_bucket_enabled: bool,
    bandit_shadow_only: bool,
    bandit_auto_promote: bool,
    bandit_promote_min_samples: int,
    bandit_promote_min_mean_reward_bps: float,
    bandit_bucket_promotion_enabled: bool,
    bandit_bucket_promote_min_samples: int,
    bandit_bucket_promote_min_mean_reward_bps: float,
    edge_realism_rank_calibration_enabled: bool,
    edge_realism_rank_calibration_session_enabled: bool,
    edge_realism_rank_calibration_min_samples: int,
    edge_realism_rank_calibration_window_trades: int,
    edge_realism_rank_calibration_prior_samples: int,
    edge_realism_rank_calibration_ratio_floor: float,
    edge_realism_rank_calibration_ratio_cap: float,
    counterfactual_enabled: bool,
    counterfactual_shadow_only: bool,
    counterfactual_auto_promote: bool,
    counterfactual_min_samples: int,
    counterfactual_promote_min_events: int,
    counterfactual_promote_min_dr_mean_bps: float,
    realized_edge_rank_enabled: bool,
    expected_capture_rank_enabled: bool,
    expected_capture_learned_fill_model_enabled: bool,
    expected_capture_learned_fill_min_samples: int,
    expected_capture_learned_fill_prior_alpha: float,
    expected_capture_learned_fill_prior_beta: float,
    expected_capture_venue_default: str,
    expected_capture_cost_model_enabled: bool,
    expected_capture_cost_model_min_samples: int,
    expected_capture_latency_bps_per_sec: float,
    expected_capture_floor_bps: float,
    expected_capture_floor_adaptive_enabled: bool,
    expected_capture_floor_adaptive_quantile: float,
    execution_learning_rank_enabled: bool,
    rejection_concentration_rank_enabled: bool,
    rejection_concentration_window_records: int,
    rejection_concentration_lookback_hours: float,
    promotion_significance_enabled: bool,
    promotion_significance_method: str,
    promotion_significance_posterior_prob_min: float,
    promotion_significance_sprt_alpha: float,
    promotion_significance_sprt_beta: float,
    promotion_significance_sprt_effect_bps: float,
    opportunity_quality_enabled: bool,
    opportunity_top_quantile: float,
    opportunity_min_symbols: int,
    opportunity_openings_only: bool,
    portfolio_log_growth_rank_enabled: bool,
) -> NettingLearningState:
    bandit_rewards_by_symbol: dict[str, list[float]] = {}
    bandit_rewards_by_symbol_session: dict[str, list[float]] = {}
    bandit_rewards_by_symbol_session_regime: dict[str, list[float]] = {}
    realized_edge_by_symbol: dict[str, list[float]] = {}
    realized_edge_by_symbol_session: dict[str, list[float]] = {}
    realized_edge_by_symbol_session_regime: dict[str, list[float]] = {}
    edge_realism_ratio_by_symbol: dict[str, list[float]] = {}
    edge_realism_ratio_by_symbol_session: dict[str, list[float]] = {}
    edge_realism_rank_factor_by_symbol: dict[str, float] = {}
    counterfactual_signal_by_symbol: dict[str, dict[str, Any]] = {}
    opportunity_quality_by_symbol: dict[str, float] = {}
    learned_fill_trials_by_bucket: dict[str, float] = {}
    learned_fill_success_by_bucket: dict[str, float] = {}
    learned_exec_cost_stats_by_bucket: dict[str, dict[str, float]] = {}
    expected_capture_observed_values: list[float] = []
    opportunity_quality_gate: dict[str, Any] = {
        "enabled": bool(opportunity_quality_enabled),
        "top_quantile": float(opportunity_top_quantile),
        "min_symbols": int(opportunity_min_symbols),
        "openings_only": bool(opportunity_openings_only),
        "threshold": None,
        "allowed_symbols": [],
    }
    bandit_active_session = session_bucket_from_ts_func(now)
    bandit_active_regime = str(get_regime_signal_profile_func() or "").strip().lower() or "unknown"
    counterfactual_state = (
        load_counterfactual_learning_state_func()
        if counterfactual_enabled
        else {"global": {}, "buckets": {}}
    )
    counterfactual_global_raw = counterfactual_state.get("global")
    counterfactual_global = (
        dict(counterfactual_global_raw)
        if isinstance(counterfactual_global_raw, Mapping)
        else {}
    )
    counterfactual_buckets_raw = counterfactual_state.get("buckets")
    counterfactual_buckets = (
        dict(counterfactual_buckets_raw)
        if isinstance(counterfactual_buckets_raw, Mapping)
        else {}
    )
    execution_learning_state = (
        load_execution_learning_state_func()
        if execution_learning_rank_enabled
        else {"global": {}, "buckets": {}, "symbol_buckets": {}}
    )
    rejection_concentration_by_symbol = (
        load_recent_rejection_concentration_by_symbol_func(
            now=now,
            max_records=int(rejection_concentration_window_records),
            lookback_hours=float(rejection_concentration_lookback_hours),
        )
        if rejection_concentration_rank_enabled
        else {}
    )

    if (
        bandit_enabled
        or edge_realism_rank_calibration_enabled
        or realized_edge_rank_enabled
        or expected_capture_rank_enabled
    ) and targets:
        bandit_rows = read_jsonl_records_func(
            str(resolved_tca_path_func()),
            max_records=max(
                500,
                int(
                    get_env(
                        "AI_TRADING_EXEC_BANDIT_MAX_RECORDS",
                        bandit_window_trades * max(len(targets), 1) * 8,
                        cast=int,
                    )
                ),
            ),
        )
        for row in bandit_rows:
            symbol = str(row.get("symbol", "")).strip().upper()
            if not symbol or symbol not in targets:
                continue
            status = str(row.get("status", "")).strip().lower()
            session_token = str(row.get("session_regime", "")).strip().lower()
            if session_token not in {"opening", "midday", "closing", "offhours"}:
                ts = parse_iso_timestamp_func(row.get("ts"))
                session_token = session_bucket_from_ts_func(ts)
            bucket_key = f"{symbol}:{session_token}"
            regime_token = str(
                row.get("regime_profile")
                or row.get("regime")
                or row.get("session_profile")
                or ""
            ).strip().lower() or "unknown"
            bucket_regime_key = f"{symbol}:{session_token}:{regime_token}"
            liquidity_role = infer_tca_liquidity_role_func(row)
            venue_session_token = str(row.get("venue_session") or "").strip()
            if not venue_session_token:
                venue_token = str(
                    row.get("venue")
                    or row.get("provider")
                    or expected_capture_venue_default
                ).strip().upper() or expected_capture_venue_default
                venue_session_token = f"{venue_token}:{session_token}"

            if expected_capture_rank_enabled and expected_capture_learned_fill_model_enabled:
                fill_success_ratio = extract_tca_fill_success_ratio_func(row)
                if fill_success_ratio is not None:
                    fill_bucket_keys = expected_capture_bucket_keys(
                        symbol=symbol,
                        session_token=session_token,
                        regime_token=regime_token,
                        liquidity_role=liquidity_role,
                        venue_session=venue_session_token,
                    )
                    for key in fill_bucket_keys:
                        learned_fill_trials_by_bucket[str(key)] = float(
                            learned_fill_trials_by_bucket.get(str(key), 0.0)
                        ) + 1.0
                        learned_fill_success_by_bucket[str(key)] = float(
                            learned_fill_success_by_bucket.get(str(key), 0.0)
                        ) + float(max(0.0, min(1.0, fill_success_ratio)))

                spread_cost_bps = max(0.0, float(safe_float(row.get("spread_paid_bps")) or 0.0))
                total_cost_bps = max(0.0, float(safe_float(row.get("is_bps")) or spread_cost_bps))
                impact_cost_bps = max(0.0, float(total_cost_bps) - float(spread_cost_bps))
                latency_ms = max(0.0, float(safe_float(row.get("fill_latency_ms")) or 0.0))
                observed_latency_drift = max(
                    0.0,
                    abs(float(safe_float(row.get("execution_drift_bps")) or 0.0)),
                )
                latency_drift_cost_bps = max(
                    float(observed_latency_drift),
                    (float(latency_ms) / 1000.0) * float(expected_capture_latency_bps_per_sec),
                )
                expected_edge_for_capture = safe_float(row.get("expected_net_edge_bps"))
                if expected_edge_for_capture is None:
                    expected_edge_for_capture = safe_float(row.get("edge_bps"))
                if expected_edge_for_capture is not None and fill_success_ratio is not None:
                    observed_capture = (
                        float(expected_edge_for_capture) * float(fill_success_ratio)
                    ) - (
                        float(spread_cost_bps)
                        + float(impact_cost_bps)
                        + float(latency_drift_cost_bps)
                    )
                    if math.isfinite(float(observed_capture)):
                        expected_capture_observed_values.append(float(observed_capture))
                for key in expected_capture_bucket_keys(
                    symbol=symbol,
                    session_token=session_token,
                    regime_token=regime_token,
                    liquidity_role=liquidity_role,
                    venue_session=venue_session_token,
                ):
                    stats_item = learned_exec_cost_stats_by_bucket.setdefault(
                        str(key),
                        {
                            "samples": 0.0,
                            "spread_sum_bps": 0.0,
                            "impact_sum_bps": 0.0,
                            "latency_sum_bps": 0.0,
                        },
                    )
                    stats_item["samples"] = float(stats_item.get("samples", 0.0) or 0.0) + 1.0
                    stats_item["spread_sum_bps"] = float(
                        stats_item.get("spread_sum_bps", 0.0) or 0.0
                    ) + float(spread_cost_bps)
                    stats_item["impact_sum_bps"] = float(
                        stats_item.get("impact_sum_bps", 0.0) or 0.0
                    ) + float(impact_cost_bps)
                    stats_item["latency_sum_bps"] = float(
                        stats_item.get("latency_sum_bps", 0.0) or 0.0
                    ) + float(latency_drift_cost_bps)

            if status in {"rejected", "canceled", "cancelled"}:
                continue

            realized_edge_bps: float | None = None
            for key in ("realized_net_edge_bps", "net_edge_bps", "realized_edge_bps"):
                try:
                    candidate = float(row.get(key))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(candidate):
                    realized_edge_bps = float(candidate)
                    break
            if realized_edge_bps is not None and realized_edge_rank_enabled:
                bounded_realized_edge = max(-200.0, min(float(realized_edge_bps), 200.0))
                realized_edge_by_symbol.setdefault(symbol, []).append(float(bounded_realized_edge))
                realized_edge_by_symbol_session.setdefault(bucket_key, []).append(
                    float(bounded_realized_edge)
                )
                realized_edge_by_symbol_session_regime.setdefault(bucket_regime_key, []).append(
                    float(bounded_realized_edge)
                )

            reward_bps: float | None = None
            if realized_edge_bps is not None:
                reward_bps = float(realized_edge_bps)
            if reward_bps is None:
                expected_net_edge_for_reward = safe_float(row.get("expected_net_edge_bps"))
                if expected_net_edge_for_reward is None:
                    expected_net_edge_for_reward = safe_float(row.get("expected_edge_bps"))
                spread_paid_bps = abs(float(safe_float(row.get("spread_paid_bps")) or 0.0))
                exec_drift_bps = abs(float(safe_float(row.get("execution_drift_bps")) or 0.0))
                is_bps = safe_float(row.get("is_bps"))
                if is_bps is not None and math.isfinite(float(is_bps)):
                    if (
                        expected_net_edge_for_reward is not None
                        and math.isfinite(float(expected_net_edge_for_reward))
                    ):
                        reward_bps = float(expected_net_edge_for_reward) - (
                            abs(float(is_bps))
                            + float(spread_paid_bps)
                            + float(exec_drift_bps)
                        )
                    else:
                        reward_bps = float(-is_bps)

            if edge_realism_rank_calibration_enabled and reward_bps is not None:
                expected_bps: float | None = None
                for key in ("expected_net_edge_bps", "expected_edge_bps", "edge_bps", "alpha_edge_bps"):
                    try:
                        candidate = float(row.get(key))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(candidate) and candidate > 0.0:
                        expected_bps = float(candidate)
                        break
                if expected_bps is not None and float(expected_bps) > 0.0:
                    ratio = float(reward_bps) / max(float(expected_bps), 0.1)
                    ratio = max(0.0, min(ratio, 5.0))
                    edge_realism_ratio_by_symbol.setdefault(symbol, []).append(float(ratio))
                    edge_realism_ratio_by_symbol_session.setdefault(bucket_key, []).append(
                        float(ratio)
                    )

            if reward_bps is None:
                continue
            reward_bps = max(-200.0, min(float(reward_bps), 200.0))
            if bandit_enabled:
                bandit_rewards_by_symbol.setdefault(symbol, []).append(float(reward_bps))
                bandit_rewards_by_symbol_session.setdefault(bucket_key, []).append(float(reward_bps))
                bandit_rewards_by_symbol_session_regime.setdefault(
                    bucket_regime_key,
                    [],
                ).append(float(reward_bps))

        for symbol, values in list(bandit_rewards_by_symbol.items()):
            if len(values) > bandit_window_trades:
                bandit_rewards_by_symbol[symbol] = values[-bandit_window_trades:]
        for key, values in list(bandit_rewards_by_symbol_session.items()):
            if len(values) > bandit_window_trades:
                bandit_rewards_by_symbol_session[key] = values[-bandit_window_trades:]
        for key, values in list(bandit_rewards_by_symbol_session_regime.items()):
            if len(values) > bandit_window_trades:
                bandit_rewards_by_symbol_session_regime[key] = values[-bandit_window_trades:]
        for symbol, values in list(realized_edge_by_symbol.items()):
            if len(values) > bandit_window_trades:
                realized_edge_by_symbol[symbol] = values[-bandit_window_trades:]
        for key, values in list(realized_edge_by_symbol_session.items()):
            if len(values) > bandit_window_trades:
                realized_edge_by_symbol_session[key] = values[-bandit_window_trades:]
        for key, values in list(realized_edge_by_symbol_session_regime.items()):
            if len(values) > bandit_window_trades:
                realized_edge_by_symbol_session_regime[key] = values[-bandit_window_trades:]
        for symbol, values in list(edge_realism_ratio_by_symbol.items()):
            if len(values) > edge_realism_rank_calibration_window_trades:
                edge_realism_ratio_by_symbol[symbol] = values[
                    -edge_realism_rank_calibration_window_trades:
                ]
        for key, values in list(edge_realism_ratio_by_symbol_session.items()):
            if len(values) > edge_realism_rank_calibration_window_trades:
                edge_realism_ratio_by_symbol_session[key] = values[
                    -edge_realism_rank_calibration_window_trades:
                ]

    expected_capture_floor_bps_effective = float(expected_capture_floor_bps)
    if expected_capture_floor_adaptive_enabled and expected_capture_observed_values:
        adaptive_capture_floor = percentile_linear_func(
            expected_capture_observed_values,
            expected_capture_floor_adaptive_quantile,
        )
        if adaptive_capture_floor is not None:
            expected_capture_floor_bps_effective = min(
                float(expected_capture_floor_bps_effective),
                float(adaptive_capture_floor),
            )

    if edge_realism_rank_calibration_enabled and targets:
        all_ratios = [
            float(value)
            for values in edge_realism_ratio_by_symbol.values()
            for value in values
            if math.isfinite(float(value))
        ]
        global_ratio = (
            float(sum(all_ratios) / max(len(all_ratios), 1))
            if all_ratios
            else None
        )
        for symbol in targets.keys():
            selected_ratios = edge_realism_ratio_by_symbol.get(symbol, [])
            if edge_realism_rank_calibration_session_enabled:
                session_key = f"{symbol}:{bandit_active_session}"
                session_ratios = edge_realism_ratio_by_symbol_session.get(session_key, [])
                if len(session_ratios) >= edge_realism_rank_calibration_min_samples:
                    selected_ratios = session_ratios
            symbol_ratio: float | None = None
            if len(selected_ratios) >= edge_realism_rank_calibration_min_samples:
                symbol_ratio = float(
                    sum(float(value) for value in selected_ratios) / max(len(selected_ratios), 1)
                )
            blended_ratio: float | None = symbol_ratio
            if blended_ratio is None:
                blended_ratio = global_ratio
            elif global_ratio is not None and edge_realism_rank_calibration_prior_samples > 0:
                numerator = (
                    float(blended_ratio) * float(len(selected_ratios))
                    + float(global_ratio) * float(edge_realism_rank_calibration_prior_samples)
                )
                denom = float(len(selected_ratios) + edge_realism_rank_calibration_prior_samples)
                if denom > 0.0:
                    blended_ratio = float(numerator / denom)
            if blended_ratio is None:
                edge_realism_rank_factor_by_symbol[symbol] = 1.0
                continue
            edge_realism_rank_factor_by_symbol[symbol] = float(
                max(
                    edge_realism_rank_calibration_ratio_floor,
                    min(edge_realism_rank_calibration_ratio_cap, float(blended_ratio)),
                )
            )

    bandit_global_samples = int(sum(len(values) for values in bandit_rewards_by_symbol.values()))
    bandit_global_mean_reward_bps = (
        float(sum(sum(values) for values in bandit_rewards_by_symbol.values()))
        / float(max(1, bandit_global_samples))
        if bandit_global_samples > 0
        else 0.0
    )
    bandit_global_series = [
        float(value)
        for values in bandit_rewards_by_symbol.values()
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    bandit_mean_for_significance, bandit_std_for_significance, bandit_samples_for_significance = mean_std_func(
        bandit_global_series
    )
    bandit_significance_context = sequential_significance_gate_func(
        mean_reward_bps=float(bandit_mean_for_significance),
        std_reward_bps=float(bandit_std_for_significance),
        samples=int(bandit_samples_for_significance),
        min_samples=int(max(bandit_promote_min_samples, bandit_min_samples)),
        target_mean_bps=float(bandit_promote_min_mean_reward_bps),
        method=str(promotion_significance_method),
        posterior_prob_min=float(promotion_significance_posterior_prob_min),
        sprt_alpha=float(promotion_significance_sprt_alpha),
        sprt_beta=float(promotion_significance_sprt_beta),
        sprt_effect_bps=float(promotion_significance_sprt_effect_bps),
    )
    bandit_significance_pass = (
        bool(bandit_significance_context.get("passed", False))
        if promotion_significance_enabled
        else True
    )
    bandit_live_promoted = bool(
        bandit_enabled
        and (
            not bandit_shadow_only
            or (
                bandit_auto_promote
                and bandit_global_samples >= bandit_promote_min_samples
                and bandit_global_mean_reward_bps >= bandit_promote_min_mean_reward_bps
                and bandit_significance_pass
            )
        )
    )
    bandit_bucket_significance_cache: dict[str, dict[str, Any]] = {}

    counterfactual_global_events = int(counterfactual_global.get("events", 0) or 0)
    counterfactual_global_dr_mean_bps = float(counterfactual_global.get("dr_mean_bps", 0.0) or 0.0)
    counterfactual_bucket_dr_means = [
        float(safe_float(item.get("dr_mean_bps")) or 0.0)
        for item in counterfactual_buckets.values()
        if isinstance(item, Mapping) and safe_float(item.get("dr_mean_bps")) is not None
    ]
    _cf_bucket_mean, counterfactual_std_proxy, _cf_bucket_samples = mean_std_func(
        counterfactual_bucket_dr_means
    )
    counterfactual_std_proxy = max(
        float(counterfactual_std_proxy),
        float(
            get_env(
                "AI_TRADING_EXEC_COUNTERFACTUAL_PROMOTION_STD_PROXY_BPS",
                15.0,
                cast=float,
            )
        ),
    )
    counterfactual_significance_context = sequential_significance_gate_func(
        mean_reward_bps=float(counterfactual_global_dr_mean_bps),
        std_reward_bps=float(counterfactual_std_proxy),
        samples=int(counterfactual_global_events),
        min_samples=int(counterfactual_promote_min_events),
        target_mean_bps=float(counterfactual_promote_min_dr_mean_bps),
        method=str(promotion_significance_method),
        posterior_prob_min=float(promotion_significance_posterior_prob_min),
        sprt_alpha=float(promotion_significance_sprt_alpha),
        sprt_beta=float(promotion_significance_sprt_beta),
        sprt_effect_bps=float(promotion_significance_sprt_effect_bps),
    )
    counterfactual_significance_pass = (
        bool(counterfactual_significance_context.get("passed", False))
        if promotion_significance_enabled
        else True
    )
    counterfactual_live_promoted = bool(
        counterfactual_enabled
        and (
            not counterfactual_shadow_only
            or (
                counterfactual_auto_promote
                and counterfactual_global_events >= counterfactual_promote_min_events
                and counterfactual_global_dr_mean_bps >= counterfactual_promote_min_dr_mean_bps
                and counterfactual_significance_pass
            )
        )
    )

    portfolio_rank_correlation: dict[str, dict[str, float]] = (
        build_symbol_return_correlation_matrix_func(symbol_returns)
        if portfolio_log_growth_rank_enabled
        else {}
    )
    portfolio_current_gross_for_rank = 0.0
    held_notional_weights: dict[str, float] = {}
    if portfolio_log_growth_rank_enabled:
        for held_symbol, held_qty in positions.items():
            held_price = safe_float(latest_price.get(held_symbol))
            if held_price is None or held_price <= 0.0:
                continue
            held_notional = abs(float(held_qty) * float(held_price))
            if held_notional <= 0.0:
                continue
            portfolio_current_gross_for_rank += held_notional
            held_notional_weights[str(held_symbol)] = held_notional
        if portfolio_current_gross_for_rank > 0.0:
            held_notional_weights = {
                symbol: float(notional) / float(portfolio_current_gross_for_rank)
                for symbol, notional in held_notional_weights.items()
            }
        else:
            held_notional_weights = {}

    return NettingLearningState(
        bandit_rewards_by_symbol=bandit_rewards_by_symbol,
        bandit_rewards_by_symbol_session=bandit_rewards_by_symbol_session,
        bandit_rewards_by_symbol_session_regime=bandit_rewards_by_symbol_session_regime,
        realized_edge_by_symbol=realized_edge_by_symbol,
        realized_edge_by_symbol_session=realized_edge_by_symbol_session,
        realized_edge_by_symbol_session_regime=realized_edge_by_symbol_session_regime,
        edge_realism_ratio_by_symbol=edge_realism_ratio_by_symbol,
        edge_realism_ratio_by_symbol_session=edge_realism_ratio_by_symbol_session,
        edge_realism_rank_factor_by_symbol=edge_realism_rank_factor_by_symbol,
        counterfactual_signal_by_symbol=counterfactual_signal_by_symbol,
        opportunity_quality_by_symbol=opportunity_quality_by_symbol,
        learned_fill_trials_by_bucket=learned_fill_trials_by_bucket,
        learned_fill_success_by_bucket=learned_fill_success_by_bucket,
        learned_exec_cost_stats_by_bucket=learned_exec_cost_stats_by_bucket,
        expected_capture_observed_values=expected_capture_observed_values,
        opportunity_quality_gate=opportunity_quality_gate,
        bandit_active_session=bandit_active_session,
        bandit_active_regime=bandit_active_regime,
        counterfactual_global=counterfactual_global,
        counterfactual_buckets=counterfactual_buckets,
        execution_learning_state=execution_learning_state,
        rejection_concentration_by_symbol=rejection_concentration_by_symbol,
        bandit_global_samples=bandit_global_samples,
        bandit_global_mean_reward_bps=bandit_global_mean_reward_bps,
        bandit_significance_context=dict(bandit_significance_context),
        bandit_significance_pass=bool(bandit_significance_pass),
        bandit_live_promoted=bool(bandit_live_promoted),
        bandit_bucket_significance_cache=bandit_bucket_significance_cache,
        counterfactual_global_events=counterfactual_global_events,
        counterfactual_global_dr_mean_bps=float(counterfactual_global_dr_mean_bps),
        counterfactual_significance_context=dict(counterfactual_significance_context),
        counterfactual_significance_pass=bool(counterfactual_significance_pass),
        counterfactual_live_promoted=bool(counterfactual_live_promoted),
        portfolio_rank_correlation=portfolio_rank_correlation,
        portfolio_current_gross_for_rank=float(portfolio_current_gross_for_rank),
        held_notional_weights=held_notional_weights,
        expected_capture_floor_bps_effective=float(expected_capture_floor_bps_effective),
    )


__all__ = [
    "NettingLearningState",
    "build_netting_learning_state",
    "expected_capture_bucket_keys",
    "lookup_learned_execution_cost_components",
    "lookup_learned_fill_probability",
]
