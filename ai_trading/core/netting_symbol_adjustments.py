"""Per-symbol quantity adjustment helpers for the live netting cycle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ai_trading.config.management import get_env


@dataclass(slots=True)
class SymbolAdjustmentResult:
    delta_shares: int
    target_shares: int
    target_dollars: float
    gates_added: tuple[str, ...]
    snapshot_updates: dict[str, Any]
    blocked_reason: str | None
    blocked_metrics: dict[str, Any] | None
    uncertainty_event: dict[str, Any] | None


def apply_symbol_adjustments(
    *,
    symbol: str,
    state: Any,
    cfg: Any,
    current_shares: int,
    delta_shares: int,
    price: float,
    expanding_exposure: bool,
    initial_requested_delta_shares: int,
    symbol_adaptive_profiles: Mapping[str, Mapping[str, Any]],
    slo_derisk_details: Mapping[str, Any],
    primary_feed_derisk: Mapping[str, Any],
    penalty_overlap_coordination_enabled: bool,
    penalty_overlap_weight_dampen: float,
    penalty_overlap_min_scale_floor: float,
    uncertainty_capital_state: Mapping[str, Any],
    safe_float: Callable[[Any], float | None],
    resolve_uncertainty_capital_auto_controls_func: Callable[..., dict[str, Any]],
    clip_delta_to_symbol_notional_cap_func: Callable[..., tuple[int, dict[str, Any] | None]],
    logger: Any,
) -> SymbolAdjustmentResult:
    snapshot_updates: dict[str, Any] = {}
    gates_added: list[str] = []
    blocked_reason: str | None = None
    blocked_metrics: dict[str, Any] | None = None
    uncertainty_event: dict[str, Any] | None = None

    adaptive_profile = symbol_adaptive_profiles.get(symbol)
    if adaptive_profile and delta_shares != 0:
        adaptive_scale = safe_float(adaptive_profile.get("scale")) or 1.0
        adaptive_scale = max(0.05, min(adaptive_scale, 1.0))
        if adaptive_scale < 1.0:
            scaled_qty = int(round(float(delta_shares) * adaptive_scale))
            if scaled_qty == 0:
                blocked_reason = "SYMBOL_EXPECTANCY_SLIPPAGE_BLOCK"
                blocked_metrics = {"symbol_adaptive_sizing": dict(adaptive_profile)}
                gates_added.append(blocked_reason)
                return SymbolAdjustmentResult(
                    delta_shares=int(delta_shares),
                    target_shares=int(current_shares + delta_shares),
                    target_dollars=float((current_shares + delta_shares) * price),
                    gates_added=tuple(gates_added),
                    snapshot_updates=snapshot_updates,
                    blocked_reason=blocked_reason,
                    blocked_metrics=blocked_metrics,
                    uncertainty_event=uncertainty_event,
                )
            if scaled_qty != delta_shares:
                delta_shares = int(scaled_qty)
                gates_added.append("SYMBOL_EXPECTANCY_SLIPPAGE_SCALE")
                snapshot_updates["symbol_adaptive_sizing"] = dict(adaptive_profile)

    if (
        delta_shares != 0
        and expanding_exposure
        and bool(get_env("AI_TRADING_UNCERTAINTY_CAPITAL_SCALING_ENABLED", True, cast=bool))
    ):
        ece_value = max(0.0, float(slo_derisk_details.get("calibration_ece", 0.0) or 0.0))
        brier_value = max(0.0, float(slo_derisk_details.get("calibration_brier", 0.0) or 0.0))
        execution_drift_bps = max(
            0.0, float(slo_derisk_details.get("execution_drift_bps", 0.0) or 0.0)
        )
        feature_drift_psi = max(0.0, float(slo_derisk_details.get("drift_psi", 0.0) or 0.0))
        label_drift_psi = max(0.0, float(slo_derisk_details.get("label_drift_psi", 0.0) or 0.0))
        residual_drift_psi = max(
            0.0, float(slo_derisk_details.get("residual_drift_psi", 0.0) or 0.0)
        )
        min_samples_required = max(
            1,
            int(get_env("AI_TRADING_UNCERTAINTY_CAPITAL_MIN_SAMPLES", 10, cast=int)),
        )
        sample_counts = [
            int(slo_derisk_details.get("calibration_ece_samples", 0) or 0),
            int(slo_derisk_details.get("calibration_brier_samples", 0) or 0),
            int(slo_derisk_details.get("drift_samples", 0) or 0),
        ]
        sample_shortfall = max(
            0.0,
            float(min_samples_required - min(sample_counts or [0]))
            / float(max(1, min_samples_required)),
        )
        data_quality_penalty = 0.0
        if bool(primary_feed_derisk.get("triggered", False)):
            data_quality_penalty += 0.4
        degraded_count = len(getattr(state, "degraded_providers", set()) or set())
        if degraded_count > 0:
            data_quality_penalty += min(float(degraded_count) * 0.2, 0.6)
        uncertainty_components = {
            "epistemic": min(
                1.0,
                max(
                    ece_value
                    / max(
                        1e-6,
                        float(get_env("AI_TRADING_DERISK_SLO_MAX_CALIB_ECE", 0.15, cast=float)),
                    ),
                    brier_value
                    / max(
                        1e-6,
                        float(get_env("AI_TRADING_DERISK_SLO_MAX_CALIB_BRIER", 0.35, cast=float)),
                    ),
                ),
            ),
            "drift": min(
                1.0,
                max(
                    execution_drift_bps
                    / max(
                        1e-6,
                        float(
                            get_env(
                                "AI_TRADING_DERISK_SLO_MAX_EXEC_DRIFT_BPS",
                                35.0,
                                cast=float,
                            )
                        ),
                    ),
                    feature_drift_psi
                    / max(
                        1e-6,
                        float(
                            get_env(
                                "AI_TRADING_DERISK_SLO_MAX_FEATURE_DRIFT_PSI",
                                0.30,
                                cast=float,
                            )
                        ),
                    ),
                    label_drift_psi
                    / max(
                        1e-6,
                        float(
                            get_env(
                                "AI_TRADING_DERISK_SLO_MAX_LABEL_DRIFT_PSI",
                                0.30,
                                cast=float,
                            )
                        ),
                    ),
                    residual_drift_psi
                    / max(
                        1e-6,
                        float(
                            get_env(
                                "AI_TRADING_DERISK_SLO_MAX_RESIDUAL_DRIFT_PSI",
                                0.30,
                                cast=float,
                            )
                        ),
                    ),
                ),
            ),
            "data_quality": min(1.0, float(data_quality_penalty)),
            "sample_shortfall": min(1.0, float(sample_shortfall)),
        }
        uncertainty_score = min(
            1.0,
            (0.40 * float(uncertainty_components["epistemic"]))
            + (0.30 * float(uncertainty_components["drift"]))
            + (0.20 * float(uncertainty_components["data_quality"]))
            + (0.10 * float(uncertainty_components["sample_shortfall"])),
        )
        base_uncertainty_weight = max(
            0.0,
            min(
                1.0,
                float(get_env("AI_TRADING_UNCERTAINTY_CAPITAL_SCALING_WEIGHT", 0.60, cast=float)),
            ),
        )
        base_uncertainty_min_scale = max(
            0.05,
            min(
                1.0,
                float(
                    get_env("AI_TRADING_UNCERTAINTY_CAPITAL_SCALING_MIN_SCALE", 0.35, cast=float)
                ),
            ),
        )
        if penalty_overlap_coordination_enabled and initial_requested_delta_shares != 0:
            prior_scale_ratio = min(
                1.0,
                abs(float(delta_shares))
                / float(max(1, abs(int(initial_requested_delta_shares)))),
            )
            if prior_scale_ratio < 0.999:
                prior_reduction = max(0.0, 1.0 - float(prior_scale_ratio))
                damp_mult = max(
                    0.0,
                    1.0 - (float(penalty_overlap_weight_dampen) * float(prior_reduction)),
                )
                base_uncertainty_weight = max(
                    0.0,
                    min(1.0, float(base_uncertainty_weight) * float(damp_mult)),
                )
                coordinated_floor = max(
                    float(base_uncertainty_min_scale),
                    float(penalty_overlap_min_scale_floor) - (0.25 * float(prior_reduction)),
                )
                base_uncertainty_min_scale = max(0.05, min(1.0, float(coordinated_floor)))
        uncertainty_auto_controls = resolve_uncertainty_capital_auto_controls_func(
            base_weight=float(base_uncertainty_weight),
            base_min_scale=float(base_uncertainty_min_scale),
            raw_score=float(uncertainty_score),
            state_payload=uncertainty_capital_state,
        )
        uncertainty_weight = float(
            safe_float(uncertainty_auto_controls.get("weight")) or base_uncertainty_weight
        )
        uncertainty_min_scale = float(
            safe_float(uncertainty_auto_controls.get("min_scale")) or base_uncertainty_min_scale
        )
        effective_uncertainty_score = float(
            safe_float(uncertainty_auto_controls.get("effective_score")) or uncertainty_score
        )
        uncertainty_scale = max(
            float(uncertainty_min_scale),
            1.0 - (float(uncertainty_weight) * float(effective_uncertainty_score)),
        )
        uncertainty_event = {
            "symbol": str(symbol),
            "score": float(uncertainty_score),
            "effective_score": float(effective_uncertainty_score),
            "scale": float(uncertainty_scale),
            "components": dict(uncertainty_components),
            "auto_tune": dict(uncertainty_auto_controls),
            "scaled": False,
            "blocked": False,
        }
        if uncertainty_scale < 0.999:
            adjusted_qty = int(round(float(delta_shares) * float(uncertainty_scale)))
            if adjusted_qty == 0:
                blocked_reason = "UNCERTAINTY_CAPITAL_BLOCK"
                blocked_metrics = {
                    "uncertainty_capital_scaling": {
                        "scale": float(uncertainty_scale),
                        "score": float(uncertainty_score),
                        "effective_score": float(effective_uncertainty_score),
                        "components": uncertainty_components,
                        "auto_tune": dict(uncertainty_auto_controls),
                    }
                }
                uncertainty_event["blocked"] = True
                gates_added.append(blocked_reason)
                return SymbolAdjustmentResult(
                    delta_shares=int(delta_shares),
                    target_shares=int(current_shares + delta_shares),
                    target_dollars=float((current_shares + delta_shares) * price),
                    gates_added=tuple(gates_added),
                    snapshot_updates=snapshot_updates,
                    blocked_reason=blocked_reason,
                    blocked_metrics=blocked_metrics,
                    uncertainty_event=uncertainty_event,
                )
            if adjusted_qty != int(delta_shares):
                delta_shares = int(adjusted_qty)
                gates_added.append("UNCERTAINTY_CAPITAL_SCALE")
                uncertainty_event["scaled"] = True
                snapshot_updates["uncertainty_capital_scaling"] = {
                    "scale": float(uncertainty_scale),
                    "score": float(uncertainty_score),
                    "effective_score": float(effective_uncertainty_score),
                    "components": uncertainty_components,
                    "auto_tune": dict(uncertainty_auto_controls),
                }

    requested_delta_shares = int(delta_shares)
    reversal_clamp_reason: str | None = None
    if current_shares > 0 and delta_shares < 0 and abs(delta_shares) > current_shares:
        delta_shares = -current_shares
        reversal_clamp_reason = "FLAT_BEFORE_REVERSAL"
    elif current_shares < 0 and delta_shares > 0 and abs(delta_shares) > abs(current_shares):
        delta_shares = abs(current_shares)
        reversal_clamp_reason = "FLAT_BEFORE_REVERSAL"
    if reversal_clamp_reason:
        gates_added.append(reversal_clamp_reason)
        logger.info(
            "POSITION_REVERSAL_CLAMP",
            extra={
                "symbol": symbol,
                "current_shares": int(current_shares),
                "requested_delta_shares": requested_delta_shares,
                "adjusted_delta_shares": int(delta_shares),
                "target_shares": 0,
                "reason": reversal_clamp_reason,
            },
        )

    try:
        max_symbol_notional = float(
            getattr(
                cfg,
                "global_max_symbol_dollars",
                get_env("GLOBAL_MAX_SYMBOL_DOLLARS", 25000.0, cast=float),
            )
            or 0.0
        )
    except (TypeError, ValueError):
        max_symbol_notional = float(get_env("GLOBAL_MAX_SYMBOL_DOLLARS", 25000.0, cast=float))

    delta_shares_capped, symbol_cap_details = clip_delta_to_symbol_notional_cap_func(
        symbol=symbol,
        current_shares=int(current_shares),
        delta_shares=int(delta_shares),
        price=float(price),
        max_symbol_notional=max_symbol_notional,
    )
    if symbol_cap_details is not None:
        if int(delta_shares_capped) == 0:
            blocked_reason = "RISK_CAP_SYMBOL"
            blocked_metrics = {"symbol_cap": symbol_cap_details}
            gates_added.append(blocked_reason)
            return SymbolAdjustmentResult(
                delta_shares=int(delta_shares),
                target_shares=int(current_shares + delta_shares),
                target_dollars=float((current_shares + delta_shares) * price),
                gates_added=tuple(gates_added),
                snapshot_updates=snapshot_updates,
                blocked_reason=blocked_reason,
                blocked_metrics=blocked_metrics,
                uncertainty_event=uncertainty_event,
            )
        delta_shares = int(delta_shares_capped)
        gates_added.append("RISK_CAP_SYMBOL_CLIP")
        snapshot_updates["symbol_cap_clip"] = symbol_cap_details
        logger.info("SYMBOL_NOTIONAL_CAP_CLIP", extra=symbol_cap_details)

    min_qty = int(getattr(cfg, "execution_min_qty", 1))
    min_notional = float(getattr(cfg, "execution_min_notional", 1.0))
    if abs(delta_shares) < min_qty or abs(delta_shares) * price < min_notional:
        blocked_reason = "RISK_CAP_SYMBOL"
        blocked_metrics = None
        gates_added.append(blocked_reason)

    target_shares = int(current_shares + delta_shares)
    target_dollars = float(target_shares * price)
    if reversal_clamp_reason:
        target_shares = 0
        target_dollars = 0.0

    return SymbolAdjustmentResult(
        delta_shares=int(delta_shares),
        target_shares=int(target_shares),
        target_dollars=float(target_dollars),
        gates_added=tuple(gates_added),
        snapshot_updates=snapshot_updates,
        blocked_reason=blocked_reason,
        blocked_metrics=blocked_metrics,
        uncertainty_event=uncertainty_event,
    )
