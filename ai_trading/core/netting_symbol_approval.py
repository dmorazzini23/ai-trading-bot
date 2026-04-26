"""Pre-approval symbol orchestration for the live netting cycle."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

from ai_trading.config.management import get_env
from ai_trading.policy.compiler import SafetyTier


def _side_for_delta(delta_shares: int, current_shares: int, target_shares: int) -> str:
    if int(delta_shares) > 0:
        return "buy"
    if int(current_shares) > 0:
        return "sell"
    if int(target_shares) < 0:
        return "sell_short"
    return "sell"


@dataclass(frozen=True, slots=True)
class NettingSymbolApprovalResult:
    delta_shares: int
    target_shares: int
    target_dollars: float
    side: str
    opening_trade: bool
    gates_added: tuple[str, ...]
    snapshot_updates: dict[str, Any]
    blocked_reason: str | None
    blocked_metrics: dict[str, Any] | None
    approval: Any | None
    approval_context: Any | None


def prepare_netting_symbol_approval(
    *,
    state: Any,
    symbol: str,
    now: Any,
    current_shares: int,
    delta_shares: int,
    price: float,
    net_target: Any,
    liq_features: Any,
    liq_regime: Any,
    exec_engine: Any,
    effective_policy: Any,
    candidate_expected_net_edge: Mapping[str, Any],
    edge_realism_rank_factor_by_symbol: Mapping[str, Any],
    edge_realism_apply_to_approval_enabled: bool,
    alpha_decay_deweight_enabled: bool,
    alpha_decay_qty_step: float,
    alpha_decay_qty_max_deweight: float,
    capacity_throttle_enabled: bool,
    capacity_spread_soft_bps: float,
    capacity_spread_hard_bps: float,
    capacity_volume_soft_participation: float,
    capacity_volume_hard_participation: float,
    capacity_min_scale: float,
    slo_derisk_scale: float,
    slo_derisk_details: dict[str, Any],
    primary_feed_derisk: dict[str, Any],
    feed_derisk_scale: float,
    portfolio_current_gross: float,
    sector_gross: Mapping[str, float],
    max_new_orders_per_cycle: int | None,
    orders_submitted: int,
    gates: list[str],
    symbol_snapshot: dict[str, Any],
    gate_blocks_func: Callable[[str], bool],
    clip_sell_qty_to_available_position_func: Callable[..., tuple[int, dict[str, Any] | None]],
    percentile_linear_func: Callable[[list[float], float], float | None],
    slippage_setting_bps_func: Callable[[], float],
    safe_float_func: Callable[[Any], float | None],
    get_sector_func: Callable[[str], str | None],
    alpha_decay_entry_guard_func: Callable[..., Mapping[str, Any]],
    evaluate_execution_approval_func: Callable[..., Any],
    approve_execution_candidate_func: Callable[..., Any],
) -> NettingSymbolApprovalResult:
    snapshot_updates: dict[str, Any] = {}
    gates_added: list[str] = []
    delta_shares_value = int(delta_shares)
    target_shares = int(current_shares + delta_shares_value)
    target_dollars = float(target_shares * price)
    side = _side_for_delta(delta_shares_value, int(current_shares), target_shares)

    def _clip_sell_to_available_position() -> NettingSymbolApprovalResult | None:
        nonlocal delta_shares_value, target_shares, target_dollars, side
        side = _side_for_delta(delta_shares_value, int(current_shares), target_shares)
        if side != "sell":
            return None
        requested_sell_qty = abs(int(delta_shares_value))
        adjusted_sell_qty, sell_qty_clip_context = clip_sell_qty_to_available_position_func(
            symbol=symbol,
            current_shares=int(current_shares),
            requested_qty=requested_sell_qty,
            exec_engine=exec_engine,
        )
        if adjusted_sell_qty <= 0:
            blocked_reason = "PRE_SUBMIT_INSUFFICIENT_POSITION_AVAILABLE"
            gates_added.append(blocked_reason)
            return NettingSymbolApprovalResult(
                delta_shares=delta_shares_value,
                target_shares=target_shares,
                target_dollars=target_dollars,
                side=side,
                opening_trade=False,
                gates_added=tuple(gates_added),
                snapshot_updates=snapshot_updates,
                blocked_reason=blocked_reason,
                blocked_metrics={"pre_submit_sell_qty_clip": sell_qty_clip_context or {}},
                approval=None,
                approval_context=None,
            )
        if adjusted_sell_qty != requested_sell_qty:
            delta_shares_value = -int(adjusted_sell_qty)
            target_shares = int(current_shares + delta_shares_value)
            target_dollars = float(target_shares * price)
            if "PRE_SUBMIT_SELL_QTY_CLIP_AVAILABLE_POSITION" not in gates_added:
                gates_added.append("PRE_SUBMIT_SELL_QTY_CLIP_AVAILABLE_POSITION")
            if sell_qty_clip_context:
                snapshot_updates["pre_submit_sell_qty_clip"] = dict(sell_qty_clip_context)
        side = _side_for_delta(delta_shares_value, int(current_shares), target_shares)
        return None

    def _clip_buy_to_short_cover() -> None:
        nonlocal delta_shares_value, target_shares, target_dollars, side
        if int(current_shares) >= 0 or int(delta_shares_value) <= 0:
            side = _side_for_delta(delta_shares_value, int(current_shares), target_shares)
            return
        max_cover_qty = abs(int(current_shares))
        requested_qty = int(delta_shares_value)
        if requested_qty <= max_cover_qty:
            side = _side_for_delta(delta_shares_value, int(current_shares), target_shares)
            return
        delta_shares_value = int(max_cover_qty)
        target_shares = int(current_shares + delta_shares_value)
        target_dollars = float(target_shares * price)
        side = _side_for_delta(delta_shares_value, int(current_shares), target_shares)
        if "PRE_SUBMIT_BUY_QTY_CLIP_SHORT_COVER" not in gates_added:
            gates_added.append("PRE_SUBMIT_BUY_QTY_CLIP_SHORT_COVER")
        snapshot_updates["pre_submit_buy_qty_clip"] = {
            "current_shares": int(current_shares),
            "requested_qty": int(requested_qty),
            "max_cover_qty": int(max_cover_qty),
        }

    def _clip_cross_zero_to_close_only() -> NettingSymbolApprovalResult | None:
        _clip_buy_to_short_cover()
        return _clip_sell_to_available_position()

    blocked_by_close_clip = _clip_cross_zero_to_close_only()
    if blocked_by_close_clip is not None:
        return blocked_by_close_clip

    opening_trade = abs(current_shares + delta_shares_value) > abs(current_shares)
    if exec_engine is not None:
        if opening_trade:
            min_notional_precheck_fn = getattr(exec_engine, "_opening_min_notional_allows_order", None)
            if callable(min_notional_precheck_fn):
                precheck_order_payload: dict[str, Any] = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": abs(int(delta_shares_value)),
                    "qty": abs(int(delta_shares_value)),
                    "price_hint": float(price),
                }
                try:
                    opening_notional_allowed, opening_notional_context = min_notional_precheck_fn(
                        precheck_order_payload
                    )
                except AI_TRADING_FALLBACK_EXCEPTIONS:
                    opening_notional_allowed, opening_notional_context = True, {}
                opening_notional_context = (
                    dict(opening_notional_context)
                    if isinstance(opening_notional_context, Mapping)
                    else {}
                )
                if not bool(opening_notional_allowed):
                    blocked_reason = "ENTRY_CONSTRAINED_MIN_NOTIONAL_PRECHECK"
                    gates_added.append(blocked_reason)
                    return NettingSymbolApprovalResult(
                        delta_shares=delta_shares_value,
                        target_shares=target_shares,
                        target_dollars=target_dollars,
                        side=side,
                        opening_trade=opening_trade,
                        gates_added=tuple(gates_added),
                        snapshot_updates=snapshot_updates,
                        blocked_reason=blocked_reason,
                        blocked_metrics={"opening_min_notional": opening_notional_context},
                        approval=None,
                        approval_context=None,
                    )
                autosized_qty: int | None = None
                try:
                    autosized_qty = int(precheck_order_payload.get("quantity"))
                except (TypeError, ValueError):
                    try:
                        autosized_qty = int(precheck_order_payload.get("qty"))
                    except (TypeError, ValueError):
                        autosized_qty = None
                if autosized_qty is not None and autosized_qty > 0:
                    signed_autosized_qty = (
                        int(autosized_qty) if int(delta_shares_value) > 0 else -int(autosized_qty)
                    )
                    if signed_autosized_qty != int(delta_shares_value):
                        delta_shares_value = int(signed_autosized_qty)
                        target_shares = int(current_shares + delta_shares_value)
                        target_dollars = float(target_shares * price)
                        gates_added.append("ENTRY_CONSTRAINED_MIN_NOTIONAL_AUTOSIZE_PRECHECK")
                        snapshot_updates["opening_min_notional_precheck"] = {
                            **opening_notional_context,
                            "autosized_qty": int(abs(delta_shares_value)),
                        }
            else:
                min_notional_fn = getattr(exec_engine, "_opening_min_notional_dollars", None)
                if callable(min_notional_fn):
                    try:
                        opening_min_notional = float(min_notional_fn(symbol=symbol) or 0.0)
                    except TypeError:
                        opening_min_notional = float(min_notional_fn() or 0.0)
                    except AI_TRADING_FALLBACK_EXCEPTIONS:
                        opening_min_notional = 0.0
                    if opening_min_notional > 0.0:
                        opening_notional = abs(float(delta_shares_value) * float(price))
                        if opening_notional < opening_min_notional:
                            blocked_reason = "ENTRY_CONSTRAINED_MIN_NOTIONAL_PRECHECK"
                            gates_added.append(blocked_reason)
                            return NettingSymbolApprovalResult(
                                delta_shares=delta_shares_value,
                                target_shares=target_shares,
                                target_dollars=target_dollars,
                                side=side,
                                opening_trade=opening_trade,
                                gates_added=tuple(gates_added),
                                snapshot_updates=snapshot_updates,
                                blocked_reason=blocked_reason,
                                blocked_metrics={
                                    "opening_min_notional": {
                                        "order_notional": float(opening_notional),
                                        "min_notional": float(opening_min_notional),
                                    }
                                },
                                approval=None,
                                approval_context=None,
                            )

            cooldown_fn = getattr(exec_engine, "_symbol_reentry_cooldown_allows_opening", None)
            if callable(cooldown_fn):
                try:
                    cooldown_allowed, cooldown_context = cooldown_fn(symbol=symbol, side=side)
                except AI_TRADING_FALLBACK_EXCEPTIONS:
                    cooldown_allowed, cooldown_context = True, {}
                if not bool(cooldown_allowed):
                    blocked_reason = "ENTRY_CONSTRAINED_SYMBOL_REENTRY_COOLDOWN_PRECHECK"
                    gates_added.append(blocked_reason)
                    return NettingSymbolApprovalResult(
                        delta_shares=delta_shares_value,
                        target_shares=target_shares,
                        target_dollars=target_dollars,
                        side=side,
                        opening_trade=opening_trade,
                        gates_added=tuple(gates_added),
                        snapshot_updates=snapshot_updates,
                        blocked_reason=blocked_reason,
                        blocked_metrics={"symbol_reentry_cooldown": cooldown_context},
                        approval=None,
                        approval_context=None,
                    )

        duplicate_fn = getattr(exec_engine, "_should_suppress_duplicate_intent", None)
        if callable(duplicate_fn):
            try:
                duplicate_suppressed = bool(duplicate_fn(symbol, side))
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                duplicate_suppressed = False
            if duplicate_suppressed:
                blocked_reason = "DUPLICATE_INTENT_PRECHECK"
                gates_added.append(blocked_reason)
                return NettingSymbolApprovalResult(
                    delta_shares=delta_shares_value,
                    target_shares=target_shares,
                    target_dollars=target_dollars,
                    side=side,
                    opening_trade=opening_trade,
                    gates_added=tuple(gates_added),
                    snapshot_updates=snapshot_updates,
                    blocked_reason=blocked_reason,
                    blocked_metrics=None,
                    approval=None,
                    approval_context=None,
                )

    expected_edge_total_raw = sum(
        max(float(proposal.expected_edge_bps), 0.0) for proposal in net_target.proposals
    )
    parsed_edge_realism_factor = safe_float_func(
        edge_realism_rank_factor_by_symbol.get(symbol, 1.0)
    )
    expected_edge_realism_factor = (
        1.0 if parsed_edge_realism_factor is None else float(parsed_edge_realism_factor)
    )
    expected_edge_total = float(expected_edge_total_raw)
    if edge_realism_apply_to_approval_enabled and float(expected_edge_total) > 0.0:
        expected_edge_total = float(expected_edge_total) * float(expected_edge_realism_factor)
    expected_cost_total = sum(
        max(float(proposal.expected_cost_bps), 0.0) for proposal in net_target.proposals
    )

    cost_aware_guard_enabled = bool(
        get_env("AI_TRADING_EXECUTION_COST_AWARE_ENTRY_GUARD_ENABLED", True, cast=bool)
    )
    if cost_aware_guard_enabled and opening_trade:
        adaptive_cost_context: dict[str, Any] = {}
        adaptive_cost_add_bps = 0.0
        opening_ramp_context: dict[str, Any] = {}
        opening_ramp_add_bps = 0.0
        if exec_engine is not None:
            adaptive_cost_hook = getattr(exec_engine, "_cost_aware_entry_adaptive_context", None)
            if callable(adaptive_cost_hook):
                try:
                    adaptive_cost_result = adaptive_cost_hook()
                except AI_TRADING_FALLBACK_EXCEPTIONS:
                    adaptive_cost_result = {}
                if isinstance(adaptive_cost_result, Mapping):
                    adaptive_cost_context = dict(adaptive_cost_result)
                    try:
                        adaptive_cost_add_bps = max(
                            0.0,
                            float(
                                adaptive_cost_context.get("additional_required_edge_bps", 0.0)
                                or 0.0
                            ),
                        )
                    except (TypeError, ValueError):
                        adaptive_cost_add_bps = 0.0
            opening_ramp_hook = getattr(exec_engine, "_opening_ramp_context", None)
            if callable(opening_ramp_hook):
                try:
                    opening_ramp_result = opening_ramp_hook()
                except AI_TRADING_FALLBACK_EXCEPTIONS:
                    opening_ramp_result = {}
                if isinstance(opening_ramp_result, Mapping):
                    opening_ramp_context = dict(opening_ramp_result)
                    try:
                        opening_ramp_add_bps = max(
                            0.0,
                            float(opening_ramp_context.get("required_edge_add_bps", 0.0) or 0.0),
                        )
                    except (TypeError, ValueError):
                        opening_ramp_add_bps = 0.0
        spread_bps_est = max(float(liq_features.spread_bps), 0.0)
        slippage_bps_est = max(
            0.0,
            float(
                get_env(
                    "AI_TRADING_EXECUTION_COST_AWARE_SLIPPAGE_BPS",
                    slippage_setting_bps_func(),
                    cast=float,
                )
            ),
        )
        fee_bps_est = max(0.0, float(effective_policy.objective.fee_bps))
        borrow_bps_est = (
            max(0.0, float(effective_policy.objective.borrow_bps))
            if side == "sell_short"
            else 0.0
        )
        edge_margin_bps = max(
            0.0,
            float(
                get_env(
                    "AI_TRADING_EXECUTION_COST_AWARE_EDGE_MARGIN_BPS",
                    2.0,
                    cast=float,
                )
            ),
        )
        cost_multiplier = max(
            1.0,
            float(
                get_env(
                    "AI_TRADING_EXECUTION_COST_AWARE_COST_MULTIPLIER",
                    1.15,
                    cast=float,
                )
            ),
        )
        min_edge_to_cost_ratio = max(
            0.0,
            float(
                get_env(
                    "AI_TRADING_EXECUTION_COST_AWARE_MIN_EDGE_TO_COST_RATIO",
                    1.2,
                    cast=float,
                )
            ),
        )
        quote_cost_bps = spread_bps_est + slippage_bps_est + fee_bps_est + borrow_bps_est
        effective_cost_floor = max(float(expected_cost_total), float(quote_cost_bps)) * float(
            cost_multiplier
        )
        required_edge_bps = max(
            float(effective_cost_floor + edge_margin_bps),
            float(effective_cost_floor * min_edge_to_cost_ratio),
        )
        required_edge_bps += float(adaptive_cost_add_bps + opening_ramp_add_bps)
        adaptive_edge_floor_enabled = bool(
            get_env(
                "AI_TRADING_EXECUTION_COST_AWARE_ADAPTIVE_EDGE_FLOOR_ENABLED",
                True,
                cast=bool,
            )
        )
        adaptive_edge_floor_percentile = max(
            0.05,
            min(
                0.95,
                float(
                    get_env(
                        "AI_TRADING_EXECUTION_COST_AWARE_ADAPTIVE_EDGE_FLOOR_PERCENTILE",
                        0.60,
                        cast=float,
                    )
                ),
            ),
        )
        adaptive_edge_floor_value = None
        if adaptive_edge_floor_enabled and candidate_expected_net_edge:
            adaptive_edge_floor_value = percentile_linear_func(
                list(candidate_expected_net_edge.values()),
                adaptive_edge_floor_percentile,
            )
            if adaptive_edge_floor_value is not None:
                required_edge_bps = max(float(required_edge_bps), float(adaptive_edge_floor_value))
        if float(expected_edge_total) < required_edge_bps and gate_blocks_func(
            "COST_AWARE_ENTRY_GUARD"
        ):
            blocked_reason = "COST_AWARE_ENTRY_GUARD"
            gates_added.append(blocked_reason)
            return NettingSymbolApprovalResult(
                delta_shares=delta_shares_value,
                target_shares=target_shares,
                target_dollars=target_dollars,
                side=side,
                opening_trade=opening_trade,
                gates_added=tuple(gates_added),
                snapshot_updates=snapshot_updates,
                blocked_reason=blocked_reason,
                blocked_metrics={
                    "cost_aware_entry_guard": {
                        "expected_edge_bps_raw": float(expected_edge_total_raw),
                        "expected_edge_bps": float(expected_edge_total),
                        "expected_edge_realism_factor": float(expected_edge_realism_factor),
                        "expected_edge_realism_applied": bool(edge_realism_apply_to_approval_enabled),
                        "expected_cost_bps": float(expected_cost_total),
                        "spread_bps": float(spread_bps_est),
                        "slippage_bps": float(slippage_bps_est),
                        "fee_bps": float(fee_bps_est),
                        "borrow_bps": float(borrow_bps_est),
                        "edge_margin_bps": float(edge_margin_bps),
                        "cost_multiplier": float(cost_multiplier),
                        "edge_to_cost_min_ratio": float(min_edge_to_cost_ratio),
                        "effective_cost_floor_bps": float(effective_cost_floor),
                        "adaptive_cost_add_bps": float(adaptive_cost_add_bps),
                        "opening_ramp_add_bps": float(opening_ramp_add_bps),
                        "required_edge_bps": float(required_edge_bps),
                        "adaptive_edge_floor_enabled": bool(adaptive_edge_floor_enabled),
                        "adaptive_edge_floor_percentile": float(adaptive_edge_floor_percentile),
                        "adaptive_edge_floor_bps": (
                            float(adaptive_edge_floor_value)
                            if adaptive_edge_floor_value is not None
                            else None
                        ),
                        "adaptive_cost_context": adaptive_cost_context,
                        "opening_ramp_context": opening_ramp_context,
                    }
                },
                approval=None,
                approval_context=None,
            )

    calibration_samples = int(
        max(
            float(slo_derisk_details.get("calibration_ece_samples", 0) or 0.0),
            float(slo_derisk_details.get("calibration_brier_samples", 0) or 0.0),
        )
    )
    ece_value = float(slo_derisk_details.get("calibration_ece", 0.0) or 0.0)
    brier_value = float(slo_derisk_details.get("calibration_brier", 0.0) or 0.0)
    if liq_regime is not None and str(getattr(liq_regime, "name", liq_regime)) == "THIN":
        ece_limit = float(effective_policy.calibration.max_ece_stress)
        brier_limit = float(effective_policy.calibration.max_brier_stress)
    else:
        ece_limit = float(effective_policy.calibration.max_ece_normal)
        brier_limit = float(effective_policy.calibration.max_brier_normal)
    calibration_ok = calibration_samples < int(effective_policy.calibration.min_samples) or (
        ece_value <= ece_limit and brier_value <= brier_limit
    )
    sector_name = str(get_sector_func(symbol) or "UNKNOWN").upper()
    safety_tier_raw = str(
        getattr(state, "operational_safety_tier", SafetyTier.NORMAL.value)
        or SafetyTier.NORMAL.value
    )
    approval_context = evaluate_execution_approval_func(
        effective_policy=effective_policy,
        symbol=symbol,
        side=side,
        delta_shares=int(delta_shares_value),
        current_shares=float(current_shares),
        price=float(price),
        expected_edge_total=float(expected_edge_total),
        expected_cost_total=float(expected_cost_total),
        proposals=net_target.proposals,
        spread_bps=float(liq_features.spread_bps),
        rolling_volume=float(liq_features.rolling_volume),
        pending_oldest_age_sec=float(slo_derisk_details.get("pending_oldest_age_sec", 0.0) or 0.0),
        calibration_ok=calibration_ok,
        reject_rate_pct=float(slo_derisk_details.get("reject_rate_pct", 0.0) or 0.0),
        portfolio_current_gross=float(portfolio_current_gross),
        sector_gross=sector_gross,
        sector_name=sector_name,
        max_new_orders_per_cycle=max_new_orders_per_cycle,
        orders_submitted=int(orders_submitted),
        engine_cycle_new_orders_submitted=(
            getattr(exec_engine, "_cycle_new_orders_submitted", orders_submitted)
            if exec_engine is not None
            else orders_submitted
        ),
        safety_tier_raw=safety_tier_raw,
        approval_func=approve_execution_candidate_func,
    )
    approval = approval_context.approval
    if approval.reasons:
        for reason in approval.reasons:
            if reason not in gates and reason not in gates_added:
                gates_added.append(reason)
    if not approval.allowed:
        return NettingSymbolApprovalResult(
            delta_shares=delta_shares_value,
            target_shares=target_shares,
            target_dollars=target_dollars,
            side=side,
            opening_trade=opening_trade,
            gates_added=tuple(gates_added),
            snapshot_updates=snapshot_updates,
            blocked_reason="EXECUTION_APPROVAL_BLOCK",
            blocked_metrics={"expected_net_edge_bps": approval.expected_net_edge_bps},
            approval=approval,
            approval_context=approval_context,
        )
    if int(approval_context.adjusted_delta_shares) != int(delta_shares_value):
        delta_shares_value = int(approval_context.adjusted_delta_shares)
        target_shares = int(current_shares + delta_shares_value)
        target_dollars = float(target_shares * price)
    blocked_by_close_clip = _clip_cross_zero_to_close_only()
    if blocked_by_close_clip is not None:
        return blocked_by_close_clip
    opening_trade = abs(current_shares + delta_shares_value) > abs(current_shares)

    if current_shares == 0 and alpha_decay_deweight_enabled:
        alpha_guard = alpha_decay_entry_guard_func(state, symbol, now)
        if alpha_guard.get("blocked"):
            if gate_blocks_func("ALPHA_DECAY_BLOCK"):
                blocked_reason = "ALPHA_DECAY_BLOCK"
                gates_added.append(blocked_reason)
                return NettingSymbolApprovalResult(
                    delta_shares=delta_shares_value,
                    target_shares=target_shares,
                    target_dollars=target_dollars,
                    side=side,
                    opening_trade=opening_trade,
                    gates_added=tuple(gates_added),
                    snapshot_updates=snapshot_updates,
                    blocked_reason=blocked_reason,
                    blocked_metrics={"alpha_decay": dict(alpha_guard)},
                    approval=approval,
                    approval_context=approval_context,
                )
            gates_added.append("ALPHA_DECAY_BLOCK_BYPASSED")
        trades_in_window = int(alpha_guard.get("trades_in_window", 0) or 0)
        start_trades = int(alpha_guard.get("start_trades", 0) or 0)
        over_start = max(0, trades_in_window - max(0, start_trades) + 1)
        if over_start > 0 and alpha_decay_qty_step > 0:
            deweight = min(alpha_decay_qty_max_deweight, over_start * alpha_decay_qty_step)
            multiplier = max(0.05, 1.0 - deweight)
            scaled_qty = int(round(float(delta_shares_value) * multiplier))
            if scaled_qty == 0:
                if gate_blocks_func("ALPHA_DECAY_ZERO_QTY_BLOCK"):
                    blocked_reason = "ALPHA_DECAY_ZERO_QTY_BLOCK"
                    gates_added.append(blocked_reason)
                    return NettingSymbolApprovalResult(
                        delta_shares=delta_shares_value,
                        target_shares=target_shares,
                        target_dollars=target_dollars,
                        side=side,
                        opening_trade=opening_trade,
                        gates_added=tuple(gates_added),
                        snapshot_updates=snapshot_updates,
                        blocked_reason=blocked_reason,
                        blocked_metrics={"alpha_decay": dict(alpha_guard) | {"multiplier": multiplier}},
                        approval=approval,
                        approval_context=approval_context,
                    )
                scaled_qty = 1 if delta_shares_value > 0 else -1
            if scaled_qty != delta_shares_value:
                delta_shares_value = scaled_qty
                target_shares = int(current_shares + delta_shares_value)
                target_dollars = float(target_shares * price)
                blocked_by_close_clip = _clip_cross_zero_to_close_only()
                if blocked_by_close_clip is not None:
                    return blocked_by_close_clip
                opening_trade = abs(current_shares + delta_shares_value) > abs(current_shares)
                gates_added.append("ALPHA_DECAY_DEWEIGHT")
                snapshot_updates["alpha_decay"] = {
                    "trades_in_window": trades_in_window,
                    "start_trades": start_trades,
                    "multiplier": multiplier,
                }

    if capacity_throttle_enabled and delta_shares_value != 0:
        capacity_scale = 1.0
        spread_bps_now = max(float(liq_features.spread_bps), 0.0)
        if capacity_spread_hard_bps > capacity_spread_soft_bps and spread_bps_now > capacity_spread_soft_bps:
            if spread_bps_now >= capacity_spread_hard_bps:
                spread_scale = capacity_min_scale
            else:
                spread_progress = (spread_bps_now - capacity_spread_soft_bps) / (
                    capacity_spread_hard_bps - capacity_spread_soft_bps
                )
                spread_scale = 1.0 - spread_progress * (1.0 - capacity_min_scale)
            capacity_scale = min(capacity_scale, max(capacity_min_scale, spread_scale))
        rolling_volume = max(float(liq_features.rolling_volume), 0.0)
        if rolling_volume > 0 and capacity_volume_hard_participation > capacity_volume_soft_participation:
            participation = abs(float(delta_shares_value)) / rolling_volume
            if participation > capacity_volume_soft_participation:
                if participation >= capacity_volume_hard_participation:
                    volume_scale = capacity_min_scale
                else:
                    participation_progress = (
                        participation - capacity_volume_soft_participation
                    ) / (capacity_volume_hard_participation - capacity_volume_soft_participation)
                    volume_scale = 1.0 - participation_progress * (1.0 - capacity_min_scale)
                capacity_scale = min(capacity_scale, max(capacity_min_scale, volume_scale))
        if slo_derisk_scale < 1.0:
            capacity_scale = min(capacity_scale, slo_derisk_scale)
        if feed_derisk_scale < 1.0:
            capacity_scale = min(capacity_scale, feed_derisk_scale)
        if capacity_scale < 1.0:
            throttled_qty = int(round(float(delta_shares_value) * capacity_scale))
            if throttled_qty == 0:
                blocked_reason = "CAPACITY_THROTTLE_BLOCK"
                gates_added.append(blocked_reason)
                return NettingSymbolApprovalResult(
                    delta_shares=delta_shares_value,
                    target_shares=target_shares,
                    target_dollars=target_dollars,
                    side=side,
                    opening_trade=opening_trade,
                    gates_added=tuple(gates_added),
                    snapshot_updates=snapshot_updates,
                    blocked_reason=blocked_reason,
                    blocked_metrics={
                        "capacity_scale": capacity_scale,
                        "spread_bps": spread_bps_now,
                        "rolling_volume": rolling_volume,
                        "slo_derisk": slo_derisk_details,
                        "primary_feed_derisk": primary_feed_derisk,
                    },
                    approval=approval,
                    approval_context=approval_context,
                )
            if throttled_qty != delta_shares_value:
                delta_shares_value = throttled_qty
                target_shares = int(current_shares + delta_shares_value)
                target_dollars = float(target_shares * price)
                blocked_by_close_clip = _clip_cross_zero_to_close_only()
                if blocked_by_close_clip is not None:
                    return blocked_by_close_clip
                opening_trade = abs(current_shares + delta_shares_value) > abs(current_shares)
                gates_added.append("CAPACITY_THROTTLE_SCALE")
                snapshot_updates["capacity_throttle"] = {
                    "scale": capacity_scale,
                    "spread_bps": spread_bps_now,
                    "rolling_volume": rolling_volume,
                    "slo_derisk": slo_derisk_details,
                    "primary_feed_derisk": primary_feed_derisk,
                }

    return NettingSymbolApprovalResult(
        delta_shares=delta_shares_value,
        target_shares=target_shares,
        target_dollars=target_dollars,
        side=side,
        opening_trade=opening_trade,
        gates_added=tuple(gates_added),
        snapshot_updates=snapshot_updates,
        blocked_reason=None,
        blocked_metrics=None,
        approval=approval,
        approval_context=approval_context,
    )


__all__ = ["NettingSymbolApprovalResult", "prepare_netting_symbol_approval"]
