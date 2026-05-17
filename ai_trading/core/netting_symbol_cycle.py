"""Per-symbol coordination for the live netting cycle."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Callable, Collection, Mapping, Sequence

from ai_trading.config.management import get_env
from ai_trading.config.launch_profiles import resolve_launch_profile
from ai_trading.risk.liquidity_regime import (
    LiquidityFeatures,
    LiquidityRegime,
    classify_liquidity_regime,
)

_NON_DISABLEABLE_GATE_NAMES = frozenset(
    {
        "LIQ_PARTICIPATION_BLOCK",
        "CAPACITY_THROTTLE_BLOCK",
    }
)
_NON_DISABLEABLE_GATE_ROOTS = frozenset({"LIQUIDITY_PARTICIPATION"})


@dataclass(slots=True)
class NettingSymbolProcessor:
    state: Any
    runtime: Any
    cfg: Any
    now: datetime
    logger: Any
    decision_snapshot_template: Mapping[str, Any]
    latest_price: Mapping[str, float]
    latest_liquidity: Mapping[str, LiquidityFeatures]
    positions: Mapping[str, float | int]
    skip_reasons: Mapping[str, Sequence[str]]
    kill_switch: bool
    policy_disabled_sleeves: set[str]
    policy_rollback_disabled_slices: Collection[str]
    sleeve_configs_map: Mapping[str, Any]
    candidate_expected_net_edge: Mapping[str, Any]
    candidate_expected_capture: Mapping[str, Any]
    alpha_time_stop_enabled: bool
    alpha_time_stop_sec: float
    alpha_time_stop_max_expected_edge_bps: float
    opportunity_quality_enabled: bool
    opportunity_allowed_symbols: set[str]
    opportunity_openings_only: bool
    opportunity_quality_by_symbol: Mapping[str, Any]
    opportunity_quality_gate: Mapping[str, Any]
    opportunity_top_quantile: float
    alpha_time_decay_enabled: bool
    alpha_stale_signal_sec: float
    live_execution_mode: bool
    burn_in_live_ready: bool
    burn_in_live_reason: str
    ramp_live_multiplier: float
    ramp_summary: Mapping[str, Any]
    liq_regime_enabled: bool
    thin_spread_bps: float
    thin_vol_mult: float
    primary_feed_derisk: Mapping[str, Any]
    quarantine_enabled: bool
    quarantine_manager: Any
    quarantine_apply_sleeve: bool
    quarantine_apply_symbol: bool
    quarantine_mode: str
    event_blackout_enabled: bool
    event_blackout_days: int
    event_blackout_cache: dict[str, bool]
    alpha_decay_deweight_enabled: bool
    alpha_decay_qty_step: float
    alpha_decay_qty_max_deweight: float
    capacity_throttle_enabled: bool
    capacity_spread_soft_bps: float
    capacity_spread_hard_bps: float
    capacity_volume_soft_participation: float
    capacity_volume_hard_participation: float
    capacity_min_scale: float
    slo_derisk_scale: float
    slo_derisk_details: Mapping[str, Any]
    execution_model_lineage: Mapping[str, Any]
    exec_engine: Any
    effective_policy: Any
    edge_realism_rank_factor_by_symbol: Mapping[str, Any]
    edge_realism_apply_to_approval_enabled: bool
    portfolio_current_gross: float
    sector_gross: Mapping[str, float]
    max_new_orders_per_cycle: int | None
    portfolio_optimizer_enabled: bool
    portfolio_optimizer: Any
    portfolio_optimizer_openings_only: bool
    portfolio_optimizer_market_data: Mapping[str, Any]
    portfolio_optimizer_context: Mapping[str, Any]
    ledger: Any
    rate_limiter: Any
    breakers: Any
    symbol_adaptive_profiles: Mapping[str, Mapping[str, Any]]
    uncertainty_capital_state: Mapping[str, Any]
    uncertainty_cycle_events: list[dict[str, Any]]
    penalty_overlap_coordination_enabled: bool
    penalty_overlap_weight_dampen: float
    penalty_overlap_min_scale_floor: float
    ineffective_gate_blocklist: set[str]
    gate_root_cause_func: Callable[[str], str]
    position_opened_at_func: Callable[..., Any]
    exit_policy_pressure_context_func: Callable[..., Mapping[str, Any]]
    is_near_event_func: Callable[..., bool]
    enforce_participation_cap_func: Callable[..., tuple[bool, float, str | None]]
    alpha_decay_entry_guard_func: Callable[..., Mapping[str, Any]]
    safe_float: Callable[[Any], float | None]
    resolve_uncertainty_capital_auto_controls_func: Callable[..., dict[str, Any]]
    clip_delta_to_symbol_notional_cap_func: Callable[..., tuple[int, dict[str, Any] | None]]
    clip_sell_qty_to_available_position_func: Callable[..., tuple[int, dict[str, Any] | None]]
    percentile_linear_func: Callable[[list[float], float], float | None]
    slippage_setting_bps_func: Callable[[], float]
    get_sector_func: Callable[[str], str | None]
    evaluate_execution_approval_func: Callable[..., Any]
    approve_execution_candidate_func: Callable[..., Any]
    gate_name_is_halt_noise_func: Callable[[str], bool]
    resolve_order_quote_basis_func: Callable[
        ...,
        tuple[
            str | None,
            float | None,
            float | None,
            float | None,
            float | None,
            datetime | None,
        ],
    ]
    portfolio_optimizer_allows_trade_func: Callable[..., tuple[bool, dict[str, Any]]]
    auth_forbidden_cooldown_remaining_seconds_func: Callable[..., float]
    safe_validate_pretrade_func: Callable[..., tuple[bool, str, dict[str, Any]]]
    extract_order_value_func: Callable[..., Any]
    extract_order_fill_timestamp_func: Callable[[Any], datetime | None]
    normalize_order_status_token_func: Callable[[Any], str]
    has_persistable_fill_func: Callable[..., bool]
    normalize_submitted_order_func: Callable[..., Any]
    record_successful_submission_func: Callable[..., None]
    build_order_metrics_and_tca_func: Callable[..., tuple[dict[str, Any], dict[str, Any] | None]]
    submit_order_func: Callable[..., Any]
    classify_exception_func: Callable[..., Any]
    handle_error_func: Callable[..., None]
    trigger_quarantine_func: Callable[..., None]
    cancel_all_open_orders_oms_func: Callable[[Any], Any]
    resolve_submit_none_reason_func: Callable[[Any], str]
    record_auth_forbidden_cooldown_func: Callable[..., None]
    get_regime_signal_profile_func: Callable[[], str]
    normalize_quote_source_token_func: Callable[[Any], str | None]
    resolve_quote_proxy_source_func: Callable[..., str | None]
    resolved_tca_path_func: Callable[[], Any]
    write_tca_record_func: Callable[[str, Mapping[str, Any]], None]
    session_bucket_from_ts_func: Callable[[datetime], str]
    compute_attribution_metrics_func: Callable[..., dict[str, Any]]
    record_decision_func: Callable[..., Any]
    prepare_symbol_prelude_func: Callable[..., Any]
    apply_symbol_adjustments_func: Callable[..., Any]
    prepare_symbol_approval_func: Callable[..., Any]
    prepare_submit_prelude_func: Callable[..., Any]
    execute_submission_func: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class NettingSymbolProcessResult:
    attempted_increment: int
    submitted_increment: int


def _gate_blocks(processor: NettingSymbolProcessor, candidate_gate: str) -> bool:
    gate_name = str(candidate_gate or "").strip().upper()
    if not gate_name:
        return True
    gate_root = str(processor.gate_root_cause_func(gate_name) or "").strip().upper()
    if gate_name in _NON_DISABLEABLE_GATE_NAMES or gate_root in _NON_DISABLEABLE_GATE_ROOTS:
        return True
    if gate_name in processor.ineffective_gate_blocklist:
        return False
    return gate_root not in processor.ineffective_gate_blocklist


def _short_openings_allowed(cfg: Any) -> bool:
    try:
        allowed = bool(
            resolve_launch_profile(
                str(getattr(cfg, "launch_profile", "") or "") or None
            ).shorts_allowed
        )
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        allowed = bool(get_env("TRADING__ALLOW_SHORTS", "0", cast=bool))
    for attr_name in ("shorts_allowed", "allow_short", "allow_shorts", "allow_short_selling"):
        if not hasattr(cfg, attr_name):
            continue
        value = getattr(cfg, attr_name, None)
        if value is not None:
            allowed = bool(allowed and bool(value))
    return bool(allowed)


def process_netting_symbol(
    *,
    processor: NettingSymbolProcessor,
    symbol: str,
    net_target: Any,
    orders_submitted: int,
) -> NettingSymbolProcessResult:
    liq_features = processor.latest_liquidity.get(
        symbol,
        LiquidityFeatures(rolling_volume=0.0, spread_bps=0.0, volatility_proxy=0.0),
    )
    liq_regime = (
        classify_liquidity_regime(
            liq_features,
            thin_spread_bps=processor.thin_spread_bps,
            thin_vol_mult=processor.thin_vol_mult,
        )
        if processor.liq_regime_enabled
        else LiquidityRegime.NORMAL
    )
    symbol_snapshot = dict(processor.decision_snapshot_template)
    if "liquidity_regime" in symbol_snapshot:
        symbol_snapshot["liquidity_regime"] = liq_regime.value
    if processor.state.halt_trading:
        reason = processor.state.halt_reason or "HALT_TRADING"
        processor.record_decision_func(
            symbol=symbol,
            bar_ts=net_target.bar_ts,
            sleeves=net_target.proposals,
            net_target=net_target,
            gates=[reason],
            config_snapshot=symbol_snapshot,
        )
        return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)

    price = processor.latest_price.get(symbol, 0.0)
    current_shares = int(processor.positions.get(symbol, 0) or 0)
    if price <= 0:
        net_target.reasons.append("BAD_DATA_CONTRACT")
        processor.record_decision_func(
            symbol=symbol,
            bar_ts=net_target.bar_ts,
            sleeves=net_target.proposals,
            net_target=net_target,
            gates=list(processor.skip_reasons.get(symbol, [])),
            config_snapshot=symbol_snapshot,
        )
        return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)

    net_target.target_shares = int(round(net_target.target_dollars / price))
    delta_shares = net_target.target_shares - current_shares
    initial_requested_delta_shares = int(delta_shares)
    gates: list[str] = []
    gates.extend(processor.skip_reasons.get(symbol, []))
    gates.extend(net_target.reasons)
    for proposal in net_target.proposals:
        if proposal.reason_code:
            gates.append(proposal.reason_code)
    if (
        int(current_shares) <= 0
        and int(net_target.target_shares) < 0
        and not _short_openings_allowed(processor.cfg)
    ):
        net_target.target_shares = 0
        net_target.target_dollars = 0.0
        delta_shares = -current_shares
        gates.append("LONG_ONLY_SHORT_SUPPRESSED")
        if int(current_shares) < 0:
            initial_requested_delta_shares = int(delta_shares)
        else:
            processor.record_decision_func(
                symbol=symbol,
                bar_ts=net_target.bar_ts,
                sleeves=net_target.proposals,
                net_target=net_target,
                gates=gates,
                config_snapshot=symbol_snapshot,
            )
            return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)
    symbol_prelude = processor.prepare_symbol_prelude_func(
        state=processor.state,
        symbol=symbol,
        now=processor.now,
        current_shares=int(current_shares),
        delta_shares=int(delta_shares),
        price=float(price),
        net_target=net_target,
        policy_disabled_sleeves=processor.policy_disabled_sleeves,
        policy_rollback_disabled_slices=processor.policy_rollback_disabled_slices,
        sleeve_configs_map=processor.sleeve_configs_map,
        candidate_expected_net_edge=processor.candidate_expected_net_edge,
        alpha_time_stop_enabled=processor.alpha_time_stop_enabled,
        alpha_time_stop_sec=processor.alpha_time_stop_sec,
        alpha_time_stop_max_expected_edge_bps=processor.alpha_time_stop_max_expected_edge_bps,
        opportunity_quality_enabled=processor.opportunity_quality_enabled,
        opportunity_allowed_symbols=processor.opportunity_allowed_symbols,
        opportunity_openings_only=processor.opportunity_openings_only,
        opportunity_quality_by_symbol=processor.opportunity_quality_by_symbol,
        opportunity_quality_gate=processor.opportunity_quality_gate,
        opportunity_top_quantile=processor.opportunity_top_quantile,
        alpha_time_decay_enabled=processor.alpha_time_decay_enabled,
        alpha_stale_signal_sec=processor.alpha_stale_signal_sec,
        live_execution_mode=processor.live_execution_mode,
        burn_in_live_ready=processor.burn_in_live_ready,
        burn_in_live_reason=processor.burn_in_live_reason,
        ramp_live_multiplier=processor.ramp_live_multiplier,
        ramp_summary=processor.ramp_summary,
        gates=gates,
        position_opened_at_func=processor.position_opened_at_func,
        exit_policy_pressure_context_func=processor.exit_policy_pressure_context_func,
    )
    gates.extend(symbol_prelude.gates_added)
    symbol_snapshot.update(symbol_prelude.snapshot_updates)
    delta_shares = int(symbol_prelude.delta_shares)
    net_target.target_shares = int(symbol_prelude.target_shares)
    net_target.target_dollars = float(symbol_prelude.target_dollars)
    if symbol_prelude.blocked_reason:
        processor.record_decision_func(
            symbol=symbol,
            bar_ts=net_target.bar_ts,
            sleeves=net_target.proposals,
            net_target=net_target,
            gates=gates,
            config_snapshot=symbol_snapshot,
        )
        return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)

    feed_derisk_scale = 1.0
    if delta_shares != 0 and bool(processor.primary_feed_derisk.get("triggered", False)):
        post_trade_shares = current_shares + delta_shares
        reducing_exposure = abs(post_trade_shares) < abs(current_shares)
        if bool(processor.primary_feed_derisk.get("block", False)) and not reducing_exposure:
            gates.append("DERISK_PRIMARY_FEED_BLOCK")
            symbol_snapshot["primary_feed_derisk"] = dict(processor.primary_feed_derisk)
            processor.record_decision_func(
                symbol=symbol,
                bar_ts=net_target.bar_ts,
                sleeves=net_target.proposals,
                net_target=net_target,
                gates=gates,
                config_snapshot=symbol_snapshot,
            )
            return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)
        if not reducing_exposure:
            feed_derisk_scale = float(processor.primary_feed_derisk.get("scale", 1.0) or 1.0)
            feed_derisk_scale = max(0.05, min(feed_derisk_scale, 1.0))
            if feed_derisk_scale < 1.0:
                gates.append("DERISK_PRIMARY_FEED_SCALE")
                symbol_snapshot["primary_feed_derisk"] = dict(processor.primary_feed_derisk)

    if processor.quarantine_enabled and processor.quarantine_manager is not None:
        reason_sleeve = None
        if processor.quarantine_apply_sleeve:
            for proposal in net_target.proposals:
                q_active, q_reason = processor.quarantine_manager.is_quarantined(
                    sleeve=proposal.sleeve,
                    now=net_target.bar_ts,
                )
                if q_active:
                    reason_sleeve = q_reason or "SLEEVE_QUARANTINED"
                    break
        q_symbol = False
        reason_symbol = None
        if processor.quarantine_apply_symbol:
            q_symbol, reason_symbol = processor.quarantine_manager.is_quarantined(
                symbol=symbol,
                now=net_target.bar_ts,
            )
        if reason_sleeve or q_symbol:
            quarantine_reason = reason_symbol or reason_sleeve or "SLEEVE_QUARANTINED"
            if quarantine_reason not in gates:
                gates.append(quarantine_reason)
            symbol_snapshot["quarantine_mode"] = processor.quarantine_mode
            if processor.quarantine_mode == "zero_targets":
                net_target.target_dollars = 0.0
                net_target.target_shares = 0
                delta_shares = -current_shares
            else:
                processor.record_decision_func(
                    symbol=symbol,
                    bar_ts=net_target.bar_ts,
                    sleeves=net_target.proposals,
                    net_target=net_target,
                    gates=gates,
                    config_snapshot=symbol_snapshot,
                )
                return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)

    lock = processor.state.stop_lock.get(symbol)
    if lock:
        lock_ts = lock.get("bar_ts")
        lock_dir = str(lock.get("direction", "")).lower()
        net_score = sum(p.score for p in net_target.proposals)
        net_conf = max((p.confidence for p in net_target.proposals), default=0.0)
        reentry_threshold = max((s.reentry_threshold for s in processor.sleeve_configs_map.values()), default=0.6)
        if lock_ts and isinstance(lock_ts, datetime) and net_target.bar_ts <= lock_ts:
            net_target.target_dollars = 0.0
            net_target.target_shares = 0
            delta_shares = -current_shares
            gates.append("STOP_LOCK")
        else:
            if abs(net_score) < reentry_threshold or net_conf < reentry_threshold:
                net_target.target_dollars = 0.0
                net_target.target_shares = 0
                delta_shares = -current_shares
                gates.append("STOP_LOCK")
            elif lock_dir:
                processor.state.stop_lock.pop(symbol, None)

    if delta_shares == 0:
        processor.record_decision_func(
            symbol=symbol,
            bar_ts=net_target.bar_ts,
            sleeves=net_target.proposals,
            net_target=net_target,
            gates=gates,
            config_snapshot=symbol_snapshot,
        )
        return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)

    if processor.kill_switch:
        gates.append("KILL_SWITCH_BLOCK")
        processor.record_decision_func(
            symbol=symbol,
            bar_ts=net_target.bar_ts,
            sleeves=net_target.proposals,
            net_target=net_target,
            gates=gates,
            config_snapshot=symbol_snapshot,
        )
        return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)

    event_risk_near = False
    if processor.event_blackout_enabled:
        cached_event_risk = processor.event_blackout_cache.get(symbol)
        if cached_event_risk is None:
            try:
                cached_event_risk = bool(
                    processor.is_near_event_func(symbol, days=processor.event_blackout_days)
                )
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                cached_event_risk = False
                processor.logger.debug(
                    "EVENT_BLACKOUT_CHECK_FAILED",
                    extra={"symbol": symbol, "days": processor.event_blackout_days},
                    exc_info=True,
                )
            processor.event_blackout_cache[symbol] = cached_event_risk
        event_risk_near = bool(cached_event_risk)
        symbol_snapshot["event_risk_near"] = event_risk_near
        if event_risk_near:
            gates.append("EVENT_RISK_BLACKOUT_BLOCK")
            processor.record_decision_func(
                symbol=symbol,
                bar_ts=net_target.bar_ts,
                sleeves=net_target.proposals,
                net_target=net_target,
                gates=gates,
                config_snapshot=symbol_snapshot,
            )
            return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)

    if processor.state.last_order_bar_ts.get(symbol) == net_target.bar_ts:
        gates.append("BAR_DEDUP")
        processor.record_decision_func(
            symbol=symbol,
            bar_ts=net_target.bar_ts,
            sleeves=net_target.proposals,
            net_target=net_target,
            gates=gates,
            config_snapshot=symbol_snapshot,
        )
        return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)

    if bool(get_env("AI_TRADING_PARTICIPATION_CAP_ENABLED", True, cast=bool)):
        participation_block_mode = str(
            get_env("AI_TRADING_PARTICIPATION_BLOCK_MODE", "block")
        ).strip().lower()
        allowed_participation, adjusted_qty, liq_reason = processor.enforce_participation_cap_func(
            order_qty=float(delta_shares),
            rolling_volume=liq_features.rolling_volume,
            max_participation_pct=float(
                get_env("AI_TRADING_MAX_PARTICIPATION_PCT", 0.015, cast=float)
            ),
            mode=participation_block_mode,
            scale_min=float(get_env("AI_TRADING_PARTICIPATION_SCALE_MIN", 0.25, cast=float)),
        )
        participation_block_bypassed = False
        if not allowed_participation:
            participation_gate = liq_reason or "LIQ_PARTICIPATION_BLOCK"
            if participation_block_mode == "block" or _gate_blocks(processor, str(participation_gate)):
                gates.append(str(participation_gate))
                processor.record_decision_func(
                    symbol=symbol,
                    bar_ts=net_target.bar_ts,
                    sleeves=net_target.proposals,
                    net_target=net_target,
                    gates=gates,
                    config_snapshot=symbol_snapshot,
                )
                return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)
            adjusted_qty_int = int(round(float(adjusted_qty)))
            if adjusted_qty_int == 0:
                adjusted_qty_int = 1 if delta_shares > 0 else -1
            delta_shares = adjusted_qty_int
            net_target.target_shares = current_shares + delta_shares
            net_target.target_dollars = net_target.target_shares * price
            gates.append("LIQ_PARTICIPATION_BLOCK_BYPASSED")
            participation_block_bypassed = True
            symbol_snapshot["gate_auto_disable"] = {
                "gate": str(participation_gate),
                "reason": "non_positive_marginal_contribution",
            }
        if not participation_block_bypassed:
            adjusted_qty_int = int(round(adjusted_qty))
            if adjusted_qty_int != delta_shares:
                delta_shares = adjusted_qty_int
                net_target.target_shares = current_shares + delta_shares
                net_target.target_dollars = net_target.target_shares * price
                if liq_reason:
                    gates.append(liq_reason)
        if liq_regime is LiquidityRegime.THIN and "LIQ_REGIME_THIN_SCALE" not in gates:
            gates.append("LIQ_REGIME_THIN_SCALE")

    if liq_regime is LiquidityRegime.THIN:
        thin_cost_mult = float(get_env("AI_TRADING_LIQ_THIN_COST_MULT", 1.3, cast=float))
        if thin_cost_mult > 1.0:
            scaled_qty = int(round(delta_shares / thin_cost_mult))
            if scaled_qty != 0 and scaled_qty != delta_shares:
                delta_shares = scaled_qty
                net_target.target_shares = current_shares + delta_shares
                net_target.target_dollars = net_target.target_shares * price
                gates.append("LIQ_THIN_COST_MULT_SCALE")
        thin_max_order_dollars = float(
            get_env("AI_TRADING_LIQ_THIN_MAX_ORDER_DOLLARS", 5000, cast=float)
        )
        if thin_max_order_dollars > 0 and abs(delta_shares) * price > thin_max_order_dollars:
            if _gate_blocks(processor, "LIQ_THIN_MAX_ORDER_BLOCK"):
                gates.append("LIQ_THIN_MAX_ORDER_BLOCK")
                processor.record_decision_func(
                    symbol=symbol,
                    bar_ts=net_target.bar_ts,
                    sleeves=net_target.proposals,
                    net_target=net_target,
                    gates=gates,
                    config_snapshot=symbol_snapshot,
                )
                return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)
            thin_cap_qty = int(
                math.copysign(
                    max(1, int(thin_max_order_dollars // max(price, 1e-9))),
                    delta_shares,
                )
            )
            delta_shares = thin_cap_qty
            net_target.target_shares = current_shares + delta_shares
            net_target.target_dollars = net_target.target_shares * price
            gates.append("LIQ_THIN_MAX_ORDER_BLOCK_BYPASSED")

    if current_shares == 0 and processor.alpha_decay_deweight_enabled:
        alpha_guard = processor.alpha_decay_entry_guard_func(processor.state, symbol, processor.now)
        if alpha_guard.get("blocked"):
            if _gate_blocks(processor, "ALPHA_DECAY_BLOCK"):
                gates.append("ALPHA_DECAY_BLOCK")
                processor.record_decision_func(
                    symbol=symbol,
                    bar_ts=net_target.bar_ts,
                    sleeves=net_target.proposals,
                    net_target=net_target,
                    gates=gates,
                    metrics={"alpha_decay": alpha_guard},
                    config_snapshot=symbol_snapshot,
                )
                return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)
            gates.append("ALPHA_DECAY_BLOCK_BYPASSED")
        trades_in_window = int(alpha_guard.get("trades_in_window", 0) or 0)
        start_trades = int(alpha_guard.get("start_trades", 0) or 0)
        over_start = max(0, trades_in_window - max(0, start_trades) + 1)
        if over_start > 0 and processor.alpha_decay_qty_step > 0:
            deweight = min(
                processor.alpha_decay_qty_max_deweight,
                over_start * processor.alpha_decay_qty_step,
            )
            multiplier = max(0.05, 1.0 - deweight)
            scaled_qty = int(round(float(delta_shares) * multiplier))
            if scaled_qty == 0:
                if _gate_blocks(processor, "ALPHA_DECAY_ZERO_QTY_BLOCK"):
                    gates.append("ALPHA_DECAY_ZERO_QTY_BLOCK")
                    processor.record_decision_func(
                        symbol=symbol,
                        bar_ts=net_target.bar_ts,
                        sleeves=net_target.proposals,
                        net_target=net_target,
                        gates=gates,
                        metrics={
                            "alpha_decay": {
                                **dict(alpha_guard),
                                "multiplier": multiplier,
                            }
                        },
                        config_snapshot=symbol_snapshot,
                    )
                    return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)
                scaled_qty = 1 if delta_shares > 0 else -1
            if scaled_qty != delta_shares:
                delta_shares = scaled_qty
                net_target.target_shares = current_shares + delta_shares
                net_target.target_dollars = net_target.target_shares * price
                gates.append("ALPHA_DECAY_DEWEIGHT")
                symbol_snapshot["alpha_decay"] = {
                    "trades_in_window": trades_in_window,
                    "start_trades": start_trades,
                    "multiplier": multiplier,
                }

    if processor.capacity_throttle_enabled and delta_shares != 0:
        capacity_scale = 1.0
        spread_bps_now = max(float(liq_features.spread_bps), 0.0)
        if (
            processor.capacity_spread_hard_bps > processor.capacity_spread_soft_bps
            and spread_bps_now > processor.capacity_spread_soft_bps
        ):
            if spread_bps_now >= processor.capacity_spread_hard_bps:
                spread_scale = processor.capacity_min_scale
            else:
                spread_progress = (spread_bps_now - processor.capacity_spread_soft_bps) / (
                    processor.capacity_spread_hard_bps - processor.capacity_spread_soft_bps
                )
                spread_scale = 1.0 - spread_progress * (1.0 - processor.capacity_min_scale)
            capacity_scale = min(capacity_scale, max(processor.capacity_min_scale, spread_scale))
        rolling_volume = max(float(liq_features.rolling_volume), 0.0)
        if (
            rolling_volume > 0
            and processor.capacity_volume_hard_participation
            > processor.capacity_volume_soft_participation
        ):
            participation = abs(float(delta_shares)) / rolling_volume
            if participation > processor.capacity_volume_soft_participation:
                if participation >= processor.capacity_volume_hard_participation:
                    volume_scale = processor.capacity_min_scale
                else:
                    participation_progress = (
                        participation - processor.capacity_volume_soft_participation
                    ) / (
                        processor.capacity_volume_hard_participation
                        - processor.capacity_volume_soft_participation
                    )
                    volume_scale = 1.0 - participation_progress * (
                        1.0 - processor.capacity_min_scale
                    )
                capacity_scale = min(
                    capacity_scale,
                    max(processor.capacity_min_scale, volume_scale),
                )
        if processor.slo_derisk_scale < 1.0:
            capacity_scale = min(capacity_scale, processor.slo_derisk_scale)
        if feed_derisk_scale < 1.0:
            capacity_scale = min(capacity_scale, feed_derisk_scale)
        if capacity_scale < 1.0:
            throttled_qty = int(round(float(delta_shares) * capacity_scale))
            if throttled_qty == 0:
                gates.append("CAPACITY_THROTTLE_BLOCK")
                processor.record_decision_func(
                    symbol=symbol,
                    bar_ts=net_target.bar_ts,
                    sleeves=net_target.proposals,
                    net_target=net_target,
                    gates=gates,
                    metrics={
                        "capacity_scale": capacity_scale,
                        "spread_bps": spread_bps_now,
                        "rolling_volume": rolling_volume,
                        "slo_derisk": dict(processor.slo_derisk_details),
                        "primary_feed_derisk": dict(processor.primary_feed_derisk),
                    },
                    config_snapshot=symbol_snapshot,
                )
                return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)
            if throttled_qty != delta_shares:
                delta_shares = throttled_qty
                net_target.target_shares = current_shares + delta_shares
                net_target.target_dollars = net_target.target_shares * price
                gates.append("CAPACITY_THROTTLE_SCALE")
                symbol_snapshot["capacity_throttle"] = {
                    "scale": capacity_scale,
                    "spread_bps": spread_bps_now,
                    "rolling_volume": rolling_volume,
                    "slo_derisk": dict(processor.slo_derisk_details),
                    "primary_feed_derisk": dict(processor.primary_feed_derisk),
                }

    expanding_exposure = abs(current_shares + delta_shares) > abs(current_shares)
    adjustment_result = processor.apply_symbol_adjustments_func(
        symbol=symbol,
        state=processor.state,
        cfg=processor.cfg,
        current_shares=int(current_shares),
        delta_shares=int(delta_shares),
        price=float(price),
        expanding_exposure=bool(expanding_exposure),
        initial_requested_delta_shares=int(initial_requested_delta_shares),
        symbol_adaptive_profiles=processor.symbol_adaptive_profiles,
        slo_derisk_details=processor.slo_derisk_details,
        primary_feed_derisk=processor.primary_feed_derisk,
        penalty_overlap_coordination_enabled=processor.penalty_overlap_coordination_enabled,
        penalty_overlap_weight_dampen=processor.penalty_overlap_weight_dampen,
        penalty_overlap_min_scale_floor=processor.penalty_overlap_min_scale_floor,
        uncertainty_capital_state=processor.uncertainty_capital_state,
        safe_float=processor.safe_float,
        resolve_uncertainty_capital_auto_controls_func=processor.resolve_uncertainty_capital_auto_controls_func,
        clip_delta_to_symbol_notional_cap_func=processor.clip_delta_to_symbol_notional_cap_func,
        logger=processor.logger,
    )
    if adjustment_result.uncertainty_event is not None:
        processor.uncertainty_cycle_events.append(dict(adjustment_result.uncertainty_event))
    delta_shares = int(adjustment_result.delta_shares)
    net_target.target_shares = int(adjustment_result.target_shares)
    net_target.target_dollars = float(adjustment_result.target_dollars)
    gates.extend(adjustment_result.gates_added)
    symbol_snapshot.update(adjustment_result.snapshot_updates)
    if adjustment_result.blocked_reason:
        processor.record_decision_func(
            symbol=symbol,
            bar_ts=net_target.bar_ts,
            sleeves=net_target.proposals,
            net_target=net_target,
            gates=gates,
            metrics=adjustment_result.blocked_metrics,
            config_snapshot=symbol_snapshot,
        )
        return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)

    approval_result = processor.prepare_symbol_approval_func(
        state=processor.state,
        symbol=symbol,
        now=processor.now,
        current_shares=int(current_shares),
        delta_shares=int(delta_shares),
        price=float(price),
        net_target=net_target,
        liq_features=liq_features,
        liq_regime=liq_regime,
        exec_engine=processor.exec_engine,
        effective_policy=processor.effective_policy,
        candidate_expected_net_edge=processor.candidate_expected_net_edge,
        edge_realism_rank_factor_by_symbol=processor.edge_realism_rank_factor_by_symbol,
        edge_realism_apply_to_approval_enabled=processor.edge_realism_apply_to_approval_enabled,
        alpha_decay_deweight_enabled=processor.alpha_decay_deweight_enabled,
        alpha_decay_qty_step=processor.alpha_decay_qty_step,
        alpha_decay_qty_max_deweight=processor.alpha_decay_qty_max_deweight,
        capacity_throttle_enabled=processor.capacity_throttle_enabled,
        capacity_spread_soft_bps=processor.capacity_spread_soft_bps,
        capacity_spread_hard_bps=processor.capacity_spread_hard_bps,
        capacity_volume_soft_participation=processor.capacity_volume_soft_participation,
        capacity_volume_hard_participation=processor.capacity_volume_hard_participation,
        capacity_min_scale=processor.capacity_min_scale,
        slo_derisk_scale=processor.slo_derisk_scale,
        slo_derisk_details=dict(processor.slo_derisk_details),
        primary_feed_derisk=dict(processor.primary_feed_derisk),
        feed_derisk_scale=feed_derisk_scale,
        portfolio_current_gross=processor.portfolio_current_gross,
        sector_gross=processor.sector_gross,
        max_new_orders_per_cycle=processor.max_new_orders_per_cycle,
        orders_submitted=orders_submitted,
        gates=gates,
        symbol_snapshot=symbol_snapshot,
        gate_blocks_func=lambda gate: _gate_blocks(processor, gate),
        clip_sell_qty_to_available_position_func=processor.clip_sell_qty_to_available_position_func,
        percentile_linear_func=processor.percentile_linear_func,
        slippage_setting_bps_func=processor.slippage_setting_bps_func,
        safe_float_func=processor.safe_float,
        get_sector_func=processor.get_sector_func,
        alpha_decay_entry_guard_func=processor.alpha_decay_entry_guard_func,
        evaluate_execution_approval_func=processor.evaluate_execution_approval_func,
        approve_execution_candidate_func=processor.approve_execution_candidate_func,
    )
    gates.extend(approval_result.gates_added)
    symbol_snapshot.update(approval_result.snapshot_updates)
    delta_shares = int(approval_result.delta_shares)
    net_target.target_shares = int(approval_result.target_shares)
    net_target.target_dollars = float(approval_result.target_dollars)
    if approval_result.blocked_reason:
        processor.record_decision_func(
            symbol=symbol,
            bar_ts=net_target.bar_ts,
            sleeves=net_target.proposals,
            net_target=net_target,
            gates=gates,
            metrics=approval_result.blocked_metrics,
            config_snapshot=symbol_snapshot,
        )
        return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)

    approval_context = approval_result.approval_context
    approval = approval_result.approval
    assert approval_context is not None
    assert approval is not None
    side = approval_result.side
    opening_trade = bool(approval_result.opening_trade)

    submit_prelude = processor.prepare_submit_prelude_func(
        state=processor.state,
        runtime=processor.runtime,
        cfg=processor.cfg,
        now=processor.now,
        symbol=symbol,
        side=side,
        price=float(price),
        delta_shares=int(delta_shares),
        current_shares=float(current_shares),
        bar_ts=net_target.bar_ts,
        liq_features=liq_features,
        liq_regime=liq_regime,
        net_target=net_target,
        slo_derisk_details=dict(processor.slo_derisk_details),
        symbol_snapshot=symbol_snapshot,
        execution_model_lineage=dict(processor.execution_model_lineage),
        event_risk_near=event_risk_near,
        opening_trade=bool(opening_trade),
        portfolio_optimizer_enabled=bool(processor.portfolio_optimizer_enabled),
        portfolio_optimizer=processor.portfolio_optimizer,
        portfolio_optimizer_openings_only=bool(processor.portfolio_optimizer_openings_only),
        positions=processor.positions,
        portfolio_optimizer_market_data=dict(processor.portfolio_optimizer_market_data),
        portfolio_optimizer_context=dict(processor.portfolio_optimizer_context),
        ledger=processor.ledger,
        rate_limiter=processor.rate_limiter,
        breakers=processor.breakers,
        kill_switch_active=bool(processor.kill_switch),
        gate_name_is_halt_noise_func=processor.gate_name_is_halt_noise_func,
        resolve_order_quote_basis_func=processor.resolve_order_quote_basis_func,
        portfolio_optimizer_allows_trade_func=processor.portfolio_optimizer_allows_trade_func,
        auth_forbidden_cooldown_remaining_seconds_func=processor.auth_forbidden_cooldown_remaining_seconds_func,
        safe_validate_pretrade_func=processor.safe_validate_pretrade_func,
        get_sector_func=processor.get_sector_func,
    )
    gates.extend(submit_prelude.gates_added)
    symbol_snapshot.update(submit_prelude.snapshot_updates)
    if submit_prelude.blocked_reason:
        processor.record_decision_func(
            symbol=symbol,
            bar_ts=net_target.bar_ts,
            sleeves=net_target.proposals,
            net_target=net_target,
            gates=gates,
            metrics=submit_prelude.blocked_metrics,
            config_snapshot=symbol_snapshot,
            order_intent=submit_prelude.blocked_order_intent,
        )
        return NettingSymbolProcessResult(attempted_increment=0, submitted_increment=0)

    execution_intent_context = submit_prelude.execution_intent_context
    assert execution_intent_context is not None
    submit_quote_source = submit_prelude.submit_quote_source
    submit_bid_at_arrival = submit_prelude.submit_bid_at_arrival
    submit_ask_at_arrival = submit_prelude.submit_ask_at_arrival
    submit_mid_at_arrival = submit_prelude.submit_mid_at_arrival
    submit_arrival_price = submit_prelude.submit_arrival_price
    client_order_id = execution_intent_context.client_order_id
    intent = execution_intent_context.pretrade_intent
    order_lineage_metadata = dict(execution_intent_context.order_lineage_metadata)
    order_annotations = dict(execution_intent_context.order_annotations)
    model_id_for_order = str(processor.execution_model_lineage.get("model_id") or "").strip()
    model_version_for_order = str(
        processor.execution_model_lineage.get("model_version") or ""
    ).strip()
    dataset_hash_for_order = str(
        processor.execution_model_lineage.get("dataset_hash") or ""
    ).strip()
    feature_version_for_order = str(
        processor.execution_model_lineage.get("feature_version") or ""
    ).strip()
    model_artifact_hash_for_order = str(
        processor.execution_model_lineage.get("model_artifact_hash") or ""
    ).strip()
    config_snapshot_hash_for_order = str(
        symbol_snapshot.get("config_snapshot_hash") or ""
    ).strip()
    policy_hash_for_order = str(symbol_snapshot.get("effective_policy_hash") or "").strip()
    decision_trace_id_for_order = execution_intent_context.decision_trace_id

    submit_result = processor.execute_submission_func(
        runtime=processor.runtime,
        state=processor.state,
        symbol=symbol,
        side=side,
        price=float(price),
        delta_shares=int(delta_shares),
        now=processor.now,
        net_target=net_target,
        approval=approval,
        intent=intent,
        client_order_id=client_order_id,
        decision_trace_id_for_order=decision_trace_id_for_order,
        model_id_for_order=model_id_for_order,
        model_version_for_order=model_version_for_order,
        config_snapshot_hash_for_order=config_snapshot_hash_for_order,
        dataset_hash_for_order=dataset_hash_for_order,
        feature_version_for_order=feature_version_for_order,
        model_artifact_hash_for_order=model_artifact_hash_for_order,
        policy_hash_for_order=policy_hash_for_order,
        order_annotations=order_annotations,
        order_lineage_metadata=order_lineage_metadata,
        submit_arrival_price=submit_arrival_price,
        submit_bid_at_arrival=submit_bid_at_arrival,
        submit_ask_at_arrival=submit_ask_at_arrival,
        submit_mid_at_arrival=submit_mid_at_arrival,
        submit_quote_source=submit_quote_source,
        candidate_expected_net_edge=processor.candidate_expected_net_edge,
        candidate_expected_capture=processor.candidate_expected_capture,
        ledger=processor.ledger,
        quarantine_enabled=processor.quarantine_enabled,
        quarantine_manager=processor.quarantine_manager,
        extract_order_value_func=processor.extract_order_value_func,
        extract_order_fill_timestamp_func=processor.extract_order_fill_timestamp_func,
        normalize_order_status_token_func=processor.normalize_order_status_token_func,
        safe_float=processor.safe_float,
        has_persistable_fill_func=processor.has_persistable_fill_func,
        normalize_submitted_order_func=processor.normalize_submitted_order_func,
        record_successful_submission_func=processor.record_successful_submission_func,
        build_order_metrics_and_tca_func=processor.build_order_metrics_and_tca_func,
        submit_order_func=processor.submit_order_func,
        classify_exception_func=processor.classify_exception_func,
        handle_error_func=processor.handle_error_func,
        trigger_quarantine_func=processor.trigger_quarantine_func,
        cancel_all_open_orders_oms_func=processor.cancel_all_open_orders_oms_func,
        resolve_submit_none_reason_func=processor.resolve_submit_none_reason_func,
        record_auth_forbidden_cooldown_func=processor.record_auth_forbidden_cooldown_func,
        get_regime_signal_profile_func=processor.get_regime_signal_profile_func,
        normalize_quote_source_token_func=processor.normalize_quote_source_token_func,
        resolve_quote_proxy_source_func=processor.resolve_quote_proxy_source_func,
        resolved_tca_path_func=processor.resolved_tca_path_func,
        write_tca_record_func=processor.write_tca_record_func,
        session_bucket_from_ts_func=processor.session_bucket_from_ts_func,
        compute_attribution_metrics_func=processor.compute_attribution_metrics_func,
        logger=processor.logger,
        breakers=processor.breakers,
    )
    gates.extend(submit_result.gates_added)
    if submit_result.status != "submitted":
        processor.record_decision_func(
            symbol=symbol,
            bar_ts=net_target.bar_ts,
            sleeves=net_target.proposals,
            net_target=net_target,
            gates=gates,
            metrics=submit_result.metrics,
            config_snapshot=symbol_snapshot,
            order_intent=submit_result.order_intent_contract,
        )
        return NettingSymbolProcessResult(
            attempted_increment=int(submit_result.attempted_increment),
            submitted_increment=int(submit_result.submitted_increment),
        )

    processor.record_decision_func(
        symbol=symbol,
        bar_ts=net_target.bar_ts,
        sleeves=net_target.proposals,
        net_target=net_target,
        gates=gates,
        order=submit_result.order_payload,
        metrics=submit_result.metrics,
        config_snapshot=symbol_snapshot,
        tca=submit_result.tca_record,
        decision_trace_id=submit_result.decision_trace_id,
        order_intent=submit_result.order_intent_contract,
    )
    return NettingSymbolProcessResult(
        attempted_increment=int(submit_result.attempted_increment),
        submitted_increment=int(submit_result.submitted_increment),
    )


__all__ = [
    "NettingSymbolProcessResult",
    "NettingSymbolProcessor",
    "process_netting_symbol",
]
