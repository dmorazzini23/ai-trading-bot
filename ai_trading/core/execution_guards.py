"""Helpers for execution approval and pre-submit guard context."""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from ai_trading.config.management import get_env
from ai_trading.policy.compiler import (
    ExecutionApproval,
    ExecutionCandidate,
    SafetyTier,
    approve_execution_candidate,
)


@dataclass(frozen=True, slots=True)
class ExecutionApprovalContext:
    approval: ExecutionApproval
    adjusted_delta_shares: int
    adjusted_side: str
    pacing_headroom: int
    stale_orders_present: bool
    portfolio_post_gross: float
    factor_post_ratio: float
    sector_name: str


def evaluate_execution_approval(
    *,
    effective_policy: Any,
    symbol: str,
    side: str,
    delta_shares: int,
    current_shares: float,
    price: float,
    expected_edge_total: float,
    expected_cost_total: float,
    proposals: Iterable[Any],
    spread_bps: float,
    rolling_volume: float,
    pending_oldest_age_sec: float,
    calibration_ok: bool,
    reject_rate_pct: float,
    portfolio_current_gross: float,
    sector_gross: Mapping[str, float],
    sector_name: str,
    max_new_orders_per_cycle: int | None,
    orders_submitted: int,
    engine_cycle_new_orders_submitted: Any,
    safety_tier_raw: str,
    approval_func: Any = approve_execution_candidate,
) -> ExecutionApprovalContext:
    pacing_headroom = 999999
    if max_new_orders_per_cycle is not None:
        pacing_used = max(0, int(orders_submitted))
        try:
            engine_submits = int(engine_cycle_new_orders_submitted)
        except (TypeError, ValueError):
            engine_submits = int(orders_submitted)
        pacing_used = max(pacing_used, max(engine_submits, 0))
        pacing_headroom = max(0, int(max_new_orders_per_cycle) - pacing_used)

    stale_orders_present = float(pending_oldest_age_sec or 0.0) > 0.0
    current_notional = abs(float(current_shares) * float(price))
    post_notional = abs(float(current_shares + delta_shares) * float(price))
    portfolio_post_gross = max(
        0.0,
        float(portfolio_current_gross) - current_notional + post_notional,
    )
    sector_current_gross = float(sector_gross.get(str(sector_name).upper(), 0.0) or 0.0)
    sector_post_gross = max(0.0, sector_current_gross - current_notional + post_notional)
    factor_post_ratio = sector_post_gross / max(portfolio_post_gross, 1.0)
    try:
        safety_tier = SafetyTier(str(safety_tier_raw or SafetyTier.NORMAL.value))
    except ValueError:
        safety_tier = SafetyTier.NORMAL

    approval = approval_func(
        effective_policy,
        ExecutionCandidate(
            symbol=symbol,
            side=side,
            proposed_delta_shares=int(delta_shares),
            current_shares=int(current_shares),
            price=float(price),
            expected_edge_bps=float(expected_edge_total),
            expected_cost_bps=float(expected_cost_total),
            confidence=max((float(getattr(p, "confidence", 0.0) or 0.0) for p in proposals), default=0.0),
            spread_bps=float(spread_bps),
            rolling_volume=float(rolling_volume),
            pending_oldest_age_sec=float(pending_oldest_age_sec or 0.0),
            pacing_headroom=int(pacing_headroom),
            stale_orders_present=bool(stale_orders_present),
            calibration_ok=bool(calibration_ok),
            portfolio_post_gross_dollars=float(portfolio_post_gross),
            sleeve_post_notional_dollars=max(
                (abs(float(getattr(p, "target_dollars", 0.0) or 0.0)) for p in proposals),
                default=0.0,
            ),
            factor_post_ratio=float(factor_post_ratio),
            reject_rate_pct=float(reject_rate_pct or 0.0),
            safety_tier=safety_tier,
        ),
    )
    adjusted_delta_shares = int(approval.adjusted_delta_shares)
    adjusted_target_shares = float(current_shares) + float(adjusted_delta_shares)
    if adjusted_delta_shares > 0:
        adjusted_side = "buy"
    elif float(current_shares) > 0.0:
        adjusted_side = "sell"
    elif adjusted_target_shares < 0:
        adjusted_side = "sell_short"
    else:
        adjusted_side = "sell"
    return ExecutionApprovalContext(
        approval=approval,
        adjusted_delta_shares=adjusted_delta_shares,
        adjusted_side=adjusted_side,
        pacing_headroom=int(pacing_headroom),
        stale_orders_present=bool(stale_orders_present),
        portfolio_post_gross=float(portfolio_post_gross),
        factor_post_ratio=float(factor_post_ratio),
        sector_name=str(sector_name).upper(),
    )


def build_portfolio_optimizer_positions(
    positions: Mapping[Any, Any],
    *,
    symbol: str,
    current_shares: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for sym, pos in positions.items():
        try:
            parsed_pos = float(pos)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(parsed_pos):
            continue
        out[str(sym)] = float(parsed_pos)
    out[str(symbol)] = float(current_shares)
    return out


def build_pretrade_validation_cfg(
    cfg: Any,
    *,
    thin_liquidity: bool,
) -> tuple[Any, float | None, float | None]:
    if not thin_liquidity:
        return cfg, None, None
    collar_mult = float(get_env("AI_TRADING_LIQ_THIN_COLLAR_MULT", 0.8, cast=float))
    if collar_mult <= 0:
        return cfg, None, None
    raw_collar = getattr(cfg, "price_collar_pct", None)
    if raw_collar is None:
        raw_collar = get_env("PRICE_COLLAR_PCT", 0.03, cast=float)
    try:
        base_collar_pct = float(raw_collar)
    except (TypeError, ValueError):
        base_collar_pct = float(get_env("PRICE_COLLAR_PCT", 0.03, cast=float))
    effective_collar_pct = max(0.0, base_collar_pct * collar_mult)
    pretrade_cfg = SimpleNamespace(
        max_order_dollars=getattr(cfg, "max_order_dollars", None),
        max_order_shares=getattr(cfg, "max_order_shares", None),
        max_symbol_notional=getattr(cfg, "max_symbol_notional", None),
        max_gross_notional=getattr(cfg, "max_gross_notional", None),
        max_sector_notional=getattr(cfg, "max_sector_notional", None),
        max_factor_exposure=getattr(cfg, "max_factor_exposure", None),
        intraday_var_limit=getattr(cfg, "intraday_var_limit", None),
        intraday_cvar_limit=getattr(cfg, "intraday_cvar_limit", None),
        intraday_drawdown_limit=getattr(cfg, "intraday_drawdown_limit", None),
        daily_loss_limit_pct=getattr(cfg, "daily_loss_limit_pct", None),
        daily_loss_limit_abs=getattr(cfg, "daily_loss_limit_abs", None),
        quote_max_age_ms=getattr(cfg, "quote_max_age_ms", None),
        min_quote_freshness_ms=getattr(cfg, "min_quote_freshness_ms", None),
        rth_only=getattr(cfg, "rth_only", None),
        allow_extended=getattr(cfg, "allow_extended", None),
        kill_switch=getattr(cfg, "kill_switch", None),
        price_collar_pct=effective_collar_pct,
    )
    return pretrade_cfg, effective_collar_pct, collar_mult
