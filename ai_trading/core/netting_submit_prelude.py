"""Pre-submit broker gate orchestration for the live netting cycle."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from ai_trading.config.management import get_env
from ai_trading.core.execution_guards import build_portfolio_optimizer_positions, build_pretrade_validation_cfg
from ai_trading.core.execution_intent import ExecutionIntentContext, build_execution_intent_context


@dataclass(frozen=True, slots=True)
class NettingSubmitPreludeResult:
    execution_intent_context: ExecutionIntentContext | None
    submit_quote_source: str | None
    submit_bid_at_arrival: float | None
    submit_ask_at_arrival: float | None
    submit_mid_at_arrival: float | None
    submit_arrival_price: float | None
    gates_added: tuple[str, ...]
    snapshot_updates: dict[str, Any]
    blocked_reason: str | None
    blocked_metrics: dict[str, Any] | None
    blocked_order_intent: Any | None


def prepare_netting_submit_prelude(
    *,
    state: Any,
    runtime: Any,
    cfg: Any,
    now: datetime,
    symbol: str,
    side: str,
    price: float,
    delta_shares: int,
    current_shares: float,
    bar_ts: datetime,
    liq_features: Any,
    liq_regime: Any,
    net_target: Any,
    slo_derisk_details: dict[str, Any],
    symbol_snapshot: dict[str, Any],
    execution_model_lineage: dict[str, Any],
    event_risk_near: bool,
    opening_trade: bool,
    portfolio_optimizer_enabled: bool,
    portfolio_optimizer: Any,
    portfolio_optimizer_openings_only: bool,
    positions: Any,
    portfolio_optimizer_market_data: dict[str, Any],
    portfolio_optimizer_context: dict[str, Any],
    ledger: Any,
    rate_limiter: Any,
    breakers: Any,
    gate_name_is_halt_noise_func: Callable[[str], bool],
    resolve_order_quote_basis_func: Callable[..., tuple[str | None, float | None, float | None, float | None, float | None]],
    portfolio_optimizer_allows_trade_func: Callable[..., tuple[bool, dict[str, Any]]],
    auth_forbidden_cooldown_remaining_seconds_func: Callable[..., float],
    safe_validate_pretrade_func: Callable[..., tuple[bool, str, dict[str, Any]]],
    get_sector_func: Callable[[str], str | None],
) -> NettingSubmitPreludeResult:
    snapshot_updates: dict[str, Any] = {}
    gates_added: list[str] = []

    if (
        portfolio_optimizer_enabled
        and portfolio_optimizer is not None
        and (not bool(portfolio_optimizer_openings_only) or bool(opening_trade))
    ):
        current_positions_for_optimizer = build_portfolio_optimizer_positions(
            positions,
            symbol=symbol,
            current_shares=float(current_shares),
        )
        proposed_position = float(current_shares + delta_shares)
        opt_allowed, opt_context = portfolio_optimizer_allows_trade_func(
            optimizer=portfolio_optimizer,
            symbol=symbol,
            proposed_position=float(proposed_position),
            current_positions=current_positions_for_optimizer,
            market_data=portfolio_optimizer_market_data,
        )
        snapshot_updates["portfolio_optimizer"] = dict(portfolio_optimizer_context | opt_context)
        if not opt_allowed:
            decision_token = str(opt_context.get("decision") or "reject").strip().lower()
            gate_token = "".join(ch if ch.isalnum() else "_" for ch in decision_token.upper()).strip("_") or "REJECT"
            blocked_reason = f"PORTFOLIO_OPTIMIZER_{gate_token}"
            gates_added.append(blocked_reason)
            return NettingSubmitPreludeResult(
                execution_intent_context=None,
                submit_quote_source=None,
                submit_bid_at_arrival=None,
                submit_ask_at_arrival=None,
                submit_mid_at_arrival=None,
                submit_arrival_price=None,
                gates_added=tuple(gates_added),
                snapshot_updates=snapshot_updates,
                blocked_reason=blocked_reason,
                blocked_metrics={"portfolio_optimizer": dict(opt_context)},
                blocked_order_intent=None,
            )

    auth_forbidden_retry_after = auth_forbidden_cooldown_remaining_seconds_func(
        state,
        symbol=symbol,
        side=side,
        now=now,
    )
    if auth_forbidden_retry_after > 0.0:
        blocked_reason = "AUTH_BROKER_HALT_FORBIDDEN_COOLDOWN"
        gates_added.append(blocked_reason)
        return NettingSubmitPreludeResult(
            execution_intent_context=None,
            submit_quote_source=None,
            submit_bid_at_arrival=None,
            submit_ask_at_arrival=None,
            submit_mid_at_arrival=None,
            submit_arrival_price=None,
            gates_added=tuple(gates_added),
            snapshot_updates=snapshot_updates,
            blocked_reason=blocked_reason,
            blocked_metrics={"auth_forbidden_retry_after_sec": round(auth_forbidden_retry_after, 3)},
            blocked_order_intent=None,
        )

    initial_intent_context = build_execution_intent_context(
        salt=str(getattr(cfg, "seed", "seed")),
        symbol=symbol,
        side=side,
        delta_shares=int(delta_shares),
        price=float(price),
        bar_ts=bar_ts,
        spread_bps=float(liq_features.spread_bps),
        liquidity_bucket=liq_regime.value.upper(),
        quote_quality_ok=not bool(state.halt_trading),
        sector=get_sector_func(symbol),
        event_risk=event_risk_near,
        slo_derisk_details={**dict(slo_derisk_details), "rolling_volume": float(liq_features.rolling_volume)},
        config_snapshot=symbol_snapshot,
        execution_model_lineage=execution_model_lineage,
        submit_quote_source=None,
        submit_bid_at_arrival=None,
        submit_ask_at_arrival=None,
        submit_mid_at_arrival=None,
    )
    client_order_id = initial_intent_context.client_order_id
    if not breakers.allow("broker_submit"):
        reason = breakers.open_reason("broker_submit") or "CIRCUIT_OPEN_broker_submit"
        state.halt_reason = reason
        gates_added.append(reason)
        return NettingSubmitPreludeResult(
            execution_intent_context=None,
            submit_quote_source=None,
            submit_bid_at_arrival=None,
            submit_ask_at_arrival=None,
            submit_mid_at_arrival=None,
            submit_arrival_price=None,
            gates_added=tuple(gates_added),
            snapshot_updates=snapshot_updates,
            blocked_reason=reason,
            blocked_metrics=None,
            blocked_order_intent=None,
        )
    if ledger is not None and ledger.seen_client_order_id(client_order_id):
        blocked_reason = "BAR_DEDUP"
        gates_added.append(blocked_reason)
        return NettingSubmitPreludeResult(
            execution_intent_context=None,
            submit_quote_source=None,
            submit_bid_at_arrival=None,
            submit_ask_at_arrival=None,
            submit_mid_at_arrival=None,
            submit_arrival_price=None,
            gates_added=tuple(gates_added),
            snapshot_updates=snapshot_updates,
            blocked_reason=blocked_reason,
            blocked_metrics=None,
            blocked_order_intent=None,
        )

    intent = initial_intent_context.pretrade_intent
    pretrade_cfg, effective_collar_pct, collar_mult = build_pretrade_validation_cfg(
        cfg,
        thin_liquidity=liq_regime.value == "thin" or str(liq_regime).endswith(".THIN"),
    )
    if effective_collar_pct is not None and collar_mult is not None:
        snapshot_updates["liquidity_collar_multiplier"] = collar_mult
        snapshot_updates["price_collar_pct_effective"] = effective_collar_pct
    allowed, pretrade_reason, pretrade_details = safe_validate_pretrade_func(
        intent,
        cfg=pretrade_cfg,
        ledger=ledger,
        rate_limiter=rate_limiter,
    )
    if not allowed:
        if effective_collar_pct is not None:
            pretrade_details.setdefault("price_collar_pct", effective_collar_pct)
        gates_added.append(pretrade_reason)
        return NettingSubmitPreludeResult(
            execution_intent_context=None,
            submit_quote_source=None,
            submit_bid_at_arrival=None,
            submit_ask_at_arrival=None,
            submit_mid_at_arrival=None,
            submit_arrival_price=None,
            gates_added=tuple(gates_added),
            snapshot_updates=snapshot_updates,
            blocked_reason=pretrade_reason,
            blocked_metrics={"pretrade": pretrade_details},
            blocked_order_intent=intent.to_contract(),
        )

    try:
        require_realtime_nbbo = bool(getattr(cfg, "execution_require_realtime_nbbo", True))
    except Exception:
        require_realtime_nbbo = True
    enforce_opening_nbbo = bool(get_env("AI_TRADING_ENFORCE_NBBO_FOR_OPENINGS", True, cast=bool))
    (
        submit_quote_source,
        submit_bid_at_arrival,
        submit_ask_at_arrival,
        submit_mid_at_arrival,
        submit_arrival_price,
    ) = resolve_order_quote_basis_func(
        runtime,
        symbol=symbol,
        side=side,
        fallback_price=price,
    )
    if bool(opening_trade) and require_realtime_nbbo and enforce_opening_nbbo and submit_mid_at_arrival is None:
        blocked_reason = "NBBO_REQUIRED_OPENING_SKIP"
        gates_added.append(blocked_reason)
        return NettingSubmitPreludeResult(
            execution_intent_context=None,
            submit_quote_source=submit_quote_source,
            submit_bid_at_arrival=submit_bid_at_arrival,
            submit_ask_at_arrival=submit_ask_at_arrival,
            submit_mid_at_arrival=submit_mid_at_arrival,
            submit_arrival_price=submit_arrival_price,
            gates_added=tuple(gates_added),
            snapshot_updates=snapshot_updates,
            blocked_reason=blocked_reason,
            blocked_metrics={
                "nbbo_guard": {
                    "required": True,
                    "opening_trade": True,
                    "price_source": submit_quote_source,
                    "bid_at_arrival": submit_bid_at_arrival,
                    "ask_at_arrival": submit_ask_at_arrival,
                }
            },
            blocked_order_intent=None,
        )

    final_intent_context = build_execution_intent_context(
        salt=str(getattr(cfg, "seed", "seed")),
        symbol=symbol,
        side=side,
        delta_shares=int(delta_shares),
        price=float(price),
        bar_ts=bar_ts,
        spread_bps=float(liq_features.spread_bps),
        liquidity_bucket=liq_regime.value.upper(),
        quote_quality_ok=not bool(state.halt_trading),
        sector=get_sector_func(symbol),
        event_risk=event_risk_near,
        slo_derisk_details={**dict(slo_derisk_details), "rolling_volume": float(liq_features.rolling_volume)},
        config_snapshot=symbol_snapshot | snapshot_updates,
        execution_model_lineage=execution_model_lineage,
        submit_quote_source=submit_quote_source,
        submit_bid_at_arrival=submit_bid_at_arrival,
        submit_ask_at_arrival=submit_ask_at_arrival,
        submit_mid_at_arrival=submit_mid_at_arrival,
    )
    return NettingSubmitPreludeResult(
        execution_intent_context=final_intent_context,
        submit_quote_source=submit_quote_source,
        submit_bid_at_arrival=submit_bid_at_arrival,
        submit_ask_at_arrival=submit_ask_at_arrival,
        submit_mid_at_arrival=submit_mid_at_arrival,
        submit_arrival_price=submit_arrival_price,
        gates_added=tuple(gates_added),
        snapshot_updates=snapshot_updates,
        blocked_reason=None,
        blocked_metrics=None,
        blocked_order_intent=None,
    )
