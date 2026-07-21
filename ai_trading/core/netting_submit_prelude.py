"""Pre-submit broker gate orchestration for the live netting cycle."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from dataclasses import dataclass, replace
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


def _fail_closed_on_optimizer_error(*, cfg: Any, opening_trade: bool) -> bool:
    if bool(opening_trade):
        return True
    execution_mode = str(getattr(cfg, "execution_mode", "") or "").strip().lower()
    launch_profile = str(
        getattr(
            cfg,
            "launch_profile",
            get_env("AI_TRADING_LAUNCH_PROFILE", "", cast=str),
        )
        or ""
    ).strip().lower()
    return execution_mode == "live" or launch_profile == "live_canary" or launch_profile.startswith("live_")


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
    correlation_id: str,
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
    kill_switch_active: bool,
    gate_name_is_halt_noise_func: Callable[[str], bool],
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
    ],
    portfolio_optimizer_allows_trade_func: Callable[..., tuple[bool, dict[str, Any]]],
    auth_forbidden_cooldown_remaining_seconds_func: Callable[..., float],
    safe_validate_pretrade_func: Callable[..., tuple[bool, str, dict[str, Any]]],
    get_sector_func: Callable[[str], str | None],
) -> NettingSubmitPreludeResult:
    snapshot_updates: dict[str, Any] = {}
    gates_added: list[str] = []

    if portfolio_optimizer_enabled and portfolio_optimizer is None:
        opt_runtime_context = dict(portfolio_optimizer_context)
        if bool(opt_runtime_context.get("init_failed")):
            snapshot_updates["portfolio_optimizer"] = opt_runtime_context
            if not bool(opt_runtime_context.get("init_fail_open")):
                blocked_reason = "PORTFOLIO_OPTIMIZER_INIT_FAILED"
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
                    blocked_metrics={"portfolio_optimizer": opt_runtime_context},
                    blocked_order_intent=None,
                )

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
        try:
            opt_allowed, opt_context = portfolio_optimizer_allows_trade_func(
                optimizer=portfolio_optimizer,
                symbol=symbol,
                proposed_position=float(proposed_position),
                current_positions=current_positions_for_optimizer,
                market_data=portfolio_optimizer_market_data,
            )
        except (RuntimeError, ValueError, TypeError, KeyError, OSError) as exc:
            opt_context = {
                "decision": "error",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "fail_closed": _fail_closed_on_optimizer_error(
                    cfg=cfg,
                    opening_trade=opening_trade,
                ),
            }
            snapshot_updates["portfolio_optimizer"] = dict(portfolio_optimizer_context | opt_context)
            if bool(opt_context["fail_closed"]):
                blocked_reason = "PORTFOLIO_OPTIMIZER_DECISION_ERROR"
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
            opt_allowed = True
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
    try:
        require_realtime_nbbo = bool(getattr(cfg, "execution_require_realtime_nbbo", True))
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        require_realtime_nbbo = True
    enforce_opening_nbbo = bool(get_env("AI_TRADING_ENFORCE_NBBO_FOR_OPENINGS", True, cast=bool))
    (
        submit_quote_source,
        submit_bid_at_arrival,
        submit_ask_at_arrival,
        submit_mid_at_arrival,
        submit_arrival_price,
        submit_quote_ts,
    ) = resolve_order_quote_basis_func(
        runtime,
        symbol=symbol,
        side=side,
        fallback_price=price,
    )

    broker_submit_allowed = bool(breakers.allow("broker_submit"))
    broker_ready = broker_submit_allowed and auth_forbidden_retry_after <= 0.0
    broker_ready_reason: str | None = None
    broker_cooldown_remaining_sec: float | None = None
    if auth_forbidden_retry_after > 0.0:
        broker_ready_reason = "AUTH_BROKER_HALT_FORBIDDEN_COOLDOWN"
        broker_cooldown_remaining_sec = float(auth_forbidden_retry_after)
    elif not broker_submit_allowed:
        broker_ready_reason = breakers.open_reason("broker_submit") or "CIRCUIT_OPEN_broker_submit"
        state.halt_reason = broker_ready_reason

    pretrade_cfg, effective_collar_pct, collar_mult = build_pretrade_validation_cfg(
        cfg,
        thin_liquidity=liq_regime.value == "thin" or str(liq_regime).endswith(".THIN"),
    )
    if effective_collar_pct is not None and collar_mult is not None:
        snapshot_updates["liquidity_collar_multiplier"] = collar_mult
        snapshot_updates["price_collar_pct_effective"] = effective_collar_pct

    quote_quality_ok = all(
        value is not None
        for value in (
            submit_bid_at_arrival,
            submit_ask_at_arrival,
            submit_mid_at_arrival,
        )
    )
    sampling_symbols = {
        str(candidate).strip().upper()
        for candidate in getattr(cfg, "paper_sampling_allowed_symbols", ())
        if str(candidate).strip()
    }
    paper_sampling_passive = (
        str(getattr(cfg, "execution_mode", "sim") or "sim").strip().lower()
        == "paper"
        and bool(getattr(cfg, "paper_sampling_enabled", False))
        and bool(getattr(cfg, "paper_sampling_passive_only", True))
        and str(symbol).strip().upper() in sampling_symbols
    )
    execution_profile = (
        "paper_sampling_passive" if paper_sampling_passive else "standard"
    )
    final_intent_context = build_execution_intent_context(
        now=now,
        salt=str(getattr(cfg, "seed", "seed")),
        symbol=symbol,
        side=side,
        delta_shares=int(delta_shares),
        price=float(price),
        bar_ts=bar_ts,
        spread_bps=float(liq_features.spread_bps),
        liquidity_bucket=liq_regime.value.upper(),
        quote_quality_ok=quote_quality_ok,
        sector=get_sector_func(symbol),
        event_risk=event_risk_near,
        slo_derisk_details={**dict(slo_derisk_details), "rolling_volume": float(liq_features.rolling_volume)},
        config_snapshot=symbol_snapshot | snapshot_updates,
        execution_model_lineage=execution_model_lineage,
        submit_quote_source=submit_quote_source,
        submit_bid_at_arrival=submit_bid_at_arrival,
        submit_ask_at_arrival=submit_ask_at_arrival,
        submit_mid_at_arrival=submit_mid_at_arrival,
        submit_quote_ts=submit_quote_ts,
        opening_trade=bool(opening_trade),
        require_realtime_nbbo=bool(require_realtime_nbbo and enforce_opening_nbbo),
        kill_switch_active=bool(kill_switch_active),
        kill_switch_reason="kill_switch" if kill_switch_active else None,
        broker_ready=bool(broker_ready),
        broker_ready_reason=broker_ready_reason,
        broker_cooldown_remaining_sec=broker_cooldown_remaining_sec,
        correlation_id=correlation_id,
        decision_timestamp=now,
        source_timestamp=bar_ts,
        order_type="limit",
        execution_profile=execution_profile,
    )
    intent = final_intent_context.pretrade_intent
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
        blocked_order_intent = intent.to_contract()
        blocked_metadata = dict(blocked_order_intent.metadata)
        blocked_metadata.update(final_intent_context.order_lineage_metadata)
        blocked_order_intent = replace(
            blocked_order_intent,
            decision_trace_id=final_intent_context.decision_trace_id,
            correlation_id=final_intent_context.correlation_id,
            metadata=blocked_metadata,
        )
        return NettingSubmitPreludeResult(
            execution_intent_context=None,
            submit_quote_source=submit_quote_source,
            submit_bid_at_arrival=submit_bid_at_arrival,
            submit_ask_at_arrival=submit_ask_at_arrival,
            submit_mid_at_arrival=submit_mid_at_arrival,
            submit_arrival_price=submit_arrival_price,
            gates_added=tuple(gates_added),
            snapshot_updates=snapshot_updates,
            blocked_reason=pretrade_reason,
            blocked_metrics={"pretrade": pretrade_details},
            blocked_order_intent=blocked_order_intent,
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
