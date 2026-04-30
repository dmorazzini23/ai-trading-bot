"""Legacy/non-netting submit runtime extracted from ``bot_engine.py``."""

from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import importlib
import math
import types
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Mapping

from ai_trading.core.execution_guards import build_pretrade_validation_cfg
from ai_trading.core.execution_intent import (
    ExecutionIntentContext,
    build_execution_intent_context,
)
from ai_trading.oms.ledger import LedgerEntry, OrderLedger
from ai_trading.oms.pretrade import safe_validate_pretrade
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def _mapping_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    try:
        return dict(value)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return {}


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _canonical_intent_side(side: str) -> str | None:
    normalized = str(side or "").strip().lower()
    if normalized in {"sell_short", "short"}:
        return "sell_short"
    if normalized in {"sell", "exit"}:
        return "sell"
    if normalized in {"buy", "long", "buy_to_cover", "cover"}:
        return "buy"
    return None


def _is_opening_trade(*, side: str, current_qty: int) -> bool:
    normalized = str(side or "").strip().lower()
    if normalized in {"sell_short", "short"}:
        return int(current_qty) >= 0
    if normalized in {"buy_to_cover", "cover"}:
        return False
    if normalized in {"sell", "exit"}:
        return int(current_qty) == 0
    return int(current_qty) == 0


def _quote_spread_bps(
    *,
    bid: float | None,
    ask: float | None,
    reference_price: float,
) -> float:
    if bid is None or ask is None or reference_price <= 0.0:
        return 0.0
    spread = float(ask) - float(bid)
    if spread <= 0.0:
        return 0.0
    return max((spread / float(reference_price)) * 10_000.0, 0.0)


def _resolve_execution_lineage(exec_kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    execution_model_lineage = {
        "model_id": exec_kwargs.get("model_id"),
        "model_version": exec_kwargs.get("model_version"),
        "dataset_hash": exec_kwargs.get("dataset_hash"),
        "feature_version": exec_kwargs.get("feature_version"),
        "model_artifact_hash": exec_kwargs.get("model_artifact_hash"),
    }
    config_snapshot = {
        "config_snapshot_hash": exec_kwargs.get("config_snapshot_hash"),
        "effective_policy_hash": exec_kwargs.get("policy_hash"),
    }
    return execution_model_lineage, config_snapshot


def _resolve_legacy_ledger(be: Any, cfg: Any) -> Any:
    execution_mode = str(getattr(cfg, "execution_mode", "sim") or "sim").strip().lower()
    if execution_mode == "live":
        if getattr(be.state, "_oms_ledger", None) is not None:
            setattr(be.state, "_oms_ledger", None)
        be.emit_once(
            be.logger,
            "OMS_LEDGER_DISABLED_LIVE",
            "info",
            "Live execution disables JSONL OMS ledger side paths; intent store is authoritative",
        )
        return None
    if not bool(getattr(cfg, "ledger_enabled", False)):
        if getattr(be.state, "_oms_ledger", None) is not None:
            setattr(be.state, "_oms_ledger", None)
        return None
    ledger = getattr(be.state, "_oms_ledger", None)
    if ledger is not None:
        return ledger
    ledger_path = resolve_runtime_artifact_path(
        str(getattr(cfg, "ledger_path", "runtime/oms_ledger.jsonl")),
        default_relative=str(getattr(cfg, "ledger_path", "runtime/oms_ledger.jsonl")),
    )
    ledger = OrderLedger(
        str(ledger_path),
        float(getattr(cfg, "ledger_lookback_hours", 24.0)),
    )
    setattr(be.state, "_oms_ledger", ledger)
    return ledger


def _record_skip_submit(
    be: Any,
    *,
    symbol: str,
    side: str,
    reason: str,
    detail: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> None:
    skip_handler = getattr(be._exec_engine, "_skip_submit", None)
    if not callable(skip_handler):
        return
    try:
        skip_handler(
            symbol=symbol,
            side=side,
            reason=reason,
            detail=detail,
            context=dict(context or {}),
        )
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        be.logger.debug("LEGACY_SUBMIT_SKIP_HANDLER_FAILED", exc_info=True)


def _resolved_submit_runtime(be: Any, ctx: Any) -> Any:
    if hasattr(ctx, "execution_engine") or hasattr(ctx, "exec_engine"):
        return ctx
    return types.SimpleNamespace(execution_engine=be._exec_engine, exec_engine=be._exec_engine)


def _attach_order_identity(order: Any, *, client_order_id: str) -> None:
    if isinstance(order, dict):
        order.setdefault("client_order_id", client_order_id)
        return
    try:
        current = getattr(order, "client_order_id", None)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        current = None
    if current not in (None, "", 0):
        return
    try:
        setattr(order, "client_order_id", client_order_id)
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        pass


def _order_field(order: Any, name: str) -> Any:
    if isinstance(order, dict):
        return order.get(name)
    return getattr(order, name, None)


def _record_legacy_ledger_submission(
    ledger: Any,
    *,
    intent_context: ExecutionIntentContext,
    order: Any,
    qty: int,
    now: datetime,
) -> None:
    if ledger is None:
        return
    broker_order_id_raw = _order_field(order, "id")
    broker_status_raw = _order_field(order, "status")
    try:
        ledger.record(
            LedgerEntry(
                client_order_id=intent_context.client_order_id,
                symbol=str(intent_context.pretrade_intent.symbol),
                bar_ts=intent_context.pretrade_intent.bar_ts.isoformat(),
                qty=float(qty),
                side=str(intent_context.pretrade_intent.side),
                limit_price=intent_context.pretrade_intent.limit_price,
                ts=now.isoformat(),
                broker_order_id=(
                    str(broker_order_id_raw).strip()
                    if broker_order_id_raw not in (None, "")
                    else None
                ),
                status=(
                    str(broker_status_raw).strip()
                    if broker_status_raw not in (None, "")
                    else None
                ),
            )
        )
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        be_logger = importlib.import_module("ai_trading.core.bot_engine").logger
        be_logger.debug("LEGACY_LEDGER_RECORD_FAILED", exc_info=True)


def _resolve_execution_intent_context(
    be: Any,
    *,
    ctx: Any,
    cfg: Any,
    symbol: str,
    qty: int,
    side_norm: str,
    price: float,
    exec_kwargs: Mapping[str, Any],
) -> tuple[ExecutionIntentContext, dict[str, Any], dict[str, Any]]:
    now = datetime.now(UTC)
    current_qty = int(be._current_qty(ctx, symbol))
    opening_trade = _is_opening_trade(side=side_norm, current_qty=current_qty)
    auth_forbidden_retry_after = be._auth_forbidden_cooldown_remaining_seconds(
        be.state,
        symbol=symbol,
        side=side_norm,
        now=now,
    )
    breakers = be._dependency_breakers(be.state)
    broker_submit_allowed = bool(breakers.allow("broker_submit"))
    broker_ready = broker_submit_allowed and auth_forbidden_retry_after <= 0.0
    broker_ready_reason: str | None = None
    broker_cooldown_remaining_sec: float | None = None
    if auth_forbidden_retry_after > 0.0:
        broker_ready_reason = "AUTH_BROKER_HALT_FORBIDDEN_COOLDOWN"
        broker_cooldown_remaining_sec = float(auth_forbidden_retry_after)
    elif not broker_submit_allowed:
        broker_ready_reason = (
            breakers.open_reason("broker_submit") or "CIRCUIT_OPEN_broker_submit"
        )

    (
        submit_quote_source,
        submit_bid_at_arrival,
        submit_ask_at_arrival,
        submit_mid_at_arrival,
        submit_arrival_price,
        submit_quote_ts,
    ) = be._resolve_order_quote_basis(
        ctx,
        symbol=symbol,
        side=side_norm,
        fallback_price=price,
    )
    execution_model_lineage, config_snapshot = _resolve_execution_lineage(exec_kwargs)
    explicit_client_order_id = str(exec_kwargs.get("client_order_id") or "").strip() or None
    explicit_decision_trace_id = (
        str(exec_kwargs.get("decision_trace_id") or "").strip() or None
    )
    intent_side = _canonical_intent_side(side_norm)
    if intent_side is None:
        raise ValueError(f"invalid order side: {side_norm!r}")
    delta_shares = int(qty) if intent_side == "buy" else -int(qty)
    reference_price = float(submit_arrival_price) if submit_arrival_price is not None else float(price)
    spread_bps = _quote_spread_bps(
        bid=submit_bid_at_arrival,
        ask=submit_ask_at_arrival,
        reference_price=reference_price,
    )
    execution_mode = str(
        getattr(ctx, "execution_mode", be.get_env("EXECUTION_MODE", "paper", cast=str))
        or "paper"
    ).strip().lower()
    require_realtime_nbbo_default = execution_mode == "live"
    try:
        require_realtime_nbbo = bool(
            getattr(cfg, "execution_require_realtime_nbbo", require_realtime_nbbo_default)
        )
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        require_realtime_nbbo = require_realtime_nbbo_default
    require_realtime_nbbo = bool(
        require_realtime_nbbo
        and be.get_env("AI_TRADING_ENFORCE_NBBO_FOR_OPENINGS", True, cast=bool)
    )
    kill_switch_active, kill_switch_reason = be._kill_switch_active(cfg)
    quote_fields = (
        submit_bid_at_arrival,
        submit_ask_at_arrival,
        submit_mid_at_arrival,
    )
    quote_quality_ok: bool | None
    if all(value is None for value in quote_fields):
        quote_quality_ok = None
    else:
        quote_quality_ok = all(value is not None for value in quote_fields)
    intent_context = build_execution_intent_context(
        now=now,
        salt=str(getattr(cfg, "seed", "seed")),
        symbol=symbol,
        side=intent_side,
        delta_shares=int(delta_shares),
        price=float(price),
        bar_ts=now,
        spread_bps=float(spread_bps),
        liquidity_bucket="NORMAL",
        quote_quality_ok=quote_quality_ok,
        sector=None,
        event_risk=False,
        slo_derisk_details={"rolling_volume": 0.0},
        config_snapshot=config_snapshot,
        execution_model_lineage=execution_model_lineage,
        submit_quote_source=submit_quote_source,
        submit_bid_at_arrival=submit_bid_at_arrival,
        submit_ask_at_arrival=submit_ask_at_arrival,
        submit_mid_at_arrival=submit_mid_at_arrival,
        submit_quote_ts=submit_quote_ts,
        opening_trade=bool(opening_trade),
        require_realtime_nbbo=bool(require_realtime_nbbo),
        kill_switch_active=bool(kill_switch_active),
        kill_switch_reason=kill_switch_reason,
        broker_ready=bool(broker_ready),
        broker_ready_reason=broker_ready_reason,
        broker_cooldown_remaining_sec=broker_cooldown_remaining_sec,
    )
    resolved_client_order_id = explicit_client_order_id or intent_context.client_order_id
    resolved_decision_trace_id = (
        explicit_decision_trace_id or intent_context.decision_trace_id
    )
    pretrade_intent = intent_context.pretrade_intent
    if resolved_client_order_id != intent_context.client_order_id:
        pretrade_intent = replace(
            pretrade_intent,
            client_order_id=resolved_client_order_id,
        )
    order_lineage_metadata = dict(intent_context.order_lineage_metadata)
    order_annotations = dict(intent_context.order_annotations)
    if resolved_decision_trace_id:
        order_lineage_metadata["decision_trace_id"] = resolved_decision_trace_id
        order_annotations["decision_trace_id"] = resolved_decision_trace_id
    if resolved_client_order_id:
        order_annotations.setdefault("client_order_id", resolved_client_order_id)
    return (
        ExecutionIntentContext(
            client_order_id=resolved_client_order_id,
            decision_trace_id=resolved_decision_trace_id,
            pretrade_intent=pretrade_intent,
            order_lineage_metadata=order_lineage_metadata,
            order_annotations=order_annotations,
        ),
        {
            "submit_arrival_price": submit_arrival_price,
            "submit_quote_source": submit_quote_source,
        },
        {
            "auth_forbidden_retry_after_sec": broker_cooldown_remaining_sec,
            "broker_ready_reason": broker_ready_reason,
        },
    )


def submit_order_runtime(
    ctx: Any,
    symbol: str,
    qty: int,
    side: str,
    *,
    price: float | None = None,
    **exec_kwargs: Any,
) -> Any | None:
    """Submit a legacy/non-netting order through shared runtime controls."""

    be = importlib.import_module("ai_trading.core.bot_engine")
    exec_kwargs = dict(exec_kwargs)

    annotations = _mapping_dict(exec_kwargs.get("annotations"))
    metadata = _mapping_dict(exec_kwargs.get("metadata"))

    annotation_price_source = (
        annotations.get("price_source")
        or annotations.get("quote_source")
        or annotations.get("fallback_source")
    )
    price_source_label = annotation_price_source
    if price_source_label in (None, ""):
        try:
            price_source_label = be.get_price_source(symbol)
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            price_source_label = None
    if annotation_price_source in (None, "") and price_source_label not in (None, ""):
        annotations["price_source"] = price_source_label

    fallback_flag = bool(
        exec_kwargs.get("using_fallback_price")
        or annotations.get("using_fallback_price")
    )
    normalized_price_source = be._normalize_quote_source_token(price_source_label)
    if not fallback_flag and normalized_price_source is not None:
        try:
            fallback_flag = not be._is_primary_price_source(normalized_price_source)
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            be.logger.debug("PRICE_SOURCE_FALLBACK_FLAG_PARSE_FAILED", exc_info=True)
    if fallback_flag:
        annotations["using_fallback_price"] = True
        exec_kwargs["using_fallback_price"] = True

    if exec_kwargs.get("expected_net_edge_bps") is None:
        for candidate_key in (
            "expected_net_edge_bps",
            "expected_edge_bps",
            "edge_bps",
            "alpha_edge_bps",
        ):
            candidate = annotations.get(candidate_key)
            parsed = _safe_float(candidate)
            if parsed is not None:
                exec_kwargs["expected_net_edge_bps"] = float(parsed)
                break

    if exec_kwargs.get("price_hint") is None and price is not None:
        exec_kwargs["price_hint"] = price

    cfg = be._resolve_trading_config(ctx)
    kill_switch, kill_reason = be._kill_switch_active(cfg)
    if kill_switch:
        be.logger.warning(
            "KILL_SWITCH_BLOCK",
            extra={"symbol": symbol, "reason": kill_reason or "kill_switch"},
        )
        _record_skip_submit(
            be,
            symbol=symbol,
            side=str(side).lower(),
            reason="KILL_SWITCH_BLOCK",
            detail=str(kill_reason or "kill_switch"),
        )
        return None

    rth_only = bool(getattr(cfg, "rth_only", True))
    allow_extended = bool(getattr(cfg, "allow_extended", False))
    if (rth_only or not allow_extended) and not be.market_is_open():
        be.logger.warning(
            "MARKET_CLOSED_ORDER_SKIP",
            extra={"symbol": symbol, "reason": "MARKET_CLOSED_BLOCK"},
        )
        _record_skip_submit(
            be,
            symbol=symbol,
            side=str(side).lower(),
            reason="MARKET_HOURS_BLOCK",
            detail="MARKET_CLOSED_BLOCK",
        )
        return None

    if be._exec_engine is None:
        be.logger.error(
            "EXEC_ENGINE_NOT_INITIALIZED",
            extra={"symbol": symbol, "qty": qty, "side": side},
        )
        raise RuntimeError("Execution engine not initialized. Cannot execute orders.")

    if hasattr(be.S, "liquidity_checks_enabled") and be.CFG.liquidity_checks_enabled:
        try:
            from ai_trading.execution.liquidity import LiquidityManager

            lm = LiquidityManager()
            lm.pre_trade_check(
                {"symbol": symbol, "qty": qty, "side": side},
                getattr(ctx, "market_data", None),
            )
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            be.JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as exc:
            execution_mode = str(getattr(cfg, "execution_mode", "sim") or "sim").lower()
            if execution_mode in {"paper", "live"} and not bool(
                be.get_env("PYTEST_RUNNING", "0", cast=bool)
            ):
                be.logger.error(
                    "LIQUIDITY_PRECHECK_FAILED_BLOCK",
                    extra={
                        "symbol": symbol,
                        "qty": qty,
                        "side": side,
                        "mode": execution_mode,
                        "error": str(exc),
                    },
                )
                _record_skip_submit(
                    be,
                    symbol=symbol,
                    side=str(side).lower(),
                    reason="LIQUIDITY_PRECHECK_FAILED_BLOCK",
                    detail=str(exc),
                )
                return None
            be.logger.warning("Liquidity checks failed open-loop: %s", exc)

    side_norm = str(side).lower().strip()
    intent_side = _canonical_intent_side(side_norm)
    if intent_side is None:
        be.logger.warning(
            "INVALID_ORDER_SIDE_BLOCK",
            extra={"symbol": symbol, "qty": qty, "side": side},
        )
        _record_skip_submit(
            be,
            symbol=symbol,
            side=side_norm or "unknown",
            reason="INVALID_ORDER_SIDE_BLOCK",
            detail=str(side),
        )
        return None
    if side_norm in ("sell_short", "short"):
        core_side = be.CoreOrderSide.SELL_SHORT
    elif side_norm in ("sell", "exit"):
        core_side = be.CoreOrderSide.SELL
    else:
        core_side = be.CoreOrderSide.BUY

    cycle_intent_compaction_enabled = bool(
        be.get_env("AI_TRADING_CYCLE_INTENT_COMPACTION_ENABLED", True, cast=bool)
    )
    if cycle_intent_compaction_enabled and bool(getattr(be.state, "running", False)):
        if not be._reserve_cycle_submit_intent(
            be.state,
            symbol=symbol,
            side=side_norm,
        ):
            normalized_side = be._normalize_submit_side(side_norm) or side_norm
            compaction = getattr(be.state, "cycle_submit_compaction", None)
            reserved_count = len(compaction) if isinstance(compaction, set) else 0
            _record_skip_submit(
                be,
                symbol=symbol,
                side=normalized_side,
                reason="cycle_duplicate_intent",
                context={
                    "source": "bot_cycle_compaction",
                    "reserved_intents": int(reserved_count),
                },
            )
            return None

    if price is None:
        price = be.get_latest_price(symbol)
        if not isinstance(price, (int, float)) or price <= 0:
            market_data = getattr(ctx, "market_data", None)
            price = be.get_latest_close(market_data) if market_data is not None else 0.0
    price_value = _safe_float(price)
    if price_value is None or price_value <= 0.0:
        be.logger.warning(
            "INVALID_PRICE_BLOCK",
            extra={"symbol": symbol, "side": side, "price": price},
        )
        _record_skip_submit(
            be,
            symbol=symbol,
            side=str(side).lower(),
            reason="INVALID_PRICE_BLOCK",
            detail=str(price),
        )
        return None
    price = float(price_value)

    intent_context, submit_snapshot, broker_snapshot = _resolve_execution_intent_context(
        be,
        ctx=ctx,
        cfg=cfg,
        symbol=symbol,
        qty=int(max(qty, 0)),
        side_norm=side_norm,
        price=float(price),
        exec_kwargs=exec_kwargs,
    )
    ledger = _resolve_legacy_ledger(be, cfg)
    rate_limiter = be._pretrade_rate_limiter(be.state)
    pretrade_cfg, _, _ = build_pretrade_validation_cfg(cfg, thin_liquidity=False)
    allowed, pretrade_reason, pretrade_details = safe_validate_pretrade(
        intent_context.pretrade_intent,
        cfg=pretrade_cfg,
        ledger=ledger,
        rate_limiter=rate_limiter,
    )
    if not allowed:
        be.logger.warning(
            pretrade_reason,
            extra={
                "symbol": symbol,
                "side": side_norm,
                **pretrade_details,
            },
        )
        _record_skip_submit(
            be,
            symbol=symbol,
            side=side_norm,
            reason=pretrade_reason,
            detail=str(pretrade_details.get("reason") or pretrade_reason),
            context=pretrade_details,
        )
        return None

    merged_annotations = dict(intent_context.order_annotations)
    merged_annotations.update(annotations)
    if fallback_flag:
        merged_annotations["using_fallback_price"] = True

    merged_metadata = dict(intent_context.order_lineage_metadata)
    merged_metadata.update(metadata)

    engine_kwargs = dict(exec_kwargs)
    engine_kwargs["annotations"] = merged_annotations
    if merged_metadata:
        engine_kwargs["metadata"] = merged_metadata
    else:
        engine_kwargs.pop("metadata", None)
    engine_kwargs.setdefault("client_order_id", intent_context.client_order_id)
    engine_kwargs.setdefault("decision_trace_id", intent_context.decision_trace_id)
    engine_kwargs.setdefault(
        "price_hint",
        float(submit_snapshot.get("submit_arrival_price") or price),
    )

    for lineage_key in (
        "model_id",
        "model_version",
        "config_snapshot_hash",
        "dataset_hash",
        "feature_version",
        "model_artifact_hash",
        "policy_hash",
    ):
        lineage_value = merged_metadata.get(lineage_key)
        if lineage_value not in (None, ""):
            engine_kwargs.setdefault(lineage_key, lineage_value)

    timing_meta = {"symbol": symbol, "side": side_norm, "qty": int(max(qty, 0))}
    runtime_like = _resolved_submit_runtime(be, ctx)
    with be.execution_span(None, **timing_meta):
        order = be._exec_engine.execute_order(
            symbol,
            core_side,
            qty,
            price=price,
            **engine_kwargs,
        )

    if order is None:
        submit_none_reason = be._resolve_submit_none_reason(runtime_like)
        be._record_auth_forbidden_cooldown(
            be.state,
            symbol=symbol,
            side=side_norm,
            reason=submit_none_reason,
            now=datetime.now(UTC),
        )
        return None

    status_token = be._normalize_order_status_token(
        be._extract_order_value(order, "status")
    )
    if status_token in {"rejected", "canceled", "cancelled", "expired", "done_for_day", "skipped"}:
        reason_code = f"BROKER_ORDER_{status_token.upper()}".replace("CANCELLED", "CANCELED")
        be._record_auth_forbidden_cooldown(
            be.state,
            symbol=symbol,
            side=side_norm,
            reason=reason_code,
            now=datetime.now(UTC),
        )
        be.logger.warning(
            "BROKER_ORDER_NOT_ACCEPTED",
            extra={"symbol": symbol, "side": side_norm, "status": status_token},
        )
        return None

    _attach_order_identity(order, client_order_id=intent_context.client_order_id)
    _record_legacy_ledger_submission(
        ledger,
        intent_context=intent_context,
        order=order,
        qty=int(qty),
        now=datetime.now(UTC),
    )
    broker_ready_reason = broker_snapshot.get("broker_ready_reason")
    if broker_ready_reason not in (None, ""):
        be.logger.info(
            "LEGACY_SUBMIT_BROKER_READY_CONTEXT",
            extra={
                "symbol": symbol,
                "side": side_norm,
                "broker_ready_reason": broker_ready_reason,
                "auth_forbidden_retry_after_sec": broker_snapshot.get(
                    "auth_forbidden_retry_after_sec"
                ),
            },
        )
    return order


__all__ = ["submit_order_runtime"]
