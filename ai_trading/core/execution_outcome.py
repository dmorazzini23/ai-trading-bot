"""Helpers for normalizing broker submission outcomes and post-submit telemetry."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence

from ai_trading.analytics.tca import ExecutionBenchmark, FillSummary
from ai_trading.config.management import get_env
from ai_trading.oms.ledger import LedgerEntry


@dataclass(frozen=True, slots=True)
class SubmittedOrderState:
    status_text: str
    status_token: str
    broker_order_id: str | None
    filled_qty: float
    requested_qty: float
    fill_price: float | None
    fill_timestamp: datetime | None
    fill_fees: float
    persistable_fill: bool


def normalize_submitted_order(
    order: Any,
    *,
    delta_shares: int,
    extract_order_value: Callable[..., Any],
    extract_order_fill_timestamp: Callable[[Any], datetime | None],
    normalize_order_status_token: Callable[[Any], str],
    safe_float: Callable[[Any], float | None],
    has_persistable_fill: Callable[..., bool],
) -> SubmittedOrderState:
    """Normalize broker order payload into stable post-submit fields."""
    order_status = extract_order_value(order, "status") if order is not None else None
    status_value = getattr(order_status, "value", order_status)
    status_text = str(status_value).strip() if status_value not in (None, "") else "submitted"
    status_token = normalize_order_status_token(status_value)
    broker_order_id_raw = (
        extract_order_value(order, "id", "order_id", "client_order_id")
        if order is not None
        else None
    )
    broker_order_id = str(broker_order_id_raw) if broker_order_id_raw is not None else None
    filled_qty = safe_float(extract_order_value(order, "filled_quantity", "filled_qty"))
    if filled_qty is None:
        filled_qty = 0.0
    requested_qty = safe_float(
        extract_order_value(order, "requested_quantity", "qty", "quantity")
    )
    if requested_qty is None:
        requested_qty = float(abs(delta_shares))
    fill_price = safe_float(
        extract_order_value(
            order,
            "filled_avg_price",
            "fill_price",
            "average_fill_price",
            "average_price",
        )
    )
    fill_timestamp = extract_order_fill_timestamp(order)
    raw_fees = safe_float(
        extract_order_value(
            order,
            "fees",
            "fee",
            "commission",
            "filled_fee",
            "filled_fees",
            "total_fees",
        )
    )
    fill_fees = abs(float(raw_fees)) if raw_fees is not None else 0.0
    persistable_fill = has_persistable_fill(
        status_token=status_token,
        filled_qty=float(filled_qty),
        fill_price=fill_price,
    )
    return SubmittedOrderState(
        status_text=status_text,
        status_token=status_token,
        broker_order_id=broker_order_id,
        filled_qty=float(filled_qty),
        requested_qty=float(requested_qty),
        fill_price=fill_price,
        fill_timestamp=fill_timestamp,
        fill_fees=float(fill_fees),
        persistable_fill=bool(persistable_fill),
    )


def record_successful_submission(
    *,
    ledger: Any,
    state: Any,
    symbol: str,
    client_order_id: str,
    bar_ts: datetime,
    delta_shares: int,
    side: str,
    price: float,
    now: datetime,
    order_state: SubmittedOrderState,
    proposals: Sequence[Any],
) -> None:
    """Persist ledger/runtime state for a submitted order."""
    if ledger is not None:
        ledger.record(
            LedgerEntry(
                client_order_id=client_order_id,
                symbol=symbol,
                bar_ts=bar_ts.isoformat(),
                qty=float(abs(delta_shares)),
                side=side,
                limit_price=price,
                ts=now.isoformat(),
                broker_order_id=order_state.broker_order_id,
                status=order_state.status_text or None,
            )
        )
    state.last_order_bar_ts[symbol] = bar_ts
    state.last_order_client_id[symbol] = client_order_id

    total_target = sum(abs(float(getattr(p, "target_dollars", 0.0) or 0.0)) for p in proposals)
    if total_target > 0:
        trade_notional = abs(delta_shares) * price
        for proposal in proposals:
            ratio = abs(float(getattr(proposal, "target_dollars", 0.0) or 0.0)) / total_target
            turnover_key = (now.date(), getattr(proposal, "sleeve", ""), symbol)
            state.turnover_dollars[turnover_key] = (
                state.turnover_dollars.get(turnover_key, 0.0) + trade_notional * ratio
            )


def build_order_metrics_and_tca(
    *,
    symbol: str,
    side: str,
    price: float,
    delta_shares: int,
    now: datetime,
    net_target: Any,
    order: Any,
    order_state: SubmittedOrderState,
    submit_arrival_price: float | None,
    submit_bid_at_arrival: float | None,
    submit_ask_at_arrival: float | None,
    submit_mid_at_arrival: float | None,
    submit_quote_source: str | None,
    candidate_expected_net_edge: Mapping[str, Any],
    candidate_expected_capture: Mapping[str, Any],
    get_regime_signal_profile_func: Callable[[], str],
    normalize_quote_source_token_func: Callable[[Any], str | None],
    resolve_quote_proxy_source_func: Callable[..., str | None],
    resolved_tca_path_func: Callable[[], Any],
    write_tca_record_func: Callable[[str, Mapping[str, Any]], None],
    session_bucket_from_ts_func: Callable[[datetime], str],
    compute_attribution_metrics_func: Callable[..., dict[str, Any]],
    safe_float: Callable[[Any], float | None],
    logger: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build attribution metrics and TCA payload for a submitted order."""
    metrics: dict[str, Any] = {}
    tca_record: dict[str, Any] | None = None
    persistable_fill = bool(order_state.persistable_fill)
    fill_price = order_state.fill_price
    arrival_price_for_metrics = (
        float(submit_arrival_price) if submit_arrival_price is not None else float(price)
    )
    try:
        metrics = compute_attribution_metrics_func(
            arrival_price=arrival_price_for_metrics,
            fill_price=float(fill_price) if fill_price is not None else None,
            side=side,
            bid=submit_bid_at_arrival,
            ask=submit_ask_at_arrival,
            order_ts=now,
            fill_ts=order_state.fill_timestamp if persistable_fill else None,
        )
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        metrics = {}
        fill_price = None
        persistable_fill = False

    if not bool(get_env("AI_TRADING_TCA_ENABLED", False, cast=bool)):
        return metrics, None

    fill_vwap = float(fill_price) if fill_price is not None else None
    arrival_benchmark = str(get_env("AI_TRADING_TCA_ARRIVAL_BENCHMARK", "decision")).strip().lower()
    if arrival_benchmark not in {"decision", "submit"}:
        arrival_benchmark = "decision"
    allow_proxy_quotes = bool(get_env("AI_TRADING_TCA_ALLOW_PROXY_QUOTES", True, cast=bool))
    arrival_ts = net_target.bar_ts if arrival_benchmark == "decision" else now
    first_fill_ts = order_state.fill_timestamp if persistable_fill else None
    if first_fill_ts is None and persistable_fill:
        first_fill_ts = now
    partial_fill = order_state.status_token == "partially_filled"
    if (
        not partial_fill
        and persistable_fill
        and float(order_state.requested_qty) > 0
        and float(order_state.filled_qty) < float(order_state.requested_qty)
    ):
        partial_fill = True
    tca_qty = (
        float(order_state.filled_qty)
        if persistable_fill and order_state.filled_qty > 0
        else float(abs(delta_shares))
    )
    mid_at_arrival = submit_mid_at_arrival
    if mid_at_arrival is None and allow_proxy_quotes:
        mid_at_arrival = float(arrival_price_for_metrics)
    benchmark = ExecutionBenchmark(
        arrival_price=float(arrival_price_for_metrics),
        mid_at_arrival=mid_at_arrival if allow_proxy_quotes else None,
        bid_at_arrival=submit_bid_at_arrival,
        ask_at_arrival=submit_ask_at_arrival,
        bar_close_price=float(price),
        decision_ts=arrival_ts,
        submit_ts=now,
        first_fill_ts=first_fill_ts,
    )
    fill_summary = FillSummary(
        fill_vwap=fill_vwap,
        total_qty=tca_qty,
        fees=float(order_state.fill_fees) if persistable_fill else 0.0,
        status=order_state.status_text,
        partial_fill=partial_fill,
    )
    from ai_trading.analytics.tca import build_tca_record

    tca_record = build_tca_record(
        client_order_id=str(getattr(order, "client_order_id", None) or getattr(order, "id", None) or ""),
        symbol=symbol,
        side=side,
        benchmark=benchmark,
        fill=fill_summary,
        sleeve=net_target.proposals[0].sleeve if net_target.proposals else None,
        regime_profile=get_regime_signal_profile_func(),
        provider="alpaca",
        order_type="limit",
        quote_proxy=allow_proxy_quotes,
    )
    session_regime_token = session_bucket_from_ts_func(now)
    spread_paid_for_role = safe_float(tca_record.get("spread_paid_bps"))
    liquidity_role_token = "maker"
    if str(side).strip().lower() in {"buy", "sell"} and spread_paid_for_role is not None:
        if float(spread_paid_for_role) >= 0.75:
            liquidity_role_token = "taker"
        elif float(spread_paid_for_role) > 0.05:
            liquidity_role_token = "mixed"
    venue_token = str(
        getattr(order, "exchange", None) or getattr(order, "venue", None) or "ALPACA"
    ).strip().upper() or "ALPACA"
    tca_record["liquidity_role"] = str(liquidity_role_token)
    tca_record["venue"] = str(venue_token)
    tca_record["session_regime"] = str(session_regime_token)
    tca_record["venue_session"] = f"{venue_token}:{session_regime_token}"
    expected_edge_for_tca = safe_float(candidate_expected_net_edge.get(symbol))
    if expected_edge_for_tca is not None:
        tca_record["expected_net_edge_bps"] = float(expected_edge_for_tca)
    expected_capture_for_tca = safe_float(candidate_expected_capture.get(symbol))
    if expected_capture_for_tca is not None:
        tca_record["expected_capture_bps"] = float(expected_capture_for_tca)
    if not persistable_fill:
        tca_record["fill_price"] = None
        tca_record["fill_vwap"] = None
        tca_record["is_bps"] = None
        tca_record["spread_paid_bps"] = None
        tca_record["fill_latency_ms"] = None
    tca_record["arrival_benchmark"] = arrival_benchmark
    tca_record["pending_write_sec"] = int(get_env("AI_TRADING_TCA_PENDING_WRITE_SEC", 60, cast=int))
    if allow_proxy_quotes:
        resolved_proxy_source = normalize_quote_source_token_func(submit_quote_source)
        if resolved_proxy_source is not None:
            tca_record["quote_proxy_source"] = resolved_proxy_source
        else:
            proxy_default = str(get_env("AI_TRADING_TCA_PROXY_MID_SOURCE", "last_trade"))
            tca_record["quote_proxy_source"] = resolve_quote_proxy_source_func(
                order,
                symbol=symbol,
                default_source=proxy_default,
            )
    metrics["tca"] = {
        "is_bps": tca_record.get("is_bps"),
        "spread_paid_bps": tca_record.get("spread_paid_bps"),
        "fill_latency_ms": tca_record.get("fill_latency_ms"),
    }
    tca_update_on_fill = bool(get_env("AI_TRADING_TCA_UPDATE_ON_FILL", True, cast=bool))
    tca_write_pending = bool(get_env("AI_TRADING_TCA_WRITE_PENDING_EVENTS", True, cast=bool))
    should_write_tca = bool(
        (tca_update_on_fill and persistable_fill)
        or (tca_write_pending and not persistable_fill)
    )
    if should_write_tca:
        resolved_tca_path = str(get_env("AI_TRADING_TCA_PATH", "runtime/tca_records.jsonl"))
        try:
            resolved_tca_path = str(resolved_tca_path_func())
            tca_payload = dict(tca_record)
            if not persistable_fill:
                tca_payload["pending_event"] = True
                tca_payload["pending_reason"] = order_state.status_token or "no_fill"
                tca_payload["order_status"] = order_state.status_text
            write_tca_record_func(resolved_tca_path, tca_payload)
        except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
            logger.warning(
                "TCA_WRITE_FAILED path=%s error=%s",
                resolved_tca_path,
                str(exc),
                extra={"error": str(exc), "path": resolved_tca_path},
            )
    return metrics, tca_record
