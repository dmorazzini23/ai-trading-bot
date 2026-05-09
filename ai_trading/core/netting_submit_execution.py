"""Broker submit outcome handling for the live netting cycle."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping

from ai_trading.config.management import get_env
from ai_trading.core.errors import ErrorCategory

_NON_ACCEPTED_ORDER_STATUSES = frozenset(
    {
        "rejected",
        "canceled",
        "cancelled",
        "expired",
        "done_for_day",
        "skipped",
        "failed",
        "error",
    }
)


def _is_short_reducing_buy(side: str, delta_shares: int, net_target: Any) -> bool:
    side_token = str(side or "").strip().lower()
    if side_token in {"cover", "buy_to_cover", "buy-to-cover", "buy to cover"}:
        return True
    if side_token != "buy" or int(delta_shares) <= 0:
        return False
    target_shares = getattr(net_target, "target_shares", None)
    if target_shares is None:
        return False
    try:
        return int(target_shares) <= 0
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class NettingSubmitExecutionResult:
    status: str
    gates_added: tuple[str, ...]
    attempted_increment: int
    submitted_increment: int
    metrics: dict[str, Any] | None
    tca_record: dict[str, Any] | None
    order_payload: dict[str, Any] | None
    decision_trace_id: str | None
    order_intent_contract: Any | None


def execute_netting_submission(
    *,
    runtime: Any,
    state: Any,
    symbol: str,
    side: str,
    price: float,
    delta_shares: int,
    now: datetime,
    net_target: Any,
    approval: Any,
    intent: Any,
    client_order_id: str,
    decision_trace_id_for_order: str | None,
    model_id_for_order: str,
    model_version_for_order: str,
    config_snapshot_hash_for_order: str,
    dataset_hash_for_order: str,
    feature_version_for_order: str,
    model_artifact_hash_for_order: str,
    policy_hash_for_order: str,
    order_annotations: Mapping[str, Any],
    order_lineage_metadata: Mapping[str, Any],
    submit_arrival_price: float | None,
    submit_bid_at_arrival: float | None,
    submit_ask_at_arrival: float | None,
    submit_mid_at_arrival: float | None,
    submit_quote_source: str | None,
    candidate_expected_net_edge: Mapping[str, Any],
    candidate_expected_capture: Mapping[str, Any],
    ledger: Any,
    quarantine_enabled: bool,
    quarantine_manager: Any,
    extract_order_value_func: Callable[..., Any],
    extract_order_fill_timestamp_func: Callable[[Any], datetime | None],
    normalize_order_status_token_func: Callable[[Any], str],
    safe_float: Callable[[Any], float | None],
    has_persistable_fill_func: Callable[..., bool],
    normalize_submitted_order_func: Callable[..., Any],
    record_successful_submission_func: Callable[..., None],
    build_order_metrics_and_tca_func: Callable[..., tuple[dict[str, Any], dict[str, Any] | None]],
    submit_order_func: Callable[..., Any],
    classify_exception_func: Callable[..., Any],
    handle_error_func: Callable[..., None],
    trigger_quarantine_func: Callable[..., None],
    cancel_all_open_orders_oms_func: Callable[[Any], Any],
    resolve_submit_none_reason_func: Callable[[Any], str],
    record_auth_forbidden_cooldown_func: Callable[..., None],
    get_regime_signal_profile_func: Callable[[], str],
    normalize_quote_source_token_func: Callable[[Any], str | None],
    resolve_quote_proxy_source_func: Callable[..., str | None],
    resolved_tca_path_func: Callable[[], Any],
    write_tca_record_func: Callable[[str, Mapping[str, Any]], None],
    session_bucket_from_ts_func: Callable[[datetime], str],
    compute_attribution_metrics_func: Callable[..., dict[str, Any]],
    logger: Any,
    breakers: Any,
) -> NettingSubmitExecutionResult:
    closing_short = _is_short_reducing_buy(side, int(delta_shares), net_target)
    submit_extra: dict[str, Any] = {}
    if closing_short:
        submit_extra["closing_position"] = True
        submit_extra["reduce_only"] = True
    try:
        order = submit_order_func(
            runtime,
            symbol,
            abs(delta_shares),
            side,
            price=price,
            client_order_id=client_order_id,
            expected_net_edge_bps=float(approval.expected_net_edge_bps),
            model_id=model_id_for_order or None,
            model_version=model_version_for_order or None,
            config_snapshot_hash=config_snapshot_hash_for_order or None,
            dataset_hash=dataset_hash_for_order or None,
            feature_version=feature_version_for_order or None,
            model_artifact_hash=model_artifact_hash_for_order or None,
            policy_hash=policy_hash_for_order or None,
            decision_trace_id=decision_trace_id_for_order or None,
            annotations=(dict(order_annotations) or None),
            price_hint=float(submit_arrival_price) if submit_arrival_price is not None else float(price),
            metadata=(dict(order_lineage_metadata) or None),
            **submit_extra,
        )
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        error_info = classify_exception_func(exc, dependency="broker_submit", symbol=symbol)
        breakers.record_failure("broker_submit", error_info)
        handle_error_func(error_info, state=state, ctx=runtime, symbol=symbol)
        submit_open_reason = breakers.open_reason("broker_submit")
        if quarantine_enabled and quarantine_manager is not None:
            reject_counts = getattr(state, "_quarantine_reject_counts", None)
            if not isinstance(reject_counts, dict):
                reject_counts = {}
                setattr(state, "_quarantine_reject_counts", reject_counts)
            if error_info.category is ErrorCategory.ORDER_REJECTED:
                reject_counts[symbol] = int(reject_counts.get(symbol, 0)) + 1
                reject_trigger = float(get_env("AI_TRADING_QUARANTINE_REJECT_RATE_TRIGGER", 0.15, cast=float))
                min_trades = int(get_env("AI_TRADING_QUARANTINE_MIN_TRADES", 12, cast=int))
                reject_threshold = max(1, int(round(reject_trigger * min_trades)))
                if reject_counts[symbol] >= reject_threshold:
                    trigger_quarantine_func(
                        manager=quarantine_manager,
                        symbol=symbol,
                        sleeve=net_target.proposals[0].sleeve if net_target.proposals else None,
                        reason="ORDER_REJECT_RATE",
                        metrics_snapshot={
                            "reject_count": reject_counts[symbol],
                            "threshold": reject_threshold,
                            "trigger": reject_trigger,
                        },
                    )
            breaker_threshold = int(get_env("AI_TRADING_QUARANTINE_BREAKER_OPEN_COUNT", 2, cast=int))
            if submit_open_reason and breaker_threshold > 0:
                breaker_counts = getattr(state, "_quarantine_breaker_open_counts", None)
                if not isinstance(breaker_counts, dict):
                    breaker_counts = {}
                    setattr(state, "_quarantine_breaker_open_counts", breaker_counts)
                breaker_counts[symbol] = int(breaker_counts.get(symbol, 0)) + 1
                if breaker_counts[symbol] >= breaker_threshold:
                    trigger_quarantine_func(
                        manager=quarantine_manager,
                        symbol=symbol,
                        sleeve=net_target.proposals[0].sleeve if net_target.proposals else None,
                        reason="BREAKER_OPEN_COUNT",
                        metrics_snapshot={
                            "breaker_open_count": breaker_counts[symbol],
                            "threshold": breaker_threshold,
                            "breaker_reason": submit_open_reason,
                        },
                    )
        if submit_open_reason and bool(get_env("AI_TRADING_CANCEL_ALL_ON_SUBMIT_BREAKER", False, cast=bool)):
            cancel_result = cancel_all_open_orders_oms_func(runtime)
            logger.warning(
                "CANCEL_ALL_TRIGGERED",
                extra={
                    "reason_code": submit_open_reason,
                    "cancelled": cancel_result.cancelled,
                    "failed": cancel_result.failed,
                },
            )
        reason_code = str(error_info.reason_code)
        record_auth_forbidden_cooldown_func(
            state,
            symbol=symbol,
            side=side,
            reason=reason_code,
            now=now,
        )
        return NettingSubmitExecutionResult(
            status="blocked",
            gates_added=(reason_code,),
            attempted_increment=1,
            submitted_increment=0,
            metrics=None,
            tca_record=None,
            order_payload=None,
            decision_trace_id=None,
            order_intent_contract=intent.to_contract(),
        )

    if order is None:
        submit_none_reason = resolve_submit_none_reason_func(runtime)
        record_auth_forbidden_cooldown_func(
            state,
            symbol=symbol,
            side=side,
            reason=submit_none_reason,
            now=now,
        )
        attempted_increment = 0 if submit_none_reason in {"CYCLE_DUPLICATE_INTENT", "DUPLICATE_INTENT"} else 1
        return NettingSubmitExecutionResult(
            status="blocked",
            gates_added=(submit_none_reason,),
            attempted_increment=attempted_increment,
            submitted_increment=0,
            metrics=None,
            tca_record=None,
            order_payload=None,
            decision_trace_id=None,
            order_intent_contract=intent.to_contract(),
        )

    order_state = normalize_submitted_order_func(
        order,
        delta_shares=int(delta_shares),
        extract_order_value=extract_order_value_func,
        extract_order_fill_timestamp=extract_order_fill_timestamp_func,
        normalize_order_status_token=normalize_order_status_token_func,
        safe_float=safe_float,
        has_persistable_fill=has_persistable_fill_func,
    )
    status_token = str(getattr(order_state, "status_token", "") or "").strip().lower()
    if status_token in _NON_ACCEPTED_ORDER_STATUSES:
        reason_code = f"BROKER_ORDER_{status_token.upper()}".replace("CANCELLED", "CANCELED")
        record_auth_forbidden_cooldown_func(
            state,
            symbol=symbol,
            side=side,
            reason=reason_code,
            now=now,
        )
        metrics, tca_record = build_order_metrics_and_tca_func(
            symbol=symbol,
            side=side,
            price=float(price),
            delta_shares=int(delta_shares),
            now=now,
            net_target=net_target,
            order=order,
            order_state=order_state,
            submit_arrival_price=submit_arrival_price,
            submit_bid_at_arrival=submit_bid_at_arrival,
            submit_ask_at_arrival=submit_ask_at_arrival,
            submit_mid_at_arrival=submit_mid_at_arrival,
            submit_quote_source=submit_quote_source,
            candidate_expected_net_edge=candidate_expected_net_edge,
            candidate_expected_capture=candidate_expected_capture,
            get_regime_signal_profile_func=get_regime_signal_profile_func,
            normalize_quote_source_token_func=normalize_quote_source_token_func,
            resolve_quote_proxy_source_func=resolve_quote_proxy_source_func,
            resolved_tca_path_func=resolved_tca_path_func,
            write_tca_record_func=write_tca_record_func,
            session_bucket_from_ts_func=session_bucket_from_ts_func,
            compute_attribution_metrics_func=compute_attribution_metrics_func,
            safe_float=safe_float,
            logger=logger,
            order_lineage_metadata=order_lineage_metadata,
        )
        return NettingSubmitExecutionResult(
            status=status_token,
            gates_added=(reason_code,),
            attempted_increment=1,
            submitted_increment=0,
            metrics=metrics,
            tca_record=tca_record,
            order_payload={
                "client_order_id": client_order_id,
                "side": side,
                "qty": abs(delta_shares),
                "price": price,
                "status": getattr(order_state, "status_text", status_token) or status_token,
            },
            decision_trace_id=decision_trace_id_for_order,
            order_intent_contract=intent.to_contract(),
        )
    breakers.record_success("broker_submit")
    record_successful_submission_func(
        ledger=ledger,
        state=state,
        symbol=symbol,
        client_order_id=client_order_id,
        bar_ts=net_target.bar_ts,
        delta_shares=int(delta_shares),
        side=side,
        price=float(price),
        now=now,
        order_state=order_state,
        proposals=net_target.proposals,
    )
    metrics, tca_record = build_order_metrics_and_tca_func(
        symbol=symbol,
        side=side,
        price=float(price),
        delta_shares=int(delta_shares),
        now=now,
        net_target=net_target,
        order=order,
        order_state=order_state,
        submit_arrival_price=submit_arrival_price,
        submit_bid_at_arrival=submit_bid_at_arrival,
        submit_ask_at_arrival=submit_ask_at_arrival,
        submit_mid_at_arrival=submit_mid_at_arrival,
        submit_quote_source=submit_quote_source,
        candidate_expected_net_edge=candidate_expected_net_edge,
        candidate_expected_capture=candidate_expected_capture,
        get_regime_signal_profile_func=get_regime_signal_profile_func,
        normalize_quote_source_token_func=normalize_quote_source_token_func,
        resolve_quote_proxy_source_func=resolve_quote_proxy_source_func,
        resolved_tca_path_func=resolved_tca_path_func,
        write_tca_record_func=write_tca_record_func,
        session_bucket_from_ts_func=session_bucket_from_ts_func,
        compute_attribution_metrics_func=compute_attribution_metrics_func,
        safe_float=safe_float,
        logger=logger,
        order_lineage_metadata=order_lineage_metadata,
    )
    return NettingSubmitExecutionResult(
        status="submitted",
        gates_added=("OK_TRADE",),
        attempted_increment=1,
        submitted_increment=1,
        metrics=metrics,
        tca_record=tca_record,
        order_payload={
            "client_order_id": client_order_id,
            "side": side,
            "qty": abs(delta_shares),
            "price": price,
            "status": order_state.status_text if order_state.status_text else None,
        },
        decision_trace_id=decision_trace_id_for_order,
        order_intent_contract=intent.to_contract(),
    )
