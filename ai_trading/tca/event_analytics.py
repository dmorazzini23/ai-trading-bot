"""Event-driven TCA analytics from immutable OMS events."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import math
from statistics import median
from typing import Any
from collections import defaultdict


def _parse_ts(raw: Any) -> datetime | None:
    if raw in (None, ""):
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _load_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload_raw = row.get("payload_json")
    if isinstance(payload_raw, str):
        try:
            decoded = json.loads(payload_raw)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            pass
    payload = row.get("payload")
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def _normalize_token(value: Any, *, default: str = "unknown") -> str:
    token = str(value or "").strip().lower()
    if not token:
        return default
    return token


def _extract_text(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _extract_scope(payload: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    kpi_scope_raw = payload.get("kpi_scope")
    kpi_scope = kpi_scope_raw if isinstance(kpi_scope_raw, dict) else {}

    symbol_text = _extract_text(
        payload,
        "symbol",
        "asset_symbol",
    )
    if symbol_text is None:
        symbol_text = _extract_text(kpi_scope, "symbol")
    symbol = str(symbol_text or "").strip().upper() or None

    strategy_text = _extract_text(
        payload,
        "strategy_id",
        "strategy",
    )
    if strategy_text is None:
        strategy_text = _extract_text(kpi_scope, "strategy_id", "strategy")
    strategy_id = str(strategy_text or "").strip() or None

    session_text = _extract_text(
        payload,
        "session_id",
        "session",
        "trading_session",
        "market_session",
    )
    if session_text is None:
        session_text = _extract_text(
            kpi_scope,
            "session_id",
            "session",
            "trading_session",
            "market_session",
        )
    session_id = str(session_text or "").strip() or None
    return symbol, strategy_id, session_id


def _top_count_rows(counter: dict[str, int], *, limit: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for token, count in sorted(
        counter.items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )[: max(1, int(limit))]:
        rows.append({"reason": str(token), "count": int(count)})
    return rows


def summarize_oms_event_tca(
    *,
    database_url: str | None = None,
    intent_store_path: str | None = None,
    lookback_days: int | None = None,
    limit: int = 50000,
) -> dict[str, Any]:
    """Summarize execution quality metrics directly from immutable OMS events."""

    from ai_trading.oms.event_store import EventStore

    event_store = EventStore(path=intent_store_path, url=database_url)
    cutoff = None
    if lookback_days is not None and int(lookback_days) > 0:
        cutoff = datetime.now(UTC) - timedelta(days=int(lookback_days))

    slippage_samples: list[float] = []
    fills_notional = 0.0
    fills_count = 0
    submit_ack_count = 0
    submit_reject_count = 0
    order_reject_count = 0
    order_cancel_count = 0
    order_filled_count = 0
    order_partially_filled_count = 0
    scanned = 0
    decision_count = 0
    decision_inserted_count = 0
    parent_execution_summary_events = 0
    parent_scope_metrics: dict[tuple[str, str, str], dict[str, Any]] = {}
    reject_reason_counts: dict[str, int] = defaultdict(int)
    cancel_reason_counts: dict[str, int] = defaultdict(int)
    slippage_adverse_count = 0
    slippage_favorable_count = 0
    slippage_adverse_sum = 0.0
    slippage_favorable_sum = 0.0
    outcomes_by_scope: dict[tuple[str, str, str], dict[str, Any]] = {}

    try:
        rows = event_store.list_oms_events(limit=max(1, int(limit)))
        for row in rows:
            scanned += 1
            event_ts = _parse_ts(row.get("event_ts"))
            if cutoff is not None and (event_ts is None or event_ts < cutoff):
                continue
            event_type = str(row.get("event_type") or "").strip().upper()
            payload = _load_payload(row)

            if event_type == "SUBMIT_ACK":
                submit_ack_count += 1
                symbol, strategy_id, session_id = _extract_scope(payload)
                if symbol is not None or strategy_id is not None or session_id is not None:
                    key = (
                        str(symbol or "UNKNOWN").upper(),
                        str(strategy_id or "unknown"),
                        str(session_id or "unknown"),
                    )
                    scope_bucket = outcomes_by_scope.setdefault(
                        key,
                        {
                            "symbol": key[0],
                            "strategy_id": key[1],
                            "session_id": key[2],
                            "submit_ack_events": 0,
                            "submit_reject_events": 0,
                            "order_reject_events": 0,
                            "order_cancel_events": 0,
                            "fill_events": 0,
                            "slippage_sample_count": 0,
                            "slippage_bps_sum": 0.0,
                            "adverse_slippage_samples": 0,
                        },
                    )
                    scope_bucket["submit_ack_events"] = int(scope_bucket["submit_ack_events"]) + 1
            elif event_type == "SUBMIT_REJECT":
                submit_reject_count += 1
                reject_reason = _normalize_token(
                    _extract_text(
                        payload,
                        "rejection_reason",
                        "error",
                        "error_code",
                        "code",
                        "reason",
                        "last_error",
                        "message",
                    ),
                    default="unknown",
                )
                reject_reason_counts[reject_reason] += 1
                symbol, strategy_id, session_id = _extract_scope(payload)
                if symbol is not None or strategy_id is not None or session_id is not None:
                    key = (
                        str(symbol or "UNKNOWN").upper(),
                        str(strategy_id or "unknown"),
                        str(session_id or "unknown"),
                    )
                    scope_bucket = outcomes_by_scope.setdefault(
                        key,
                        {
                            "symbol": key[0],
                            "strategy_id": key[1],
                            "session_id": key[2],
                            "submit_ack_events": 0,
                            "submit_reject_events": 0,
                            "order_reject_events": 0,
                            "order_cancel_events": 0,
                            "fill_events": 0,
                            "slippage_sample_count": 0,
                            "slippage_bps_sum": 0.0,
                            "adverse_slippage_samples": 0,
                        },
                    )
                    scope_bucket["submit_reject_events"] = int(scope_bucket["submit_reject_events"]) + 1
            elif event_type == "ORDER_REJECTED":
                order_reject_count += 1
                reject_reason = _normalize_token(
                    _extract_text(
                        payload,
                        "rejection_reason",
                        "last_error",
                        "error",
                        "error_code",
                        "code",
                        "reason",
                        "message",
                    ),
                    default="unknown",
                )
                reject_reason_counts[reject_reason] += 1
                symbol, strategy_id, session_id = _extract_scope(payload)
                if symbol is not None or strategy_id is not None or session_id is not None:
                    key = (
                        str(symbol or "UNKNOWN").upper(),
                        str(strategy_id or "unknown"),
                        str(session_id or "unknown"),
                    )
                    scope_bucket = outcomes_by_scope.setdefault(
                        key,
                        {
                            "symbol": key[0],
                            "strategy_id": key[1],
                            "session_id": key[2],
                            "submit_ack_events": 0,
                            "submit_reject_events": 0,
                            "order_reject_events": 0,
                            "order_cancel_events": 0,
                            "fill_events": 0,
                            "slippage_sample_count": 0,
                            "slippage_bps_sum": 0.0,
                            "adverse_slippage_samples": 0,
                        },
                    )
                    scope_bucket["order_reject_events"] = int(scope_bucket["order_reject_events"]) + 1
            elif event_type == "ORDER_CANCELED":
                order_cancel_count += 1
                cancel_reason = _normalize_token(
                    _extract_text(
                        payload,
                        "reason",
                        "last_error",
                        "cancel_reason",
                        "message",
                        "final_status",
                    ),
                    default="unknown",
                )
                cancel_reason_counts[cancel_reason] += 1
                symbol, strategy_id, session_id = _extract_scope(payload)
                if symbol is not None or strategy_id is not None or session_id is not None:
                    key = (
                        str(symbol or "UNKNOWN").upper(),
                        str(strategy_id or "unknown"),
                        str(session_id or "unknown"),
                    )
                    scope_bucket = outcomes_by_scope.setdefault(
                        key,
                        {
                            "symbol": key[0],
                            "strategy_id": key[1],
                            "session_id": key[2],
                            "submit_ack_events": 0,
                            "submit_reject_events": 0,
                            "order_reject_events": 0,
                            "order_cancel_events": 0,
                            "fill_events": 0,
                            "slippage_sample_count": 0,
                            "slippage_bps_sum": 0.0,
                            "adverse_slippage_samples": 0,
                        },
                    )
                    scope_bucket["order_cancel_events"] = int(scope_bucket["order_cancel_events"]) + 1
            elif event_type in {"ORDER_FILLED", "ORDER_PARTIALLY_FILLED"}:
                fills_count += 1
                if event_type == "ORDER_FILLED":
                    order_filled_count += 1
                else:
                    order_partially_filled_count += 1
                qty = _as_float(payload.get("fill_qty"))
                if qty is None:
                    qty = _as_float(payload.get("qty"))
                price = _as_float(payload.get("fill_price"))
                if price is None:
                    price = _as_float(payload.get("price"))
                expected_price = _as_float(payload.get("expected_price"))
                if qty is not None and price is not None and qty > 0 and price > 0:
                    fills_notional += abs(float(qty) * float(price))
                if (
                    expected_price is not None
                    and price is not None
                    and expected_price > 0
                    and price > 0
                ):
                    slippage_bps = ((price - expected_price) / expected_price) * 10000.0
                    slippage_samples.append(float(slippage_bps))
                    if float(slippage_bps) > 0.0:
                        slippage_adverse_count += 1
                        slippage_adverse_sum += float(slippage_bps)
                    elif float(slippage_bps) < 0.0:
                        slippage_favorable_count += 1
                        slippage_favorable_sum += abs(float(slippage_bps))
                    symbol, strategy_id, session_id = _extract_scope(payload)
                    if symbol is not None or strategy_id is not None or session_id is not None:
                        key = (
                            str(symbol or "UNKNOWN").upper(),
                            str(strategy_id or "unknown"),
                            str(session_id or "unknown"),
                        )
                        scope_bucket = outcomes_by_scope.setdefault(
                            key,
                            {
                                "symbol": key[0],
                                "strategy_id": key[1],
                                "session_id": key[2],
                                "submit_ack_events": 0,
                                "submit_reject_events": 0,
                                "order_reject_events": 0,
                                "order_cancel_events": 0,
                                "fill_events": 0,
                                "slippage_sample_count": 0,
                                "slippage_bps_sum": 0.0,
                                "adverse_slippage_samples": 0,
                            },
                        )
                        scope_bucket["fill_events"] = int(scope_bucket["fill_events"]) + 1
                        scope_bucket["slippage_sample_count"] = (
                            int(scope_bucket["slippage_sample_count"]) + 1
                        )
                        scope_bucket["slippage_bps_sum"] = float(scope_bucket["slippage_bps_sum"]) + float(
                            slippage_bps
                        )
                        if float(slippage_bps) > 0.0:
                            scope_bucket["adverse_slippage_samples"] = (
                                int(scope_bucket["adverse_slippage_samples"]) + 1
                            )
            elif event_type == "RECONCILE_UPDATE":
                record_type = str(payload.get("record_type") or "").strip().lower()
                if record_type != "parent_execution_summary":
                    continue
                parent_execution_summary_events += 1
                symbol = str(payload.get("symbol") or "").strip().upper() or "UNKNOWN"
                strategy_id = str(payload.get("strategy_id") or "").strip() or "unknown"
                session_id = str(payload.get("session_id") or "").strip() or "unknown"
                key = (symbol, strategy_id, session_id)
                bucket = parent_scope_metrics.setdefault(
                    key,
                    {
                        "symbol": symbol,
                        "strategy_id": strategy_id,
                        "session_id": session_id,
                        "parent_orders": 0,
                        "requested_quantity": 0.0,
                        "submitted_quantity": 0.0,
                        "failed_slices": 0,
                        "retry_count": 0,
                        "cancel_replace_count": 0,
                        "success_ratio_sum": 0.0,
                        "arrival_slippage_sum": 0.0,
                        "arrival_slippage_count": 0,
                    },
                )
                bucket["parent_orders"] = int(bucket["parent_orders"]) + 1
                bucket["requested_quantity"] = float(bucket["requested_quantity"]) + float(
                    _as_float(payload.get("requested_quantity")) or 0.0
                )
                bucket["submitted_quantity"] = float(bucket["submitted_quantity"]) + float(
                    _as_float(payload.get("submitted_quantity")) or 0.0
                )
                bucket["failed_slices"] = int(bucket["failed_slices"]) + int(
                    _as_float(payload.get("failed_slices")) or 0
                )
                bucket["retry_count"] = int(bucket["retry_count"]) + int(
                    _as_float(payload.get("retry_count")) or 0
                )
                bucket["cancel_replace_count"] = int(bucket["cancel_replace_count"]) + int(
                    _as_float(payload.get("cancel_replace_count")) or 0
                )
                bucket["success_ratio_sum"] = float(bucket["success_ratio_sum"]) + float(
                    _as_float(payload.get("success_ratio")) or 0.0
                )
                arrival_mean = _as_float(payload.get("arrival_slippage_bps_mean"))
                arrival_samples = int(_as_float(payload.get("arrival_slippage_sample_count")) or 0)
                if arrival_mean is not None and arrival_samples > 0:
                    bucket["arrival_slippage_sum"] = float(bucket["arrival_slippage_sum"]) + (
                        float(arrival_mean) * float(arrival_samples)
                    )
                    bucket["arrival_slippage_count"] = int(bucket["arrival_slippage_count"]) + int(
                        arrival_samples
                    )

        decision_rows = event_store.list_decision_events(limit=max(1, int(limit)))
        for row in decision_rows:
            decision_count += 1
            decision_ts = _parse_ts(row.get("decision_ts"))
            if cutoff is not None and (decision_ts is None or decision_ts < cutoff):
                continue
            decision_inserted_count += 1
    finally:
        event_store.close()

    sorted_slippage = sorted(slippage_samples)
    p90_slippage = 0.0
    if sorted_slippage:
        index = int(math.ceil(len(sorted_slippage) * 0.90)) - 1
        index = max(0, min(index, len(sorted_slippage) - 1))
        p90_slippage = float(sorted_slippage[index])
    reject_rate = (
        (float(submit_reject_count) / float(submit_ack_count + submit_reject_count)) * 100.0
        if (submit_ack_count + submit_reject_count) > 0
        else 0.0
    )
    parent_execution_kpis_by_scope: list[dict[str, Any]] = []
    for bucket in parent_scope_metrics.values():
        parent_orders = max(1, int(bucket["parent_orders"]))
        slippage_count = int(bucket["arrival_slippage_count"])
        parent_execution_kpis_by_scope.append(
            {
                "symbol": bucket["symbol"],
                "strategy_id": bucket["strategy_id"],
                "session_id": bucket["session_id"],
                "parent_orders": parent_orders,
                "requested_quantity": float(bucket["requested_quantity"]),
                "submitted_quantity": float(bucket["submitted_quantity"]),
                "failed_slices": int(bucket["failed_slices"]),
                "retry_count": int(bucket["retry_count"]),
                "cancel_replace_count": int(bucket["cancel_replace_count"]),
                "avg_success_ratio": float(bucket["success_ratio_sum"]) / float(parent_orders),
                "avg_fill_ratio": (
                    float(bucket["submitted_quantity"]) / float(bucket["requested_quantity"])
                    if float(bucket["requested_quantity"]) > 0.0
                    else 0.0
                ),
                "avg_arrival_slippage_bps": (
                    float(bucket["arrival_slippage_sum"]) / float(slippage_count)
                    if slippage_count > 0
                    else 0.0
                ),
                "arrival_slippage_sample_count": slippage_count,
            }
        )
    parent_execution_kpis_by_scope.sort(
        key=lambda row: (
            -int(row["parent_orders"]),
            str(row["symbol"]),
            str(row["strategy_id"]),
            str(row["session_id"]),
        )
    )
    outcomes_by_scope_rows: list[dict[str, Any]] = []
    for bucket in outcomes_by_scope.values():
        slippage_count = int(bucket["slippage_sample_count"])
        outcomes_by_scope_rows.append(
            {
                "symbol": str(bucket["symbol"]),
                "strategy_id": str(bucket["strategy_id"]),
                "session_id": str(bucket["session_id"]),
                "submit_ack_events": int(bucket["submit_ack_events"]),
                "submit_reject_events": int(bucket["submit_reject_events"]),
                "order_reject_events": int(bucket["order_reject_events"]),
                "order_cancel_events": int(bucket["order_cancel_events"]),
                "fill_events": int(bucket["fill_events"]),
                "slippage_sample_count": slippage_count,
                "avg_realized_slippage_bps": (
                    float(bucket["slippage_bps_sum"]) / float(slippage_count)
                    if slippage_count > 0
                    else 0.0
                ),
                "adverse_slippage_rate": (
                    float(bucket["adverse_slippage_samples"]) / float(slippage_count)
                    if slippage_count > 0
                    else 0.0
                ),
            }
        )
    outcomes_by_scope_rows.sort(
        key=lambda row: (
            -int(
                int(row["submit_ack_events"])
                + int(row["submit_reject_events"])
                + int(row["order_reject_events"])
                + int(row["order_cancel_events"])
                + int(row["fill_events"])
            ),
            str(row["symbol"]),
            str(row["strategy_id"]),
            str(row["session_id"]),
        )
    )
    total_submit_events = int(submit_ack_count + submit_reject_count)
    total_reject_cancel_events = int(submit_reject_count + order_reject_count + order_cancel_count)
    return {
        "scanned_oms_events": int(scanned),
        "filled_events": int(fills_count),
        "submit_ack_events": int(submit_ack_count),
        "submit_reject_events": int(submit_reject_count),
        "order_reject_events": int(order_reject_count),
        "order_cancel_events": int(order_cancel_count),
        "order_filled_events": int(order_filled_count),
        "order_partially_filled_events": int(order_partially_filled_count),
        "submit_reject_rate_pct": float(reject_rate),
        "cancel_to_submit_ack_rate_pct": (
            float(order_cancel_count) / float(submit_ack_count) * 100.0
            if int(submit_ack_count) > 0
            else 0.0
        ),
        "reject_cancel_rate_pct": (
            float(total_reject_cancel_events) / float(total_submit_events) * 100.0
            if int(total_submit_events) > 0
            else 0.0
        ),
        "fill_notional": float(fills_notional),
        "avg_fill_notional": (
            float(fills_notional / fills_count) if fills_count > 0 else 0.0
        ),
        "slippage_sample_count": int(len(slippage_samples)),
        "median_slippage_bps": float(median(sorted_slippage)) if sorted_slippage else 0.0,
        "p90_slippage_bps": float(p90_slippage),
        "realized_slippage_decomposition": {
            "sample_count": int(len(slippage_samples)),
            "adverse_sample_count": int(slippage_adverse_count),
            "favorable_sample_count": int(slippage_favorable_count),
            "adverse_share": (
                float(slippage_adverse_count) / float(len(slippage_samples))
                if len(slippage_samples) > 0
                else 0.0
            ),
            "mean_adverse_bps": (
                float(slippage_adverse_sum) / float(slippage_adverse_count)
                if slippage_adverse_count > 0
                else 0.0
            ),
            "mean_favorable_bps": (
                float(slippage_favorable_sum) / float(slippage_favorable_count)
                if slippage_favorable_count > 0
                else 0.0
            ),
        },
        "submit_reject_reasons_top": _top_count_rows(reject_reason_counts, limit=5),
        "cancel_reasons_top": _top_count_rows(cancel_reason_counts, limit=5),
        "event_outcomes_by_scope": outcomes_by_scope_rows[:10],
        "parent_execution_summary_events": int(parent_execution_summary_events),
        "parent_execution_kpis_by_scope": parent_execution_kpis_by_scope,
        "decision_events_scanned": int(decision_count),
        "decision_events_in_window": int(decision_inserted_count),
        "lookback_days": int(lookback_days) if lookback_days is not None else None,
    }


__all__ = ["summarize_oms_event_tca"]
