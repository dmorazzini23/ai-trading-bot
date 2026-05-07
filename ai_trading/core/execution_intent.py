"""Helpers for building pre-submit execution intent context."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from ai_trading.oms.ledger import deterministic_client_order_id
from ai_trading.oms.pretrade import OrderIntent as PretradeOrderIntent


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _session_regime_from_ts(value: datetime | None) -> str:
    if value is None:
        return "offhours"
    ts = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    eastern = ts.astimezone(ZoneInfo("America/New_York"))
    minute = eastern.hour * 60 + eastern.minute
    if eastern.weekday() >= 5 or minute < (9 * 60 + 30) or minute >= (16 * 60):
        return "offhours"
    if minute < (10 * 60 + 15):
        return "opening"
    if minute >= (15 * 60 + 15):
        return "closing"
    return "midday"


def _regime_context_from_snapshot(
    config_snapshot: Mapping[str, Any],
    *,
    bar_ts: datetime | None,
) -> dict[str, str]:
    session_regime = (
        _safe_text(config_snapshot.get("session_regime"))
        or _safe_text(config_snapshot.get("session_bucket"))
        or _session_regime_from_ts(bar_ts)
    )
    regime_profile = (
        _safe_text(config_snapshot.get("regime_profile"))
        or _safe_text(config_snapshot.get("regime_signal_profile"))
    )
    market_regime = (
        _safe_text(config_snapshot.get("market_regime"))
        or _safe_text(config_snapshot.get("current_regime"))
        or regime_profile
    )
    volatility_regime = _safe_text(config_snapshot.get("volatility_regime"))
    trend_regime = _safe_text(config_snapshot.get("trend_regime"))
    context = {"session_regime": session_regime}
    if market_regime:
        context["market_regime"] = market_regime
    if regime_profile:
        context["regime_profile"] = regime_profile
    if volatility_regime:
        context["volatility_regime"] = volatility_regime
    if trend_regime:
        context["trend_regime"] = trend_regime
    return context


@dataclass(frozen=True, slots=True)
class ExecutionIntentContext:
    client_order_id: str
    decision_trace_id: str
    pretrade_intent: PretradeOrderIntent
    order_lineage_metadata: dict[str, Any] = field(default_factory=dict)
    order_annotations: dict[str, Any] = field(default_factory=dict)


def build_execution_intent_context(
    *,
    now: datetime,
    salt: str,
    symbol: str,
    side: str,
    delta_shares: int,
    price: float,
    bar_ts: datetime,
    spread_bps: float,
    liquidity_bucket: str,
    quote_quality_ok: bool,
    sector: str | None,
    event_risk: bool,
    slo_derisk_details: Mapping[str, Any],
    config_snapshot: Mapping[str, Any],
    execution_model_lineage: Mapping[str, Any],
    submit_quote_source: str | None,
    submit_bid_at_arrival: float | None,
    submit_ask_at_arrival: float | None,
    submit_mid_at_arrival: float | None,
    submit_quote_ts: datetime | None,
    opening_trade: bool,
    require_realtime_nbbo: bool,
    kill_switch_active: bool,
    kill_switch_reason: str | None,
    broker_ready: bool,
    broker_ready_reason: str | None,
    broker_cooldown_remaining_sec: float | None,
) -> ExecutionIntentContext:
    """Build pretrade intent and broker annotation context for a candidate."""
    client_order_id = deterministic_client_order_id(
        salt=salt,
        symbol=symbol,
        bar_ts=bar_ts.isoformat(),
        side=side,
        qty=abs(delta_shares),
        limit_price=price,
    )
    decision_trace_id = str(client_order_id or "").strip() or f"{symbol}:{bar_ts.isoformat()}:decision_trace"
    now_utc = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    quote_ts = submit_quote_ts if submit_quote_ts is None else (
        submit_quote_ts if submit_quote_ts.tzinfo is not None else submit_quote_ts.replace(tzinfo=UTC)
    )
    quote_age_ms = None
    if quote_ts is not None:
        quote_age_ms = max(
            (now_utc.astimezone(UTC) - quote_ts.astimezone(UTC)).total_seconds() * 1000.0,
            0.0,
        )
    derived_mid = submit_mid_at_arrival if submit_mid_at_arrival is not None else float(price)
    derived_last = (
        submit_ask_at_arrival
        if str(side or "").strip().lower() in {"buy", "cover"} and submit_ask_at_arrival is not None
        else submit_bid_at_arrival
        if submit_bid_at_arrival is not None
        else float(price)
    )
    derived_spread = None
    if submit_bid_at_arrival is not None and submit_ask_at_arrival is not None:
        spread_candidate = float(submit_ask_at_arrival) - float(submit_bid_at_arrival)
        if spread_candidate >= 0.0:
            derived_spread = spread_candidate
    if derived_spread is None:
        derived_spread = (float(spread_bps) / 10_000.0) * float(price)
    regime_context = _regime_context_from_snapshot(config_snapshot, bar_ts=bar_ts)

    pretrade_intent = PretradeOrderIntent(
        symbol=symbol,
        side=side,
        qty=abs(delta_shares),
        notional=abs(delta_shares) * price,
        limit_price=price,
        bar_ts=bar_ts,
        client_order_id=client_order_id,
        last_price=derived_last,
        mid=derived_mid,
        bid=submit_bid_at_arrival,
        ask=submit_ask_at_arrival,
        spread=derived_spread,
        quote_ts=quote_ts,
        quote_age_ms=quote_age_ms,
        submit_quote_source=submit_quote_source,
        avg_daily_volume=max(
            float(_safe_float(slo_derisk_details.get("rolling_volume")) or 0.0) * 390.0,
            0.0,
        ),
        minute_volume=max(
            float(_safe_float(slo_derisk_details.get("rolling_volume")) or 0.0),
            0.0,
        ),
        liquidity_bucket=str(liquidity_bucket or "").upper() or None,
        quote_quality_ok=quote_quality_ok,
        sector=sector,
        event_risk=event_risk,
        event_type="earnings" if event_risk else None,
        execution_drift_bps=float(
            _safe_float(slo_derisk_details.get("execution_drift_bps")) or 0.0
        ),
        reject_rate_pct=float(_safe_float(slo_derisk_details.get("reject_rate_pct")) or 0.0),
        session_regime=regime_context.get("session_regime"),
        opening_trade=bool(opening_trade),
        require_realtime_nbbo=bool(require_realtime_nbbo),
        kill_switch_active=bool(kill_switch_active),
        kill_switch_reason=kill_switch_reason,
        broker_ready=bool(broker_ready),
        broker_ready_reason=broker_ready_reason,
        broker_cooldown_remaining_sec=broker_cooldown_remaining_sec,
    )

    order_lineage_metadata: dict[str, Any] = {}
    for source_key in (
        "model_id",
        "model_version",
        "dataset_hash",
        "feature_version",
        "model_artifact_hash",
    ):
        text = str(execution_model_lineage.get(source_key) or "").strip()
        if text:
            order_lineage_metadata[source_key] = text

    config_snapshot_hash = str(config_snapshot.get("config_snapshot_hash") or "").strip()
    if config_snapshot_hash:
        order_lineage_metadata["config_snapshot_hash"] = config_snapshot_hash
    policy_hash = str(config_snapshot.get("effective_policy_hash") or "").strip()
    if policy_hash:
        order_lineage_metadata["policy_hash"] = policy_hash
    if decision_trace_id:
        order_lineage_metadata["decision_trace_id"] = decision_trace_id
    if submit_quote_source:
        order_lineage_metadata["price_source"] = submit_quote_source
    order_lineage_metadata.update(regime_context)

    order_annotations: dict[str, Any] = {}
    if submit_quote_source:
        order_annotations["price_source"] = submit_quote_source
    if policy_hash:
        order_annotations["policy_hash"] = policy_hash
    if decision_trace_id:
        order_annotations["decision_trace_id"] = decision_trace_id
    order_annotations.update(regime_context)
    if (
        submit_bid_at_arrival is not None
        and submit_ask_at_arrival is not None
        and submit_mid_at_arrival is not None
    ):
        order_annotations["quote_source"] = "broker_nbbo"
        order_annotations["quote"] = {
            "bid": float(submit_bid_at_arrival),
            "ask": float(submit_ask_at_arrival),
            "midpoint": float(submit_mid_at_arrival),
            "source": "broker_nbbo",
            "synthetic": False,
        }
        if quote_ts is not None:
            order_annotations["quote"]["ts"] = quote_ts.isoformat()
        if quote_age_ms is not None:
            order_annotations["quote"]["quote_age_ms"] = float(quote_age_ms)
    elif quote_ts is not None or quote_age_ms is not None:
        order_annotations["quote"] = {
            "source": submit_quote_source or "unknown",
            "ts": quote_ts.isoformat() if quote_ts is not None else None,
            "quote_age_ms": float(quote_age_ms) if quote_age_ms is not None else None,
        }

    return ExecutionIntentContext(
        client_order_id=client_order_id,
        decision_trace_id=decision_trace_id,
        pretrade_intent=pretrade_intent,
        order_lineage_metadata=order_lineage_metadata,
        order_annotations=order_annotations,
    )
