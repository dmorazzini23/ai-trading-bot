"""Helpers for building pre-submit execution intent context."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from ai_trading.oms.ledger import deterministic_client_order_id
from ai_trading.oms.pretrade import OrderIntent as PretradeOrderIntent


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class ExecutionIntentContext:
    client_order_id: str
    decision_trace_id: str
    pretrade_intent: PretradeOrderIntent
    order_lineage_metadata: dict[str, Any] = field(default_factory=dict)
    order_annotations: dict[str, Any] = field(default_factory=dict)


def build_execution_intent_context(
    *,
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

    pretrade_intent = PretradeOrderIntent(
        symbol=symbol,
        side=side,
        qty=abs(delta_shares),
        notional=abs(delta_shares) * price,
        limit_price=price,
        bar_ts=bar_ts,
        client_order_id=client_order_id,
        last_price=price,
        mid=price,
        spread=(float(spread_bps) / 10_000.0) * float(price),
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

    order_annotations: dict[str, Any] = {}
    if submit_quote_source:
        order_annotations["price_source"] = submit_quote_source
    if policy_hash:
        order_annotations["policy_hash"] = policy_hash
    if decision_trace_id:
        order_annotations["decision_trace_id"] = decision_trace_id
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

    return ExecutionIntentContext(
        client_order_id=client_order_id,
        decision_trace_id=decision_trace_id,
        pretrade_intent=pretrade_intent,
        order_lineage_metadata=order_lineage_metadata,
        order_annotations=order_annotations,
    )
