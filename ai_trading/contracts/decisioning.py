"""Canonical contracts for strategy, risk, and order decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

from .market import BrokerOrderSnapshot, ExecutionResult


DECISION_JOURNAL_SCHEMA_VERSION = "1.0.0"


def _normalize_side(side: Any) -> str:
    value = str(side or "").strip().lower()
    if value in {"buy", "long", "entry"}:
        return "buy"
    if value in {"sell", "short", "sell_short", "exit"}:
        return "sell"
    return "hold"


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _safe_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalize_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _normalize_reasons(raw: Any) -> list[str]:
    if isinstance(raw, str):
        items: Sequence[Any] = [raw]
    elif isinstance(raw, Sequence):
        items = raw
    else:
        items = []
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    side: str
    bar_ts: datetime | None
    strength: float
    confidence: float
    signal_id: str | None = None
    strategy_id: str | None = None
    timeframe: str | None = None
    signal_type: str | None = None
    price_target: float | None = None
    stop_loss: float | None = None
    expected_return: float | None = None
    risk_score: float | None = None
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "bar_ts": self.bar_ts.isoformat() if isinstance(self.bar_ts, datetime) else None,
            "strength": self.strength,
            "confidence": self.confidence,
            "signal_id": self.signal_id,
            "strategy_id": self.strategy_id,
            "timeframe": self.timeframe,
            "signal_type": self.signal_type,
            "price_target": self.price_target,
            "stop_loss": self.stop_loss,
            "expected_return": self.expected_return,
            "risk_score": self.risk_score,
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_strategy_signal(
        cls,
        signal: Any,
        *,
        bar_ts: datetime | None = None,
        reasons: Sequence[str] | None = None,
    ) -> Signal:
        metadata = getattr(signal, "metadata", None)
        return cls(
            symbol=str(getattr(signal, "symbol", "") or "").upper(),
            side=_normalize_side(getattr(signal, "side", None)),
            bar_ts=bar_ts or _normalize_timestamp(getattr(signal, "timestamp", None)),
            strength=float(getattr(signal, "strength", 0.0) or 0.0),
            confidence=float(getattr(signal, "confidence", 0.0) or 0.0),
            signal_id=_safe_text(getattr(signal, "id", None)),
            strategy_id=_safe_text(getattr(signal, "strategy_id", None)),
            timeframe=_safe_text(getattr(signal, "timeframe", None)),
            signal_type=_safe_text(getattr(signal, "signal_type", None)),
            price_target=_safe_float(getattr(signal, "price_target", None)),
            stop_loss=_safe_float(getattr(signal, "stop_loss", None)),
            expected_return=_safe_float(getattr(signal, "expected_return", None)),
            risk_score=_safe_float(getattr(signal, "risk_score", None)),
            reasons=_normalize_reasons(reasons),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )


@dataclass(frozen=True, slots=True)
class RiskDecision:
    symbol: str
    bar_ts: datetime | None
    accepted: bool
    gates: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    veto_gate: str | None = None
    expected_net_edge_bps: float | None = None
    liquidity_regime: str | None = None
    config_snapshot_hash: str | None = None
    policy_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bar_ts": self.bar_ts.isoformat() if isinstance(self.bar_ts, datetime) else None,
            "accepted": self.accepted,
            "gates": list(self.gates),
            "reasons": list(self.reasons),
            "veto_gate": self.veto_gate,
            "expected_net_edge_bps": self.expected_net_edge_bps,
            "liquidity_regime": self.liquidity_regime,
            "config_snapshot_hash": self.config_snapshot_hash,
            "policy_hash": self.policy_hash,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class OrderIntent:
    symbol: str
    side: str
    bar_ts: datetime | None
    qty: float | None = None
    notional: float | None = None
    limit_price: float | None = None
    client_order_id: str | None = None
    decision_trace_id: str | None = None
    strategy_id: str | None = None
    status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "bar_ts": self.bar_ts.isoformat() if isinstance(self.bar_ts, datetime) else None,
            "qty": self.qty,
            "notional": self.notional,
            "limit_price": self.limit_price,
            "client_order_id": self.client_order_id,
            "decision_trace_id": self.decision_trace_id,
            "strategy_id": self.strategy_id,
            "status": self.status,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_pretrade(cls, intent: Any) -> OrderIntent:
        return cls(
            symbol=str(getattr(intent, "symbol", "") or "").upper(),
            side=_normalize_side(getattr(intent, "side", None)),
            bar_ts=_normalize_timestamp(getattr(intent, "bar_ts", None)),
            qty=_safe_float(getattr(intent, "qty", None)),
            notional=_safe_float(getattr(intent, "notional", None)),
            limit_price=_safe_float(getattr(intent, "limit_price", None)),
            client_order_id=_safe_text(getattr(intent, "client_order_id", None)),
            metadata={
                "last_price": _safe_float(getattr(intent, "last_price", None)),
                "mid": _safe_float(getattr(intent, "mid", None)),
                "bid": _safe_float(getattr(intent, "bid", None)),
                "ask": _safe_float(getattr(intent, "ask", None)),
                "spread": _safe_float(getattr(intent, "spread", None)),
                "quote_age_ms": _safe_float(getattr(intent, "quote_age_ms", None)),
                "submit_quote_source": _safe_text(getattr(intent, "submit_quote_source", None)),
                "quote_quality_ok": getattr(intent, "quote_quality_ok", None),
                "opening_trade": getattr(intent, "opening_trade", None),
                "require_realtime_nbbo": getattr(intent, "require_realtime_nbbo", None),
                "kill_switch_active": getattr(intent, "kill_switch_active", None),
                "broker_ready": getattr(intent, "broker_ready", None),
            },
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        symbol: str,
        bar_ts: datetime | None,
    ) -> OrderIntent:
        metadata = dict(payload)
        return cls(
            symbol=symbol,
            side=_normalize_side(payload.get("side")),
            bar_ts=bar_ts or _normalize_timestamp(payload.get("bar_ts")),
            qty=_safe_float(payload.get("qty") or payload.get("shares")),
            notional=_safe_float(payload.get("notional") or payload.get("target_dollars")),
            limit_price=_safe_float(payload.get("limit_price") or payload.get("price")),
            client_order_id=_safe_text(payload.get("client_order_id") or payload.get("id")),
            decision_trace_id=_safe_text(payload.get("decision_trace_id")),
            strategy_id=_safe_text(payload.get("strategy_id")),
            status=_safe_text(payload.get("status")),
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class DecisionJournalEntry:
    event: str
    symbol: str
    bar_ts: datetime | None
    provider: str | None
    feed: str | None
    data_freshness_sec: float | None
    signal: Signal
    risk_decision: RiskDecision
    order_intent: OrderIntent | None = None
    target_delta_shares: float | None = None
    broker_result: ExecutionResult | None = None
    client_order_id: str | None = None
    reasons: list[str] = field(default_factory=list)
    decision_trace_id: str | None = None
    config_snapshot_hash: str | None = None
    policy_hash: str | None = None
    dataset_hash: str | None = None
    feature_version: str | None = None
    model_artifact_hash: str | None = None
    accepted: bool = False
    submitted: bool = False
    fills_count: int = 0
    schema_version: str = DECISION_JOURNAL_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event": self.event,
            "symbol": self.symbol,
            "bar_ts": self.bar_ts.isoformat() if isinstance(self.bar_ts, datetime) else None,
            "provider": self.provider,
            "feed": self.feed,
            "data_freshness_sec": self.data_freshness_sec,
            "client_order_id": self.client_order_id,
            "decision_trace_id": self.decision_trace_id,
            "accepted": self.accepted,
            "submitted": self.submitted,
            "fills_count": self.fills_count,
            "target_delta_shares": self.target_delta_shares,
            "reasons": list(self.reasons),
            "config_snapshot_hash": self.config_snapshot_hash,
            "policy_hash": self.policy_hash,
            "dataset_hash": self.dataset_hash,
            "feature_version": self.feature_version,
            "model_artifact_hash": self.model_artifact_hash,
            "signal": self.signal.to_dict(),
            "risk_decision": self.risk_decision.to_dict(),
            "order_intent": self.order_intent.to_dict() if self.order_intent is not None else None,
            "broker_result": (
                self.broker_result.to_dict() if self.broker_result is not None else None
            ),
            "metadata": dict(self.metadata),
        }


def _derive_signal_from_record(record: Any) -> Signal:
    explicit = getattr(record, "signal", None)
    if isinstance(explicit, Signal):
        return explicit
    symbol = str(getattr(record, "symbol", "") or "").upper()
    bar_ts = _normalize_timestamp(getattr(record, "bar_ts", None))
    sleeves = getattr(record, "sleeves", None)
    proposals = list(sleeves) if isinstance(sleeves, Sequence) else []
    primary = None
    if proposals:
        primary = max(
            proposals,
            key=lambda proposal: abs(float(getattr(proposal, "score", 0.0) or 0.0)),
        )
    net_target = getattr(record, "net_target", None)
    target_dollars = _safe_float(getattr(net_target, "target_dollars", None)) or 0.0
    side = "hold"
    if target_dollars > 0:
        side = "buy"
    elif target_dollars < 0:
        side = "sell"
    confidence = float(getattr(primary, "confidence", 0.0) or 0.0) if primary is not None else 0.0
    strength = abs(float(getattr(primary, "score", 0.0) or 0.0)) if primary is not None else 0.0
    signal_type = "netted_sleeve_signal" if primary is not None else "decision_record"
    strategy_id = _safe_text(getattr(primary, "sleeve", None)) if primary is not None else None
    reasons = _normalize_reasons(getattr(net_target, "reasons", None))
    metadata = {
        "inferred": True,
        "proposals": len(proposals),
        "target_dollars": target_dollars,
        "target_shares": _safe_float(getattr(net_target, "target_shares", None)),
    }
    return Signal(
        symbol=symbol,
        side=side,
        bar_ts=bar_ts,
        strength=strength,
        confidence=confidence,
        strategy_id=strategy_id,
        signal_type=signal_type,
        reasons=reasons,
        metadata=metadata,
    )


def _derive_risk_decision_from_record(record: Any) -> RiskDecision:
    explicit = getattr(record, "risk_decision", None)
    if isinstance(explicit, RiskDecision):
        return explicit
    symbol = str(getattr(record, "symbol", "") or "").upper()
    bar_ts = _normalize_timestamp(getattr(record, "bar_ts", None))
    gates = _normalize_reasons(getattr(record, "gates", None))
    accepted = "OK_TRADE" in gates
    veto_gate = next((gate for gate in gates if gate != "OK_TRADE"), None) if not accepted else None
    metrics = getattr(record, "metrics", None)
    metrics_map = dict(metrics) if isinstance(metrics, Mapping) else {}
    tca = getattr(record, "tca", None)
    tca_map = dict(tca) if isinstance(tca, Mapping) else {}
    config_snapshot = getattr(record, "config_snapshot", None)
    config_map = dict(config_snapshot) if isinstance(config_snapshot, Mapping) else {}
    reasons = [veto_gate] if veto_gate else []
    return RiskDecision(
        symbol=symbol,
        bar_ts=bar_ts,
        accepted=accepted,
        gates=gates,
        reasons=reasons,
        veto_gate=veto_gate,
        expected_net_edge_bps=(
            _safe_float(metrics_map.get("expected_net_edge_bps"))
            if _safe_float(metrics_map.get("expected_net_edge_bps")) is not None
            else _safe_float(tca_map.get("expected_net_edge_bps"))
        ),
        liquidity_regime=_safe_text(config_map.get("liquidity_regime")),
        config_snapshot_hash=_safe_text(config_map.get("config_snapshot_hash")),
        policy_hash=_safe_text(config_map.get("effective_policy_hash")),
        metadata={
            "metrics_keys": sorted(metrics_map.keys()),
        },
    )


def _derive_order_intent_from_record(record: Any) -> OrderIntent | None:
    explicit = getattr(record, "order_intent", None)
    if isinstance(explicit, OrderIntent):
        return explicit
    payload = getattr(record, "order", None)
    if not isinstance(payload, Mapping):
        return None
    symbol = str(getattr(record, "symbol", "") or "").upper()
    bar_ts = _normalize_timestamp(getattr(record, "bar_ts", None))
    return OrderIntent.from_mapping(payload, symbol=symbol, bar_ts=bar_ts)


def _derive_target_delta_shares(
    record: Any,
    order_intent: OrderIntent | None,
) -> float | None:
    if order_intent is not None and order_intent.qty is not None:
        qty = float(order_intent.qty)
        if order_intent.side == "sell":
            return -qty
        if order_intent.side == "buy":
            return qty
    payload = getattr(record, "order", None)
    if isinstance(payload, Mapping):
        qty = _safe_float(payload.get("qty") or payload.get("shares"))
        side = _normalize_side(payload.get("side"))
        if qty is not None:
            if side == "sell":
                return -float(qty)
            if side == "buy":
                return float(qty)
    return None


def _derive_provider_and_feed(record: Any) -> tuple[str | None, str | None]:
    metrics = getattr(record, "metrics", None)
    metrics_map = dict(metrics) if isinstance(metrics, Mapping) else {}
    tca = getattr(record, "tca", None)
    tca_map = dict(tca) if isinstance(tca, Mapping) else {}
    config_snapshot = getattr(record, "config_snapshot", None)
    config_map = dict(config_snapshot) if isinstance(config_snapshot, Mapping) else {}
    provider = (
        _safe_text(tca_map.get("provider"))
        or _safe_text(config_map.get("data_provider"))
        or _safe_text(config_map.get("primary_data_provider"))
    )
    nbbo_guard = metrics_map.get("nbbo_guard")
    nbbo_guard_map = dict(nbbo_guard) if isinstance(nbbo_guard, Mapping) else {}
    feed = (
        _safe_text(tca_map.get("quote_proxy_source"))
        or _safe_text(tca_map.get("feed"))
        or _safe_text(nbbo_guard_map.get("price_source"))
    )
    return provider, feed


def _derive_event(record: Any) -> str:
    metrics = getattr(record, "metrics", None)
    metrics_map = dict(metrics) if isinstance(metrics, Mapping) else {}
    config_snapshot = getattr(record, "config_snapshot", None)
    config_map = dict(config_snapshot) if isinstance(config_snapshot, Mapping) else {}
    event = _safe_text(metrics_map.get("event")) or _safe_text(config_map.get("event"))
    return event or "decision_record"


def _derive_data_freshness_sec(record: Any) -> float | None:
    metrics = getattr(record, "metrics", None)
    metrics_map = dict(metrics) if isinstance(metrics, Mapping) else {}
    config_snapshot = getattr(record, "config_snapshot", None)
    config_map = dict(config_snapshot) if isinstance(config_snapshot, Mapping) else {}
    for key in (
        "data_freshness_sec",
        "signal_age_sec",
        "quote_age_sec",
        "age_sec",
    ):
        parsed = _safe_float(metrics_map.get(key))
        if parsed is not None:
            return parsed
    for key in ("data_freshness_sec", "signal_age_sec", "quote_age_sec"):
        parsed = _safe_float(config_map.get(key))
        if parsed is not None:
            return parsed
    bar_ts = _normalize_timestamp(getattr(record, "bar_ts", None))
    if bar_ts is None:
        return None
    now_ts = datetime.now(UTC)
    return max((now_ts - bar_ts).total_seconds(), 0.0)


def _derive_journal_reasons(
    signal: Signal,
    risk_decision: RiskDecision,
    record: Any,
) -> list[str]:
    raw_reasons: list[str] = []
    raw_reasons.extend(_normalize_reasons(signal.reasons))
    raw_reasons.extend(_normalize_reasons(risk_decision.reasons))
    raw_reasons.extend(_normalize_reasons(getattr(record, "gates", None)))
    deduped: list[str] = []
    for reason in raw_reasons:
        normalized = str(reason or "").strip()
        if not normalized:
            continue
        if normalized in deduped:
            continue
        deduped.append(normalized)
    return deduped


def _derive_broker_result(
    record: Any,
    order_intent: OrderIntent | None,
    *,
    provider: str | None,
) -> ExecutionResult | None:
    payload = getattr(record, "order", None)
    payload_map = dict(payload) if isinstance(payload, Mapping) else {}
    tca = getattr(record, "tca", None)
    tca_map = dict(tca) if isinstance(tca, Mapping) else {}
    fills = getattr(record, "fills", None)
    fills_seq = list(fills) if isinstance(fills, Sequence) else []
    gates = _normalize_reasons(getattr(record, "gates", None))
    submitted = order_intent is not None
    accepted = "OK_TRADE" in gates
    if not submitted and not payload_map:
        return None
    broker_order = BrokerOrderSnapshot(
        client_order_id=(
            _safe_text(payload_map.get("client_order_id"))
            or (order_intent.client_order_id if order_intent is not None else None)
        ),
        broker_order_id=_safe_text(
            payload_map.get("broker_order_id") or payload_map.get("order_id")
        ),
        side=_safe_text(payload_map.get("side"))
        or (order_intent.side if order_intent is not None else None),
        qty=_safe_float(payload_map.get("qty") or payload_map.get("shares"))
        or (order_intent.qty if order_intent is not None else None),
        filled_qty=_safe_float(payload_map.get("filled_qty"))
        or _safe_float(tca_map.get("total_qty")),
        limit_price=_safe_float(payload_map.get("price") or payload_map.get("limit_price"))
        or (order_intent.limit_price if order_intent is not None else None),
        fill_price=_safe_float(tca_map.get("fill_price") or tca_map.get("fill_vwap")),
        status=_safe_text(payload_map.get("status")),
        venue=_safe_text(tca_map.get("venue")),
        ts=_normalize_timestamp(payload_map.get("ts") or payload_map.get("timestamp")),
        metadata={"raw_order": payload_map},
    )
    error_reason = None
    if not accepted:
        error_reason = next((gate for gate in gates if gate != "OK_TRADE"), None)
    return ExecutionResult(
        submitted=submitted,
        accepted=accepted,
        status=broker_order.status or ("accepted" if accepted else "rejected"),
        provider=provider,
        venue=broker_order.venue,
        broker_order=broker_order,
        fill_count=len(fills_seq),
        filled_qty=broker_order.filled_qty,
        realized_slippage_bps=_safe_float(
            tca_map.get("is_bps") or tca_map.get("spread_paid_bps")
        ),
        fees=_safe_float(tca_map.get("fees") or tca_map.get("fee")),
        error_reason=_safe_text(error_reason),
        metadata={
            "arrival_benchmark": _safe_text(tca_map.get("arrival_benchmark")),
            "quote_proxy_source": _safe_text(tca_map.get("quote_proxy_source")),
        },
    )


def build_decision_journal(record: Any) -> DecisionJournalEntry:
    signal = _derive_signal_from_record(record)
    risk_decision = _derive_risk_decision_from_record(record)
    order_intent = _derive_order_intent_from_record(record)
    provider, feed = _derive_provider_and_feed(record)
    target_delta_shares = _derive_target_delta_shares(record, order_intent)
    broker_result = _derive_broker_result(
        record,
        order_intent,
        provider=provider,
    )
    config_snapshot = getattr(record, "config_snapshot", None)
    config_map = dict(config_snapshot) if isinstance(config_snapshot, Mapping) else {}
    metrics = getattr(record, "metrics", None)
    metrics_map = dict(metrics) if isinstance(metrics, Mapping) else {}
    fills = getattr(record, "fills", None)
    fills_seq = list(fills) if isinstance(fills, Sequence) else []
    decision_trace_id = (
        getattr(record, "decision_trace_id", None)
        or (order_intent.decision_trace_id if order_intent is not None else None)
        or (order_intent.client_order_id if order_intent is not None else None)
    )
    metadata = {
        "legacy_schema_version": str(getattr(record, "schema_version", "") or ""),
        "has_order": order_intent is not None,
        "has_fills": bool(fills_seq),
    }
    market_bar = metrics_map.get("market_bar")
    if isinstance(market_bar, Mapping):
        metadata["market_bar"] = dict(market_bar)
    return DecisionJournalEntry(
        event=_derive_event(record),
        symbol=signal.symbol,
        bar_ts=signal.bar_ts or _normalize_timestamp(getattr(record, "bar_ts", None)),
        provider=provider,
        feed=feed,
        data_freshness_sec=_derive_data_freshness_sec(record),
        signal=signal,
        risk_decision=risk_decision,
        order_intent=order_intent,
        target_delta_shares=target_delta_shares,
        broker_result=broker_result,
        client_order_id=(
            order_intent.client_order_id
            if order_intent is not None
            else (
                broker_result.broker_order.client_order_id
                if broker_result is not None and broker_result.broker_order is not None
                else None
            )
        ),
        reasons=_derive_journal_reasons(signal, risk_decision, record),
        decision_trace_id=_safe_text(decision_trace_id),
        config_snapshot_hash=_safe_text(config_map.get("config_snapshot_hash")),
        policy_hash=_safe_text(config_map.get("effective_policy_hash")),
        dataset_hash=_safe_text(config_map.get("dataset_hash")),
        feature_version=_safe_text(config_map.get("feature_version")),
        model_artifact_hash=_safe_text(config_map.get("model_artifact_hash")),
        accepted=risk_decision.accepted,
        submitted=order_intent is not None,
        fills_count=len(fills_seq),
        metadata=metadata,
    )
