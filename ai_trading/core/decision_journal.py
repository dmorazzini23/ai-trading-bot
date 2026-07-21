"""Canonical decision-journal helpers for non-netting trade execution paths."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Mapping, Sequence

from ai_trading.contracts import Bar, OrderIntent, RiskDecision, Signal
from ai_trading.core.evidence_lineage import (
    deterministic_opportunity_correlation_id,
    normalize_evidence_timestamp,
)
from ai_trading.core.netting import NettedTarget, build_decision_record


def _normalize_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalize_side(side: Any) -> str:
    value = str(side or "").strip().lower()
    if value in {"buy", "long"}:
        return "buy"
    if value in {"sell", "short", "sell_short"}:
        return "sell"
    return "hold"


def _normalize_reasons(raw: Sequence[Any] | None) -> list[str]:
    if raw is None:
        return []
    out: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


@dataclass(slots=True)
class DecisionJournalRecorder:
    path: str | None
    write_impl: Callable[[Any, str | None], Any]

    def record(
        self,
        *,
        symbol: str,
        bar_ts: datetime | None,
        market_bar: Bar | None = None,
        signal_side: str,
        final_score: float,
        confidence: float,
        strategy_id: str | None,
        accepted: bool,
        gates: Sequence[str] | None = None,
        reasons: Sequence[str] | None = None,
        target_delta_shares: float | None = None,
        submitted: bool = False,
        client_order_id: str | None = None,
        broker_order_id: str | None = None,
        broker_status: str | None = None,
        provider: str | None = None,
        feed: str | None = None,
        venue: str | None = None,
        limit_price: float | None = None,
        reference_price: float | None = None,
        filled_qty: float | None = None,
        fill_price: float | None = None,
        realized_slippage_bps: float | None = None,
        fees: float | None = None,
        event: str = "trade_decision",
        data_freshness_sec: float | None = None,
        correlation_id: str | None = None,
        decision_ts: datetime | None = None,
        source_timestamp: datetime | None = None,
        quote_timestamp: datetime | None = None,
        quote_age_ms: float | None = None,
        spread_bps: float | None = None,
        order_type: str | None = None,
        session: str | None = None,
        market_regime: str | None = None,
        volatility_regime: str | None = None,
        trend_regime: str | None = None,
        execution_profile: str | None = None,
        opportunity_eligible: bool | None = None,
        config_snapshot: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        normalized_symbol = str(symbol or "").strip().upper() or "UNKNOWN"
        normalized_bar_ts = (
            _normalize_timestamp(bar_ts)
            or (market_bar.ts if isinstance(market_bar, Bar) else None)
            or datetime.now(UTC)
        )
        normalized_source_ts = (
            normalize_evidence_timestamp(source_timestamp)
            or (market_bar.ts if isinstance(market_bar, Bar) else None)
            or normalized_bar_ts
        )
        normalized_decision_ts = (
            normalize_evidence_timestamp(decision_ts) or normalized_bar_ts
        )
        normalized_quote_ts = normalize_evidence_timestamp(quote_timestamp)
        normalized_side = _normalize_side(signal_side)
        resolved_correlation_id = str(correlation_id or "").strip() or (
            deterministic_opportunity_correlation_id(
                symbol=normalized_symbol,
                source_timestamp=normalized_source_ts,
                side=normalized_side,
                strategy_id=strategy_id,
            )
        )
        resolved_opportunity_eligible = (
            bool(opportunity_eligible)
            if opportunity_eligible is not None
            else normalized_side in {"buy", "sell"}
        )
        gates_list = _normalize_reasons(gates)
        reasons_list = _normalize_reasons(reasons)
        if accepted and "OK_TRADE" not in gates_list:
            gates_list = ["OK_TRADE", *gates_list]
        if not accepted and not gates_list:
            gates_list = ["DECISION_BLOCKED"]
        provider_text = _safe_text(provider) or (
            market_bar.provider if isinstance(market_bar, Bar) else None
        )
        feed_text = _safe_text(feed) or (
            market_bar.feed if isinstance(market_bar, Bar) else None
        )
        signal = Signal(
            symbol=normalized_symbol,
            side=normalized_side,
            bar_ts=normalized_bar_ts,
            strength=float(abs(final_score)),
            confidence=float(confidence),
            strategy_id=_safe_text(strategy_id),
            signal_type="trade_signal",
            reasons=list(reasons_list),
            metadata={"runtime": "non_netting"},
        )
        risk_decision = RiskDecision(
            symbol=normalized_symbol,
            bar_ts=normalized_bar_ts,
            accepted=bool(accepted),
            gates=list(gates_list),
            reasons=list(reasons_list),
            veto_gate=(next((gate for gate in gates_list if gate != "OK_TRADE"), None)),
            metadata={"runtime": "non_netting"},
        )

        price_value = _safe_float(reference_price)
        if price_value is None and isinstance(market_bar, Bar):
            price_value = market_bar.close
        delta_shares = _safe_float(target_delta_shares) or 0.0
        target_dollars = (
            float(delta_shares * price_value)
            if price_value is not None
            else 0.0
        )
        if delta_shares == 0.0 and price_value is not None and limit_price is not None:
            target_dollars = 0.0

        intent: OrderIntent | None = None
        if (
            submitted
            or client_order_id is not None
            or limit_price is not None
            or delta_shares != 0.0
        ):
            order_side = signal.side
            if delta_shares < 0:
                order_side = "sell"
            elif delta_shares > 0:
                order_side = "buy"
            intent = OrderIntent(
                symbol=normalized_symbol,
                side=order_side,
                bar_ts=normalized_bar_ts,
                qty=abs(delta_shares) if delta_shares != 0.0 else None,
                notional=abs(target_dollars) if target_dollars != 0.0 else None,
                limit_price=_safe_float(limit_price) or price_value,
                client_order_id=_safe_text(client_order_id),
                correlation_id=resolved_correlation_id,
                status=_safe_text(broker_status),
                strategy_id=_safe_text(strategy_id),
                metadata={"runtime": "non_netting"},
            )

        order_payload: dict[str, Any] | None = None
        if intent is not None:
            order_payload = {
                "side": intent.side,
                "qty": intent.qty,
                "price": intent.limit_price,
                "client_order_id": intent.client_order_id,
                "correlation_id": resolved_correlation_id,
                "broker_order_id": _safe_text(broker_order_id),
                "status": _safe_text(broker_status),
            }

        tca_payload: dict[str, Any] | None = None
        if submitted or provider_text or feed_text or venue:
            tca_payload = {
                "provider": provider_text,
                "feed": feed_text,
                "quote_proxy_source": feed_text,
                "venue": _safe_text(venue),
                "fill_price": _safe_float(fill_price),
                "total_qty": _safe_float(filled_qty),
                "is_bps": _safe_float(realized_slippage_bps),
                "fees": _safe_float(fees),
            }

        snapshot = dict(config_snapshot) if isinstance(config_snapshot, Mapping) else {}
        if provider_text and "data_provider" not in snapshot:
            snapshot["data_provider"] = provider_text
        if provider_text and "primary_data_provider" not in snapshot:
            snapshot["primary_data_provider"] = provider_text

        metrics = dict(metadata) if isinstance(metadata, Mapping) else {}
        metrics["event"] = str(event or "trade_decision")
        metrics["correlation_id"] = resolved_correlation_id
        metrics["decision_ts"] = normalized_decision_ts.isoformat()
        metrics["source_timestamp"] = normalized_source_ts.isoformat()
        metrics["opportunity_eligible"] = resolved_opportunity_eligible
        metrics.setdefault("evidence_type", "decision_opportunity")
        if normalized_quote_ts is not None:
            metrics["quote_timestamp"] = normalized_quote_ts.isoformat()
        for key, value in (
            ("quote_age_ms", _safe_float(quote_age_ms)),
            ("spread_bps", _safe_float(spread_bps)),
            ("order_type", _safe_text(order_type)),
            ("session_regime", _safe_text(session)),
            ("market_regime", _safe_text(market_regime)),
            ("volatility_regime", _safe_text(volatility_regime)),
            ("trend_regime", _safe_text(trend_regime)),
            ("execution_profile", _safe_text(execution_profile)),
        ):
            if value is not None:
                metrics[key] = value
        if data_freshness_sec is not None:
            metrics["data_freshness_sec"] = float(data_freshness_sec)
        if reference_price is not None:
            metrics["reference_price"] = float(reference_price)
        elif price_value is not None:
            metrics["reference_price"] = float(price_value)
        if isinstance(market_bar, Bar):
            metrics.setdefault("market_bar", market_bar.to_dict())

        record = build_decision_record(
            symbol=normalized_symbol,
            bar_ts=normalized_bar_ts,
            net_target=NettedTarget(
                symbol=normalized_symbol,
                bar_ts=normalized_bar_ts,
                target_dollars=float(target_dollars),
                target_shares=float(delta_shares),
                reasons=list(reasons_list),
            ),
            gates=list(gates_list),
            order=order_payload,
            metrics=metrics,
            config_snapshot=snapshot,
            tca=tca_payload,
            signal=signal,
            risk_decision=risk_decision,
            order_intent=intent,
            correlation_id=resolved_correlation_id,
        )
        self.write_impl(record, self.path)
        return record


__all__ = ["DecisionJournalRecorder"]
