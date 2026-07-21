"""Helpers for canonical decision recording within the netting cycle."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping, Sequence

from ai_trading.core.netting import (
    DECISION_RECORD_SCHEMA_VERSION,
    DecisionRecord,
    NettedTarget,
    SleeveProposal,
    build_decision_record,
)
from ai_trading.core.evidence_lineage import (
    deterministic_opportunity_correlation_id,
    normalize_evidence_timestamp,
    normalize_opportunity_side,
)
from ai_trading.logging import get_logger


logger = get_logger(__name__)


_MODEL_LINEAGE_KEYS: tuple[str, ...] = (
    "model_id",
    "model_version",
    "dataset_hash",
    "feature_version",
    "model_artifact_hash",
)

_DECISION_METADATA_FIELDS: tuple[str, ...] = (
    "quote_age_ms",
    "spread_bps",
    "order_type",
    "session_regime",
    "market_regime",
    "execution_profile",
)


def _safe_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        return None
    return numeric


def _primary_sleeve_debug(sleeves: Sequence[SleeveProposal]) -> Mapping[str, Any]:
    if not sleeves:
        return {}
    primary = max(
        sleeves,
        key=lambda sleeve: (
            abs(float(sleeve.target_dollars)),
            abs(float(sleeve.score)),
            str(sleeve.sleeve),
        ),
    )
    debug = getattr(primary, "debug", None)
    return debug if isinstance(debug, Mapping) else {}


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = _safe_text(value)
        if text:
            return text
    return None


def _derive_ml_model_lineage(
    sleeves: Sequence[SleeveProposal],
) -> dict[str, str]:
    """Return lineage declared by ML-influenced sleeve proposals only."""

    lineage: dict[str, str] = {}
    for sleeve in sleeves:
        debug = getattr(sleeve, "debug", None)
        if not isinstance(debug, Mapping) or debug.get("ml_influenced") is not True:
            continue
        proposal_lineage = debug.get("model_lineage")
        if not isinstance(proposal_lineage, Mapping):
            continue
        for key in _MODEL_LINEAGE_KEYS:
            if key in lineage:
                continue
            value = str(proposal_lineage.get(key) or "").strip()
            if value:
                lineage[key] = value
    return lineage


@dataclass(slots=True)
class DecisionRecorder:
    runtime: Any
    path: str | None
    write_impl: Callable[[Any, str | None], None]
    dedupe_gate_root_causes: Callable[[Sequence[str]], list[str]]
    session_bucket_from_ts: Callable[[datetime], str]
    safe_float: Callable[[Any], float | None]
    quote_snapshot_func: Callable[[str], Mapping[str, Any]] | None = None
    decision_gate_counts: Counter[str] = field(default_factory=Counter)
    decision_records_total: int = 0
    decision_observations: list[dict[str, Any]] = field(default_factory=list)

    def build_record(
        self,
        *,
        symbol: str,
        bar_ts: datetime,
        net_target: NettedTarget,
        gates: Sequence[str],
        sleeves: Sequence[SleeveProposal] | None = None,
        order: Mapping[str, Any] | None = None,
        fills: Sequence[Mapping[str, Any]] | None = None,
        metrics: Mapping[str, Any] | None = None,
        config_snapshot: Mapping[str, Any] | None = None,
        tca: Mapping[str, Any] | None = None,
        decision_trace_id: str | None = None,
        correlation_id: str | None = None,
        order_intent: Any | None = None,
        schema_version: str = DECISION_RECORD_SCHEMA_VERSION,
    ) -> DecisionRecord:
        resolved_sleeves = (
            list(sleeves) if sleeves is not None else list(net_target.proposals)
        )
        metrics_payload = dict(metrics) if isinstance(metrics, Mapping) else {}
        model_lineage = _derive_ml_model_lineage(resolved_sleeves)
        for key, value in model_lineage.items():
            if not str(metrics_payload.get(key) or "").strip():
                metrics_payload[key] = value
        order_payload = dict(order) if isinstance(order, Mapping) else {}
        config_payload = (
            dict(config_snapshot) if isinstance(config_snapshot, Mapping) else {}
        )
        quote_snapshot: dict[str, Any] = {}
        quote_snapshot_error: str | None = None
        if self.quote_snapshot_func is not None and symbol.strip().upper() != "ALL":
            try:
                observed_quote = self.quote_snapshot_func(symbol)
                if isinstance(observed_quote, Mapping):
                    quote_snapshot = dict(observed_quote)
                else:
                    quote_snapshot_error = "quote_snapshot_invalid"
            except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
                quote_snapshot_error = "quote_snapshot_error"
                logger.warning(
                    "DECISION_METADATA_QUOTE_SNAPSHOT_FAILED",
                    extra={
                        "symbol": str(symbol).strip().upper(),
                        "error": str(exc),
                    },
                )
        side = order_payload.get("side") or getattr(order_intent, "side", None)
        if normalize_opportunity_side(side) == "hold":
            if float(net_target.target_shares) > 0.0:
                side = "buy"
            elif float(net_target.target_shares) < 0.0:
                side = "sell"
            elif resolved_sleeves:
                primary = max(
                    resolved_sleeves,
                    key=lambda sleeve: (
                        abs(float(sleeve.target_dollars)),
                        abs(float(sleeve.score)),
                        str(sleeve.sleeve),
                    ),
                )
                side = (
                    "buy"
                    if float(primary.score) > 0.0
                    else "sell"
                    if float(primary.score) < 0.0
                    else "hold"
                )
        source_timestamp = (
            normalize_evidence_timestamp(metrics_payload.get("source_timestamp"))
            or normalize_evidence_timestamp(metrics_payload.get("source_ts"))
            or bar_ts
        )
        resolved_correlation_id = str(correlation_id or "").strip() or (
            deterministic_opportunity_correlation_id(
                symbol=symbol,
                source_timestamp=source_timestamp,
                side=side,
                strategy_id=(
                    metrics_payload.get("strategy_id")
                    or metrics_payload.get("strategy")
                ),
                sleeves=(sleeve.sleeve for sleeve in resolved_sleeves),
                opportunity_key=metrics_payload.get("opportunity_key"),
            )
        )
        metrics_payload.setdefault("correlation_id", resolved_correlation_id)
        metrics_payload.setdefault("decision_ts", bar_ts.isoformat())
        metrics_payload.setdefault("source_timestamp", source_timestamp.isoformat())
        metrics_payload.setdefault(
            "opportunity_eligible",
            normalize_opportunity_side(side) in {"buy", "sell"},
        )
        metrics_payload.setdefault("evidence_type", "decision_opportunity")
        quote_age_ms = _finite_float(quote_snapshot.get("quote_age_ms"))
        if quote_age_ms is None:
            quote_age_sec = _finite_float(quote_snapshot.get("age_sec"))
            if quote_age_sec is not None:
                quote_age_ms = max(0.0, quote_age_sec * 1000.0)
        if quote_age_ms is not None:
            metrics_payload.setdefault("quote_age_ms", max(0.0, quote_age_ms))
        quote_timestamp = _first_text(
            metrics_payload.get("quote_timestamp"),
            metrics_payload.get("quote_ts"),
            quote_snapshot.get("quote_timestamp"),
            quote_snapshot.get("updated"),
        )
        if quote_timestamp:
            metrics_payload.setdefault("quote_timestamp", quote_timestamp)
        quote_status = _first_text(
            metrics_payload.get("quote_status"),
            quote_snapshot.get("status"),
        )
        if quote_status:
            metrics_payload.setdefault("quote_status", quote_status.lower())
        quote_source = _first_text(
            metrics_payload.get("quote_source"),
            quote_snapshot.get("source"),
        )
        if quote_source:
            metrics_payload.setdefault("quote_source", quote_source)
        bid = _finite_float(quote_snapshot.get("bid"))
        ask = _finite_float(quote_snapshot.get("ask"))
        if bid is not None and ask is not None and bid > 0.0 and ask >= bid:
            midpoint = (bid + ask) / 2.0
            if midpoint > 0.0:
                metrics_payload.setdefault(
                    "spread_bps",
                    ((ask - bid) / midpoint) * 10_000.0,
                )
                metrics_payload.setdefault("reference_price", midpoint)
        if "reference_price" not in metrics_payload:
            last_price = _finite_float(quote_snapshot.get("last_price"))
            if last_price is not None and last_price > 0.0:
                metrics_payload["reference_price"] = last_price

        order_intent_metadata = getattr(order_intent, "metadata", None)
        intent_metadata = (
            dict(order_intent_metadata)
            if isinstance(order_intent_metadata, Mapping)
            else {}
        )
        order_type = _first_text(
            metrics_payload.get("order_type"),
            order_payload.get("order_type"),
            order_payload.get("type"),
            intent_metadata.get("order_type"),
            getattr(order_intent, "order_type", None),
        )
        if order_type is None:
            intent_limit_price = _finite_float(
                getattr(order_intent, "limit_price", None)
            )
            if intent_limit_price is not None and intent_limit_price > 0.0:
                order_type = "limit"
        if order_type:
            metrics_payload.setdefault("order_type", order_type.lower())
        elif not order_payload and order_intent is None:
            metrics_payload.setdefault("order_type", "not_submitted")

        metrics_payload.setdefault(
            "session_regime",
            self.session_bucket_from_ts(bar_ts),
        )
        primary_debug = _primary_sleeve_debug(resolved_sleeves)
        market_regime = _first_text(
            metrics_payload.get("market_regime"),
            primary_debug.get("ml_serving_regime"),
            primary_debug.get("market_regime"),
            primary_debug.get("regime"),
            quote_snapshot.get("market_regime"),
            config_payload.get("market_regime"),
            config_payload.get("current_regime"),
        )
        if market_regime:
            normalized_market_regime = market_regime.lower()
            metrics_payload.setdefault("market_regime", normalized_market_regime)
            if "volatile" in normalized_market_regime:
                metrics_payload.setdefault("volatility_regime", normalized_market_regime)
            if normalized_market_regime in {
                "uptrend",
                "downtrend",
                "trending",
                "trend",
            }:
                metrics_payload.setdefault("trend_regime", normalized_market_regime)
        regime_profile = _first_text(
            metrics_payload.get("regime_profile"),
            config_payload.get("regime_signal_profile"),
            config_payload.get("regime_profile"),
        )
        if regime_profile:
            metrics_payload.setdefault("regime_profile", regime_profile.lower())
        liquidity_regime = _first_text(
            metrics_payload.get("liquidity_regime"),
            config_payload.get("liquidity_regime"),
        )
        if liquidity_regime:
            metrics_payload.setdefault("liquidity_regime", liquidity_regime.lower())
        execution_profile = _first_text(
            metrics_payload.get("execution_profile"),
            quote_snapshot.get("execution_profile"),
            intent_metadata.get("execution_profile"),
            order_payload.get("execution_profile"),
            config_payload.get("execution_profile"),
        )
        if execution_profile:
            metrics_payload.setdefault("execution_profile", execution_profile.lower())

        missing_reasons: dict[str, str] = {}
        for field_name in _DECISION_METADATA_FIELDS:
            if metrics_payload.get(field_name) not in (None, ""):
                continue
            if field_name in {"quote_age_ms", "spread_bps"}:
                missing_reasons[field_name] = (
                    quote_snapshot_error
                    or _first_text(quote_snapshot.get("reason"))
                    or "quote_snapshot_unavailable"
                )
            elif field_name == "order_type":
                missing_reasons[field_name] = "submitted_order_type_unavailable"
            elif field_name == "market_regime":
                missing_reasons[field_name] = "market_regime_unavailable"
            else:
                missing_reasons[field_name] = f"{field_name}_unavailable"
        if quote_timestamp is None:
            missing_reasons["quote_timestamp"] = (
                quote_snapshot_error
                or _first_text(quote_snapshot.get("reason"))
                or "quote_timestamp_unavailable"
            )
        metrics_payload["metadata_quality_status"] = (
            "complete" if not missing_reasons else "partial"
        )
        metrics_payload["metadata_missing_reasons"] = missing_reasons
        return build_decision_record(
            symbol=symbol,
            bar_ts=bar_ts,
            net_target=net_target,
            sleeves=resolved_sleeves,
            gates=list(gates),
            order=order_payload or None,
            fills=[dict(fill) for fill in fills] if fills is not None else None,
            metrics=(metrics_payload if metrics_payload else None),
            config_snapshot=config_payload or None,
            tca=dict(tca) if isinstance(tca, Mapping) else None,
            decision_trace_id=decision_trace_id,
            correlation_id=resolved_correlation_id,
            order_intent=order_intent,
            schema_version=schema_version,
        )

    def write(self, record: DecisionRecord) -> None:
        self.decision_records_total += 1
        gates_raw = getattr(record, "gates", None)
        gates: list[str] = []
        if isinstance(gates_raw, Sequence):
            for gate in gates_raw:
                gate_text = str(gate or "").strip()
                if gate_text:
                    gates.append(gate_text)
        gates = self.dedupe_gate_root_causes(gates)
        try:
            setattr(record, "gates", list(gates))
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            pass
        for gate_text in gates:
            self.decision_gate_counts[gate_text] += 1
        symbol_value = str(getattr(record, "symbol", "") or "").strip().upper() or "UNKNOWN"
        config_snapshot_raw = getattr(record, "config_snapshot", None)
        config_snapshot = (
            dict(config_snapshot_raw)
            if isinstance(config_snapshot_raw, Mapping)
            else {}
        )
        regime_value = (
            str(config_snapshot.get("liquidity_regime", "") or "").strip().upper()
            or "UNKNOWN"
        )
        metrics_raw = getattr(record, "metrics", None)
        metrics = dict(metrics_raw) if isinstance(metrics_raw, Mapping) else {}
        tca_raw = getattr(record, "tca", None)
        tca = dict(tca_raw) if isinstance(tca_raw, Mapping) else {}
        expected_net_edge_bps = self.safe_float(metrics.get("expected_net_edge_bps"))
        if expected_net_edge_bps is None:
            expected_net_edge_bps = self.safe_float(tca.get("expected_net_edge_bps"))
        if expected_net_edge_bps is None:
            candidate_rank_raw = getattr(
                self.runtime,
                "execution_candidate_rank_expected_edge_bps",
                {},
            )
            candidate_rank = (
                dict(candidate_rank_raw)
                if isinstance(candidate_rank_raw, Mapping)
                else {}
            )
            expected_net_edge_bps = self.safe_float(candidate_rank.get(symbol_value))
        if expected_net_edge_bps is None:
            expected_net_edge_bps = 0.0
        try:
            realized_is_bps = float(tca.get("is_bps", 0.0) or 0.0)
        except (TypeError, ValueError):
            realized_is_bps = 0.0
        accepted = "OK_TRADE" in gates
        realized_net_edge_bps = self.safe_float(tca.get("realized_net_edge_bps"))
        if realized_net_edge_bps is None and accepted:
            realized_net_edge_bps = float(expected_net_edge_bps - abs(realized_is_bps))
        edge_proxy_bps = (
            float(realized_net_edge_bps)
            if accepted and realized_net_edge_bps is not None
            else float(expected_net_edge_bps)
        )
        bar_ts_value = getattr(record, "bar_ts", None)
        session_bucket = (
            self.session_bucket_from_ts(bar_ts_value)
            if isinstance(bar_ts_value, datetime)
            else "offhours"
        )
        sleeves_raw = getattr(record, "sleeves", None)
        sleeves: list[str] = []
        if isinstance(sleeves_raw, Sequence):
            for sleeve in sleeves_raw:
                sleeve_name = str(getattr(sleeve, "sleeve", "") or "").strip().lower()
                if not sleeve_name:
                    sleeve_name = (
                        str(getattr(sleeve, "name", "") or "").strip().lower()
                    )
                if sleeve_name:
                    sleeves.append(sleeve_name)
        self.decision_observations.append(
            {
                "symbol": symbol_value,
                "gates": list(gates),
                "sleeves": list(sorted(set(sleeves))),
                "accepted": bool(accepted),
                "regime": regime_value,
                "session_bucket": session_bucket,
                "expected_net_edge_bps": float(expected_net_edge_bps),
                "realized_is_bps": float(realized_is_bps),
                "realized_net_edge_bps": (
                    float(realized_net_edge_bps)
                    if realized_net_edge_bps is not None
                    else None
                ),
                "edge_proxy_bps": float(edge_proxy_bps),
                "correlation_id": str(getattr(record, "correlation_id", "") or "")
                or None,
            }
        )
        self.write_impl(record, self.path)

    def record(self, **kwargs: Any) -> DecisionRecord:
        record = self.build_record(**kwargs)
        self.write(record)
        return record

    def record_global_block(
        self,
        *,
        bar_ts: datetime,
        gates: Sequence[str],
        config_snapshot: Mapping[str, Any] | None = None,
        decision_trace_id: str | None = None,
        symbol: str = "ALL",
    ) -> DecisionRecord:
        return self.record(
            symbol=symbol,
            bar_ts=bar_ts,
            sleeves=[],
            net_target=NettedTarget(
                symbol=symbol,
                bar_ts=bar_ts,
                target_dollars=0.0,
                target_shares=0.0,
            ),
            gates=list(gates),
            config_snapshot=dict(config_snapshot) if isinstance(config_snapshot, Mapping) else {},
            decision_trace_id=decision_trace_id,
            metrics={"opportunity_eligible": False},
        )


__all__ = ["DecisionRecorder"]
