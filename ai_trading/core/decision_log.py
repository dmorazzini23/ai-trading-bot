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


@dataclass(slots=True)
class DecisionRecorder:
    runtime: Any
    path: str | None
    write_impl: Callable[[Any, str | None], None]
    dedupe_gate_root_causes: Callable[[Sequence[str]], list[str]]
    session_bucket_from_ts: Callable[[datetime], str]
    safe_float: Callable[[Any], float | None]
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
        order_intent: Any | None = None,
        schema_version: str = DECISION_RECORD_SCHEMA_VERSION,
    ) -> DecisionRecord:
        return build_decision_record(
            symbol=symbol,
            bar_ts=bar_ts,
            net_target=net_target,
            sleeves=list(sleeves) if sleeves is not None else list(net_target.proposals),
            gates=list(gates),
            order=dict(order) if isinstance(order, Mapping) else None,
            fills=[dict(fill) for fill in fills] if fills is not None else None,
            metrics=dict(metrics) if isinstance(metrics, Mapping) else None,
            config_snapshot=(
                dict(config_snapshot) if isinstance(config_snapshot, Mapping) else None
            ),
            tca=dict(tca) if isinstance(tca, Mapping) else None,
            decision_trace_id=decision_trace_id,
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
        )


__all__ = ["DecisionRecorder"]
