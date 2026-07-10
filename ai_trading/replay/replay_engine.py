"""Deterministic replay harness using live decision/OMS style pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Callable, Iterable, Mapping

from ai_trading.analytics.tca import (
    ExecutionBenchmark,
    FillSummary,
    build_tca_record,
)
from ai_trading.logging import get_logger
from ai_trading.replay.live_cost_alignment import resolve_live_cost_alignment

logger = get_logger(__name__)


@dataclass(slots=True)
class ReplayConfig:
    symbols: tuple[str, ...]
    start: datetime | None = None
    end: datetime | None = None
    timeframes: tuple[str, ...] = ("5Min",)
    rth_only: bool = True
    seed: int = 42
    speedup: int = 1
    simulate_fills: bool = True
    fill_model: str = "next_bar_mid"
    fill_slippage_bps: float = 5.0
    fill_fee_bps: float = 0.0
    enforce_oms_gates: bool = True
    live_cost_model: Mapping[str, Any] | None = None
    cost_alignment_now: datetime | None = None
    cost_max_age_seconds: float = 86_400.0
    cost_min_samples: int = 5
    cost_metric: str = "p90_total_cost_bps"


def _deterministic_price(
    *,
    base_price: float,
    model: str,
    slippage_bps: float,
    side: str,
) -> float:
    sign = 1.0 if side == "buy" else -1.0
    slippage = (float(slippage_bps) / 10_000.0) * float(base_price)
    if model == "close":
        return float(base_price)
    if model == "next_bar_vwap":
        return float(base_price + sign * slippage * 0.5)
    return float(base_price + sign * slippage)


def _uses_next_bar_fill(model: str) -> bool:
    return str(model or "").lower().startswith("next_bar")


def _slippage_multiplier(model: str) -> float:
    model_token = str(model or "").lower()
    if model_token == "close":
        return 0.0
    if model_token == "next_bar_vwap":
        return 0.5
    return 1.0


def _fill_base_price(fill_bar: dict[str, Any], *, model: str, fallback_price: float) -> float:
    model_token = str(model or "").lower()
    if model_token == "next_bar_vwap":
        raw = fill_bar.get("vwap", fill_bar.get("mid", fill_bar.get("close", fallback_price)))
    elif model_token == "next_bar_mid":
        raw = fill_bar.get("mid", fill_bar.get("close", fallback_price))
    else:
        raw = fill_bar.get("close", fallback_price)
    return float(raw or fallback_price)


def _parse_event_ts(raw: Any) -> datetime:
    text = str(raw or "")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = datetime.now(UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _lineage(decision: Mapping[str, Any]) -> dict[str, Any]:
    nested = next(
        (
            value
            for key in ("lineage", "model_lineage", "model")
            if isinstance((value := decision.get(key)), Mapping)
        ),
        {},
    )

    def _value(*keys: str) -> Any:
        for key in keys:
            value = decision.get(key)
            if value not in (None, ""):
                return value
            value = nested.get(key)
            if value not in (None, ""):
                return value
        return None

    return {
        "prediction_id": _value("prediction_id"),
        "decision_id": _value("decision_id", "intent_id"),
        "model_id": _value("model_id"),
        "model_version": _value("model_version", "version"),
        "model_artifact_hash": _value(
            "model_artifact_hash",
            "artifact_hash",
        ),
        "feature_version": _value("feature_version"),
        "required_bar_timeframe": _value(
            "required_bar_timeframe",
            "bar_timeframe",
            "timeframe",
        ),
    }


def _context_value(pending: Mapping[str, Any], *keys: str) -> str:
    for source_key in ("order", "decision", "decision_bar"):
        source = pending.get(source_key)
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            value = source.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def _decision_allows_order(decision: dict[str, Any]) -> bool:
    if decision.get("accepted") is False or decision.get("blocked") is True:
        return False
    if decision.get("rejected") is True:
        return False
    status = str(decision.get("status") or "").strip().lower()
    if status in {"blocked", "rejected", "denied"}:
        return False
    gates_raw = decision.get("gates")
    if gates_raw is None:
        gates_raw = decision.get("reason_codes", [])
    for gate in list(gates_raw or []):
        token = str(gate or "").strip().upper()
        if not token or token == "OK_TRADE" or token.endswith("_BYPASSED"):
            continue
        if any(fragment in token for fragment in ("BLOCK", "HALT", "REJECT", "DENY")):
            return False
    return True


class ReplayEngine:
    def __init__(
        self,
        config: ReplayConfig,
        *,
        pipeline: Callable[[dict[str, Any]], dict[str, Any]],
        broker_submit: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.pipeline = pipeline
        self._broker_submit = broker_submit
        self._cost_alignment_now = config.cost_alignment_now or datetime.now(UTC)

    def _guard_no_real_broker_submit(self) -> None:
        if self._broker_submit is not None:
            raise RuntimeError("Replay mode must not execute real broker submits")

    def run(self, bars: Iterable[dict[str, Any]]) -> dict[str, Any]:
        self._guard_no_real_broker_submit()
        decision_records: list[dict[str, Any]] = []
        ledger_entries: list[dict[str, Any]] = []
        tca_records: list[dict[str, Any]] = []
        cost_records: list[dict[str, Any]] = []
        pending_orders: list[dict[str, Any]] = []
        configured_symbols = set(self.config.symbols)

        for bar in bars:
            symbol = str(bar.get("symbol", "")).upper()
            if symbol and configured_symbols and symbol not in configured_symbols:
                continue

            if self.config.simulate_fills:
                still_pending: list[dict[str, Any]] = []
                for pending in pending_orders:
                    if str(pending["symbol"]) != symbol:
                        still_pending.append(pending)
                        continue
                    self._append_fill(
                        pending=pending,
                        fill_bar=dict(bar),
                        ledger_entries=ledger_entries,
                        tca_records=tca_records,
                        cost_records=cost_records,
                    )
                pending_orders = still_pending

            decision = dict(self.pipeline(dict(bar)))
            decision.setdefault("reason_codes", [])
            decision.setdefault("replay", True)
            decision_records.append(decision)

            order = decision.get("order")
            if not isinstance(order, dict):
                continue
            if self.config.enforce_oms_gates and not _decision_allows_order(decision):
                continue
            if not self.config.simulate_fills:
                continue
            side = str(order.get("side", "buy")).lower()
            qty = float(order.get("qty", 0.0) or 0.0)
            price = float(order.get("price", bar.get("close", 0.0)) or 0.0)
            client_order_id = str(order.get("client_order_id") or "")
            if not client_order_id:
                raw = f"{symbol}|{bar.get('ts')}|{side}|{qty}|{price}|{self.config.seed}"
                client_order_id = sha256(raw.encode("utf-8")).hexdigest()[:16]

            pending = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "order": order,
                "decision": decision,
                "decision_bar": dict(bar),
                "client_order_id": client_order_id,
            }
            if _uses_next_bar_fill(self.config.fill_model):
                pending_orders.append(pending)
                continue
            self._append_fill(
                pending=pending,
                fill_bar=dict(bar),
                ledger_entries=ledger_entries,
                tca_records=tca_records,
                cost_records=cost_records,
            )

        for pending in pending_orders:
            self._append_unfilled(
                pending=pending,
                ledger_entries=ledger_entries,
                tca_records=tca_records,
            )

        return {
            "decision_records": decision_records,
            "ledger_entries": ledger_entries,
            "tca_records": tca_records,
            "cost_diagnostics": {
                "orders": int(len(cost_records)),
                "source_counts": {
                    source: int(
                        sum(1 for item in cost_records if item.get("source") == source)
                    )
                    for source in ("fixed", "live", "fallback")
                },
                "alignment_counts": {
                    alignment: int(
                        sum(
                            1
                            for item in cost_records
                            if item.get("alignment") == alignment
                        )
                    )
                    for alignment in sorted(
                        {
                            str(item.get("alignment") or "unknown")
                            for item in cost_records
                        }
                    )
                },
                "max_resolved_cost_bps": (
                    max(float(item["resolved_cost_bps"]) for item in cost_records)
                    if cost_records
                    else 0.0
                ),
                "items": cost_records,
            },
        }

    def _append_unfilled(
        self,
        *,
        pending: dict[str, Any],
        ledger_entries: list[dict[str, Any]],
        tca_records: list[dict[str, Any]],
    ) -> None:
        symbol = str(pending["symbol"])
        side = str(pending["side"])
        price = float(pending["price"])
        order = pending["order"]
        decision = pending["decision"]
        decision_bar = pending["decision_bar"]
        client_order_id = str(pending["client_order_id"])
        event_ts = _parse_event_ts(decision_bar.get("ts"))
        benchmark = ExecutionBenchmark(
            arrival_price=price,
            mid_at_arrival=float(decision_bar.get("mid", price) or price),
            bid_at_arrival=decision_bar.get("bid"),
            ask_at_arrival=decision_bar.get("ask"),
            bar_close_price=decision_bar.get("close"),
            decision_ts=event_ts,
            submit_ts=event_ts,
            first_fill_ts=None,
        )
        fill = FillSummary(
            fill_vwap=price,
            total_qty=0.0,
            fees=0.0,
            status="canceled",
            partial_fill=False,
        )
        lineage = _lineage(decision)
        tca = build_tca_record(
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                benchmark=benchmark,
                fill=fill,
                sleeve=decision.get("sleeve"),
                regime_profile=decision.get("regime_profile"),
                provider="replay",
                order_type=str(order.get("order_type", "limit")),
                quote_proxy=True,
                generated_ts=event_ts,
                model_id=lineage["model_id"],
                model_version=lineage["model_version"],
        )
        tca.update(lineage)
        tca["replay_cost"] = {
            "source": "not_filled",
            "applied": False,
            "resolved_cost_bps": 0.0,
        }
        tca_records.append(tca)
        ledger_entries.append(
            {
                "client_order_id": client_order_id,
                "symbol": symbol,
                "side": side,
                "qty": 0.0,
                "price": price,
                "status": "canceled",
                "replay": True,
                **lineage,
            }
        )

    def _resolve_cost(self, pending: Mapping[str, Any]) -> dict[str, Any]:
        fallback = max(
            0.0,
            float(self.config.fill_slippage_bps) + float(self.config.fill_fee_bps),
        )
        if not isinstance(self.config.live_cost_model, Mapping):
            return {
                "source": "fixed",
                "alignment": "fixed",
                "resolved_cost_bps": float(fallback),
                "fallback_cost_bps": float(fallback),
                "observed_live_cost_bps": None,
                "sample_count": 0,
            }
        resolution = resolve_live_cost_alignment(
            self.config.live_cost_model,
            symbol=str(pending.get("symbol") or ""),
            side=_context_value(pending, "side", "order_side")
            or str(pending.get("side") or ""),
            session_bucket=_context_value(
                pending,
                "session_bucket",
                "session_regime",
                "session",
            ),
            order_type=_context_value(pending, "order_type", "type"),
            volatility_bucket=_context_value(
                pending,
                "volatility_bucket",
                "vol_bucket",
                "liquidity_bucket",
            ),
            fallback_cost_bps=fallback,
            now=self._cost_alignment_now,
            max_age_seconds=max(0.0, float(self.config.cost_max_age_seconds)),
            min_samples=max(1, int(self.config.cost_min_samples)),
            cost_metric=str(self.config.cost_metric or "p90_total_cost_bps"),
        )
        return {
            "source": str(resolution.get("source") or "fallback"),
            "alignment": str(resolution.get("alignment") or "unknown"),
            "resolved_cost_bps": max(
                fallback,
                float(resolution.get("resolved_cost_bps") or fallback),
            ),
            "fallback_cost_bps": float(fallback),
            "observed_live_cost_bps": resolution.get("observed_live_cost_bps"),
            "sample_count": int(resolution.get("sample_count") or 0),
        }

    def _append_fill(
        self,
        *,
        pending: dict[str, Any],
        fill_bar: dict[str, Any],
        ledger_entries: list[dict[str, Any]],
        tca_records: list[dict[str, Any]],
        cost_records: list[dict[str, Any]],
    ) -> None:
        symbol = str(pending["symbol"])
        side = str(pending["side"])
        qty = float(pending["qty"])
        price = float(pending["price"])
        order = pending["order"]
        decision = pending["decision"]
        decision_bar = pending["decision_bar"]
        client_order_id = str(pending["client_order_id"])
        base_price = _fill_base_price(
            fill_bar,
            model=self.config.fill_model,
            fallback_price=price,
        )
        replay_cost = self._resolve_cost(pending)
        resolved_cost_bps = float(replay_cost["resolved_cost_bps"])
        effective_fee_bps = float(self.config.fill_fee_bps)
        effective_slippage_bps = float(self.config.fill_slippage_bps)
        if isinstance(self.config.live_cost_model, Mapping):
            multiplier = _slippage_multiplier(self.config.fill_model)
            if multiplier <= 0.0:
                effective_fee_bps = resolved_cost_bps
                effective_slippage_bps = 0.0
            else:
                effective_fee_bps = min(
                    max(0.0, float(self.config.fill_fee_bps)),
                    resolved_cost_bps,
                )
                effective_slippage_bps = max(
                    0.0,
                    (resolved_cost_bps - effective_fee_bps) / multiplier,
                )
        fill_price = _deterministic_price(
            base_price=base_price,
            model=self.config.fill_model,
            slippage_bps=effective_slippage_bps,
            side=side,
        )
        fees = abs(qty) * fill_price * (effective_fee_bps / 10_000.0)
        event_ts = _parse_event_ts(decision_bar.get("ts"))
        fill_ts = _parse_event_ts(fill_bar.get("ts"))
        benchmark = ExecutionBenchmark(
            arrival_price=price,
            mid_at_arrival=float(decision_bar.get("mid", price) or price),
            bid_at_arrival=decision_bar.get("bid"),
            ask_at_arrival=decision_bar.get("ask"),
            bar_close_price=decision_bar.get("close"),
            decision_ts=event_ts,
            submit_ts=event_ts,
            first_fill_ts=fill_ts,
        )
        fill = FillSummary(
            fill_vwap=fill_price,
            total_qty=qty,
            fees=fees,
            status="filled",
            partial_fill=False,
        )
        lineage = _lineage(decision)
        tca = build_tca_record(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            benchmark=benchmark,
            fill=fill,
            sleeve=decision.get("sleeve"),
            regime_profile=decision.get("regime_profile"),
            provider="replay",
            order_type=str(order.get("order_type", "limit")),
            quote_proxy=True,
            generated_ts=fill_ts,
            model_id=lineage["model_id"],
            model_version=lineage["model_version"],
        )
        replay_cost.update(
            {
                "client_order_id": client_order_id,
                "symbol": symbol,
                "side": side,
                "effective_fee_bps": float(effective_fee_bps),
                "effective_slippage_bps": float(effective_slippage_bps),
            }
        )
        cost_records.append(dict(replay_cost))
        tca.update(lineage)
        tca["replay_cost"] = dict(replay_cost)
        tca_records.append(tca)
        ledger_entries.append(
            {
                "client_order_id": client_order_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": fill_price,
                "replay": True,
                "replay_cost": dict(replay_cost),
                **lineage,
            }
        )
