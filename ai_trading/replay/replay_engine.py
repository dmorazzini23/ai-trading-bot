"""Deterministic replay harness using live decision/OMS style pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Callable, Iterable

from ai_trading.analytics.tca import (
    ExecutionBenchmark,
    FillSummary,
    build_tca_record,
)
from ai_trading.logging import get_logger

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

    def _guard_no_real_broker_submit(self) -> None:
        if self._broker_submit is not None:
            raise RuntimeError("Replay mode must not execute real broker submits")

    def run(self, bars: Iterable[dict[str, Any]]) -> dict[str, Any]:
        self._guard_no_real_broker_submit()
        decision_records: list[dict[str, Any]] = []
        ledger_entries: list[dict[str, Any]] = []
        tca_records: list[dict[str, Any]] = []
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
        tca_records.append(
            build_tca_record(
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
            )
        )
        ledger_entries.append(
            {
                "client_order_id": client_order_id,
                "symbol": symbol,
                "side": side,
                "qty": 0.0,
                "price": price,
                "status": "canceled",
                "replay": True,
            }
        )

    def _append_fill(
        self,
        *,
        pending: dict[str, Any],
        fill_bar: dict[str, Any],
        ledger_entries: list[dict[str, Any]],
        tca_records: list[dict[str, Any]],
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
        fill_price = _deterministic_price(
            base_price=base_price,
            model=self.config.fill_model,
            slippage_bps=self.config.fill_slippage_bps,
            side=side,
        )
        fees = abs(qty) * fill_price * (self.config.fill_fee_bps / 10_000.0)
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
        )
        tca_records.append(tca)
        ledger_entries.append(
            {
                "client_order_id": client_order_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": fill_price,
                "replay": True,
            }
        )
