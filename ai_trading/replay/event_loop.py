"""Event-driven replay loop with async simulated broker events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Iterable, Mapping

from ai_trading.config.management import get_env
from ai_trading.execution.simulated_broker import SimulatedBroker


def _to_utc(raw: Any) -> datetime:
    text = str(raw)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class ReplayInvariantViolation:
    code: str
    message: str
    ts: str


class ReplayEventLoop:
    """Replay bars while processing asynchronous order events between bars."""

    def __init__(
        self,
        *,
        strategy: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
        broker: SimulatedBroker | None = None,
        seed: int | None = None,
        max_symbol_notional: float | None = None,
        max_gross_notional: float | None = None,
    ) -> None:
        replay_seed = int(seed if seed is not None else get_env("REPLAY_SEED", "42", cast=int))
        self.strategy = strategy
        self.broker = broker or SimulatedBroker(seed=replay_seed)
        self.seed = replay_seed
        self.max_symbol_notional = float(
            max_symbol_notional
            if max_symbol_notional is not None
            else get_env("GLOBAL_MAX_SYMBOL_DOLLARS", "25000", cast=float)
        )
        self.max_gross_notional = float(
            max_gross_notional
            if max_gross_notional is not None
            else get_env("GLOBAL_MAX_GROSS_DOLLARS", "150000", cast=float)
        )
        self._seen_intent_keys: set[str] = set()
        self._positions: dict[str, float] = {}

    def run(self, bars: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Execute replay loop and enforce parity invariants."""

        ordered = sorted((dict(bar) for bar in bars), key=lambda row: _to_utc(row.get("ts")))
        intents: list[dict[str, Any]] = []
        orders: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        violations: list[ReplayInvariantViolation] = []

        for bar in ordered:
            ts = _to_utc(bar.get("ts"))
            symbol = str(bar.get("symbol", "")).upper()
            close = float(bar.get("close", 0.0) or 0.0)
            if close <= 0:
                continue

            fill_events = self.broker.process_until(
                now=ts,
                market_price_by_symbol={symbol: close},
            )
            for event in fill_events:
                events.append(event)
                if event.get("event_type") != "fill":
                    continue
                fill_symbol = str(event.get("symbol", "")).upper()
                fill_qty = float(event.get("fill_qty", 0.0) or 0.0)
                side = str(event.get("side", "buy")).lower()
                signed_qty = fill_qty if side == "buy" else -fill_qty
                self._positions[fill_symbol] = self._positions.get(fill_symbol, 0.0) + signed_qty
                if abs(self._positions[fill_symbol]) < 1e-9:
                    self._positions.pop(fill_symbol, None)

            proposal = self.strategy(bar)
            if proposal is None:
                continue

            side = str(proposal.get("side", "buy")).lower()
            qty = float(proposal.get("qty", 0.0) or 0.0)
            if qty <= 0:
                continue
            intent_key = str(
                proposal.get(
                    "intent_key",
                    f"{symbol}|{side}|{round(qty, 6)}|{ts.isoformat()}",
                )
            )
            if intent_key in self._seen_intent_keys:
                violations.append(
                    ReplayInvariantViolation(
                        code="duplicate_intent",
                        message=f"Duplicate intent key: {intent_key}",
                        ts=ts.isoformat(),
                    )
                )
                continue
            self._seen_intent_keys.add(intent_key)

            if not self._passes_notional_limits(symbol=symbol, close=close, side=side, qty=qty):
                violations.append(
                    ReplayInvariantViolation(
                        code="position_cap_exceeded",
                        message=f"Position/gross cap exceeded for {symbol}",
                        ts=ts.isoformat(),
                    )
                )
                continue

            order_payload = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "type": str(proposal.get("type", proposal.get("order_type", "limit"))),
                "price": proposal.get("price", close),
                "limit_price": proposal.get("limit_price", proposal.get("price", close)),
                "client_order_id": str(proposal.get("client_order_id", intent_key)),
            }
            intents.append(
                {
                    "ts": ts.isoformat(),
                    "symbol": symbol,
                    "intent_key": intent_key,
                    "side": side,
                    "qty": qty,
                }
            )
            order = self.broker.submit_order(order_payload, timestamp=ts)
            orders.append(order)

        if ordered:
            last_ts = _to_utc(ordered[-1].get("ts"))
            trailing_events = self.broker.process_until(
                now=last_ts + timedelta(minutes=5),
                market_price_by_symbol={},
            )
            for event in trailing_events:
                events.append(event)
                if event.get("event_type") != "fill":
                    continue
                fill_symbol = str(event.get("symbol", "")).upper()
                fill_qty = float(event.get("fill_qty", 0.0) or 0.0)
                side = str(event.get("side", "buy")).lower()
                signed_qty = fill_qty if side == "buy" else -fill_qty
                self._positions[fill_symbol] = self._positions.get(fill_symbol, 0.0) + signed_qty
                if abs(self._positions[fill_symbol]) < 1e-9:
                    self._positions.pop(fill_symbol, None)

        return {
            "seed": self.seed,
            "intents": intents,
            "orders": orders,
            "events": events,
            "violations": [
                {
                    "code": violation.code,
                    "message": violation.message,
                    "ts": violation.ts,
                }
                for violation in violations
            ],
            "positions": dict(self._positions),
        }

    def _passes_notional_limits(self, *, symbol: str, close: float, side: str, qty: float) -> bool:
        signed_qty = qty if side == "buy" else -qty
        projected_symbol_qty = self._positions.get(symbol, 0.0) + signed_qty
        projected_symbol_notional = abs(projected_symbol_qty * close)
        if projected_symbol_notional > self.max_symbol_notional:
            return False

        gross = 0.0
        for sym, position_qty in self._positions.items():
            px = close if sym == symbol else close
            gross += abs(position_qty * px)
        gross += abs(signed_qty * close)
        return gross <= self.max_gross_notional
