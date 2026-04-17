"""Shared OMS lifecycle driver for simulated/backtest execution paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ai_trading.logging import get_logger
from ai_trading.oms.lifecycle import (
    normalize_terminal_status,
    resolve_terminal_intent_status,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class SimulatedIntentRef:
    """Reference to a simulated intent persisted through IntentStore."""

    intent_id: str
    idempotency_key: str
    broker_order_id: str


class SimulatedLifecycleDriver:
    """Drive simulated order lifecycle through the shared durable IntentStore path."""

    def __init__(
        self,
        *,
        enabled: bool,
        source: str,
        database_url: str | None = None,
        intent_store_path: str | None = None,
    ) -> None:
        self._enabled = bool(enabled)
        self._source = str(source or "").strip() or "simulation"
        self._database_url = (
            str(database_url).strip() if database_url not in (None, "") else None
        )
        self._intent_store_path = (
            str(intent_store_path).strip()
            if intent_store_path not in (None, "")
            else None
        )
        self._store: Any | None = None
        self._store_init_failed = False

    def _resolve_store(self) -> Any | None:
        if not self._enabled:
            return None
        if self._store_init_failed:
            return None
        if self._store is not None:
            return self._store
        try:
            from ai_trading.oms.intent_store import IntentStore

            self._store = IntentStore(
                path=self._intent_store_path,
                url=self._database_url,
                event_dual_write_enabled=self._enabled,
            )
        except Exception as exc:
            self._store_init_failed = True
            logger.warning(
                "SIM_OMS_INTENT_STORE_INIT_FAILED",
                extra={"source": self._source, "error": str(exc)},
            )
            return None
        return self._store

    @staticmethod
    def _ts_text(value: Any) -> str:
        iso = getattr(value, "isoformat", None)
        if callable(iso):
            try:
                return str(iso())
            except Exception:
                return str(value)
        return str(value)

    def open_submitted_intent(
        self,
        *,
        intent_id: str,
        idempotency_key: str,
        symbol: str,
        side: str,
        quantity: float,
        decision_ts: Any,
        broker_order_id: str | None = None,
        strategy_id: str | None = None,
        expected_edge_bps: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SimulatedIntentRef | None:
        """Persist a simulated intent through create -> claim -> submit ack lifecycle."""

        store = self._resolve_store()
        if store is None:
            return None
        intent_token = str(intent_id or "").strip()
        key_token = str(idempotency_key or "").strip()
        if not intent_token or not key_token:
            return None
        metadata_payload = dict(metadata or {})
        metadata_payload.setdefault("simulation", True)
        metadata_payload.setdefault("simulation_source", self._source)
        metadata_payload.setdefault("lifecycle_driver", "intent_store")

        try:
            record, _created = store.create_intent(
                intent_id=intent_token,
                idempotency_key=key_token,
                symbol=str(symbol or "").upper(),
                side=str(side or "").lower(),
                quantity=float(quantity),
                decision_ts=self._ts_text(decision_ts),
                strategy_id=(str(strategy_id) if strategy_id not in (None, "") else None),
                expected_edge_bps=(
                    float(expected_edge_bps)
                    if expected_edge_bps is not None
                    else None
                ),
                metadata=metadata_payload,
            )
            # Immediate simulated submit path mirrors live state transitions.
            store.claim_for_submit(record.intent_id, stale_after_seconds=1)
            resolved_broker_order_id = str(
                broker_order_id or record.broker_order_id or record.intent_id
            )
            if not str(record.broker_order_id or "").strip():
                store.mark_submitted(record.intent_id, resolved_broker_order_id)
        except Exception:
            logger.warning(
                "SIM_OMS_INTENT_OPEN_FAILED",
                extra={
                    "source": self._source,
                    "intent_id": intent_token,
                    "symbol": str(symbol or "").upper(),
                },
                exc_info=True,
            )
            return None
        return SimulatedIntentRef(
            intent_id=str(record.intent_id),
            idempotency_key=key_token,
            broker_order_id=resolved_broker_order_id,
        )

    def record_fill_and_close_intent(
        self,
        *,
        intent_id: str,
        fill_qty: float,
        fill_price: float | None,
        fee: float = 0.0,
        liquidity_flag: str | None = None,
        fill_ts: Any | None = None,
        terminal_status: str | None = None,
        last_error: str | None = None,
    ) -> bool:
        """Persist simulated fill and terminal transition through shared lifecycle path."""

        store = self._resolve_store()
        if store is None:
            return False
        resolved_intent_id = str(intent_id or "").strip()
        if not resolved_intent_id:
            return False
        try:
            qty = max(0.0, float(fill_qty or 0.0))
        except (TypeError, ValueError):
            qty = 0.0
        parsed_price: float | None
        try:
            parsed_price = (
                float(fill_price) if fill_price is not None else None
            )
        except (TypeError, ValueError):
            parsed_price = None
        try:
            parsed_fee = float(fee or 0.0)
        except (TypeError, ValueError):
            parsed_fee = 0.0

        resolved_terminal = str(terminal_status or "").strip().upper()
        if not resolved_terminal:
            resolved_terminal = "FILLED" if qty > 0.0 else "FAILED"
        mapped_terminal = resolve_terminal_intent_status(
            status=resolved_terminal,
            status_is_terminal=True,
        )
        normalized_terminal = (
            str(mapped_terminal)
            if mapped_terminal
            else normalize_terminal_status(resolved_terminal)
        )

        try:
            if qty > 0.0:
                store.record_fill(
                    resolved_intent_id,
                    fill_qty=qty,
                    fill_price=parsed_price,
                    fee=parsed_fee,
                    liquidity_flag=(
                        str(liquidity_flag)
                        if liquidity_flag not in (None, "")
                        else None
                    ),
                    fill_ts=(
                        self._ts_text(fill_ts)
                        if fill_ts not in (None, "")
                        else None
                    ),
                )
            store.close_intent(
                resolved_intent_id,
                final_status=normalized_terminal,
                last_error=(str(last_error) if last_error not in (None, "") else None),
            )
        except Exception:
            logger.warning(
                "SIM_OMS_INTENT_CLOSE_FAILED",
                extra={
                    "source": self._source,
                    "intent_id": resolved_intent_id,
                    "final_status": normalized_terminal,
                },
                exc_info=True,
            )
            return False
        return True

    def close(self) -> None:
        """Dispose underlying store resources when initialized."""

        store = self._store
        self._store = None
        if store is None:
            return
        try:
            store.close()
        except Exception:
            logger.debug("SIM_OMS_INTENT_STORE_CLOSE_FAILED", exc_info=True)


__all__ = ["SimulatedIntentRef", "SimulatedLifecycleDriver"]
