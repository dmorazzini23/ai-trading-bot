from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Protocol


class BrokerAdapter(Protocol):
    """Minimal adapter contract for broker capacity and order primitives."""

    provider: str

    def get_account(self) -> Any | None:
        """Return broker account payload."""

    def list_orders(self, status: str = "open") -> list[Any]:
        """Return open orders (or empty list when unavailable)."""

    def submit_order(self, order_data: Mapping[str, Any]) -> Any:
        """Submit order payload and return broker response."""


@dataclass(slots=True)
class AlpacaBrokerAdapter:
    """Thin adapter over an Alpaca-style client object."""

    client: Any
    provider: str = "alpaca"

    def get_account(self) -> Any | None:
        getter = getattr(self.client, "get_account", None)
        if not callable(getter):
            return None
        return getter()

    def list_orders(self, status: str = "open") -> list[Any]:
        lister = getattr(self.client, "list_orders", None)
        if not callable(lister):
            return []
        orders = lister(status=status)
        if orders is None:
            return []
        return list(orders)

    def submit_order(self, order_data: Mapping[str, Any]) -> Any:
        submit = getattr(self.client, "submit_order", None)
        if not callable(submit):
            raise RuntimeError("Broker client does not expose submit_order")
        return submit(dict(order_data))


def _decimal_from(value: Any, *, default: Decimal) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return default


@dataclass(slots=True)
class PaperBrokerAdapter:
    """In-memory paper broker adapter used as a non-Alpaca fallback."""

    buying_power: Decimal = Decimal("100000")
    maintenance_margin: Decimal = Decimal("0")
    provider: str = "paper"
    _orders: list[dict[str, Any]] = field(default_factory=list)

    def get_account(self) -> dict[str, str]:
        return {
            "buying_power": str(self.buying_power),
            "maintenance_margin": str(self.maintenance_margin),
            "non_marginable_buying_power": str(self.buying_power),
        }

    def list_orders(self, status: str = "open") -> list[dict[str, Any]]:
        if status != "open":
            return []
        return list(self._orders)

    def submit_order(self, order_data: Mapping[str, Any]) -> dict[str, Any]:
        qty = order_data.get("quantity", order_data.get("qty"))
        order_id = f"paper-{int(datetime.now(UTC).timestamp() * 1000)}"
        payload: dict[str, Any] = {
            "id": order_id,
            "status": "accepted",
            "symbol": order_data.get("symbol"),
            "side": order_data.get("side"),
            "qty": qty,
            "limit_price": order_data.get("limit_price"),
            "client_order_id": order_data.get("client_order_id", order_id),
        }
        self._orders.append(payload)
        return payload


def build_broker_adapter(
    *,
    provider: str | None,
    client: Any | None,
    paper_buying_power: Any | None = None,
) -> BrokerAdapter | None:
    """Return broker adapter for provider; defaults to Alpaca wrapper."""

    normalized = str(provider or "alpaca").strip().lower()
    if normalized in {"paper", "paper_sim", "sim"}:
        buying_power = _decimal_from(
            paper_buying_power if paper_buying_power is not None else "100000",
            default=Decimal("100000"),
        )
        return PaperBrokerAdapter(buying_power=buying_power)
    if client is None:
        return None
    return AlpacaBrokerAdapter(client=client)


__all__ = [
    "AlpacaBrokerAdapter",
    "BrokerAdapter",
    "PaperBrokerAdapter",
    "build_broker_adapter",
]
