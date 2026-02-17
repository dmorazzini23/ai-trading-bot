from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Protocol

try:
    import requests
except ImportError:  # pragma: no cover - requests is a runtime dependency
    requests = None  # type: ignore[assignment]


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


def _float_from(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_MANAGED_GET_ENV_READY = False
_MANAGED_GET_ENV: Any | None = None


def _managed_env(name: str, default: Any = None) -> Any:
    """Read env via config management with os.getenv fallback."""

    global _MANAGED_GET_ENV_READY, _MANAGED_GET_ENV
    if not _MANAGED_GET_ENV_READY:
        try:
            from ai_trading.config.management import get_env as _cfg_get_env
        except (ImportError, RuntimeError):
            _MANAGED_GET_ENV = None
        else:
            _MANAGED_GET_ENV = _cfg_get_env
        _MANAGED_GET_ENV_READY = True

    if _MANAGED_GET_ENV is not None:
        try:
            return _MANAGED_GET_ENV(name, default)
        except (TypeError, ValueError, RuntimeError):
            return os.getenv(name, default)
    return os.getenv(name, default)


@dataclass(slots=True)
class TradierBrokerAdapter:
    """Tradier HTTP adapter implementing the broker contract."""

    token: str
    account_id: str
    base_url: str = "https://api.tradier.com/v1"
    timeout_seconds: float = 10.0
    session: Any | None = None
    provider: str = "tradier"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        client = self.session
        if client is None:
            if requests is None:
                raise RuntimeError("requests package is required for Tradier broker adapter")
            client = requests
        request_fn = getattr(client, "request", None)
        if not callable(request_fn):
            raise RuntimeError("Tradier session/client does not expose request")
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        response = request_fn(
            method,
            url,
            headers=self._headers(),
            params=dict(params or {}),
            data=dict(data or {}),
            timeout=self.timeout_seconds,
        )
        raise_for_status = getattr(response, "raise_for_status", None)
        if callable(raise_for_status):
            raise_for_status()
        json_fn = getattr(response, "json", None)
        if not callable(json_fn):
            raise RuntimeError("Tradier response missing json parser")
        payload = json_fn()
        if not isinstance(payload, Mapping):
            raise RuntimeError("Tradier response payload must be a JSON object")
        return dict(payload)

    def get_account(self) -> dict[str, Any]:
        payload = self._request_json("GET", f"accounts/{self.account_id}/balances")
        balances = payload.get("balances")
        if isinstance(balances, Mapping):
            return dict(balances)
        return payload

    def list_orders(self, status: str = "open") -> list[dict[str, Any]]:
        payload = self._request_json(
            "GET",
            f"accounts/{self.account_id}/orders",
            params={"status": status},
        )
        orders_root = payload.get("orders")
        if not isinstance(orders_root, Mapping):
            return []
        order_payload = orders_root.get("order")
        if order_payload is None:
            return []
        if isinstance(order_payload, Mapping):
            return [dict(order_payload)]
        if isinstance(order_payload, list):
            normalized: list[dict[str, Any]] = []
            for item in order_payload:
                if isinstance(item, Mapping):
                    normalized.append(dict(item))
            return normalized
        return []

    def submit_order(self, order_data: Mapping[str, Any]) -> dict[str, Any]:
        symbol = str(order_data.get("symbol") or "").strip().upper()
        if not symbol:
            raise ValueError("Tradier order requires symbol")
        qty = order_data.get("quantity", order_data.get("qty"))
        if qty in (None, ""):
            raise ValueError("Tradier order requires quantity")
        side_raw = str(order_data.get("side") or "buy").strip().lower()
        side_map = {
            "buy": "buy",
            "sell": "sell",
            "short": "sell_short",
            "cover": "buy_to_cover",
        }
        side = side_map.get(side_raw, side_raw)
        order_type = str(order_data.get("type") or "market").strip().lower()
        tif_raw = str(order_data.get("time_in_force") or "day").strip().lower()
        duration = {"day": "day", "gtc": "gtc", "ioc": "day", "fok": "day"}.get(
            tif_raw,
            "day",
        )
        request_data: dict[str, Any] = {
            "class": "equity",
            "symbol": symbol,
            "side": side,
            "quantity": str(qty),
            "type": order_type,
            "duration": duration,
        }
        if order_type in {"limit", "stop_limit"}:
            limit_price = order_data.get("limit_price", order_data.get("price"))
            if limit_price in (None, ""):
                raise ValueError("Tradier limit orders require limit_price")
            request_data["price"] = str(limit_price)
        payload = self._request_json(
            "POST",
            f"accounts/{self.account_id}/orders",
            data=request_data,
        )
        order_payload = payload.get("order")
        order_id = None
        status = "submitted"
        if isinstance(order_payload, Mapping):
            order_id_val = order_payload.get("id")
            if order_id_val not in (None, ""):
                order_id = str(order_id_val)
            status_val = order_payload.get("status")
            if status_val not in (None, ""):
                status = str(status_val).strip().lower()
        if order_id is None:
            fallback_id = payload.get("id")
            if fallback_id not in (None, ""):
                order_id = str(fallback_id)
        if not order_id:
            raise RuntimeError("Tradier submit response missing order id")
        client_order_id = order_data.get("client_order_id", order_id)
        return {
            "id": order_id,
            "status": status,
            "symbol": symbol,
            "side": side_raw,
            "qty": qty,
            "client_order_id": client_order_id,
            "raw": payload,
        }


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
    if normalized in {"tradier", "tradier_api"}:
        if isinstance(client, TradierBrokerAdapter):
            return client
        token = str(_managed_env("TRADIER_ACCESS_TOKEN", "") or "").strip()
        account_id = str(_managed_env("TRADIER_ACCOUNT_ID", "") or "").strip()
        if not token or not account_id:
            return None
        base_url = str(_managed_env("TRADIER_BASE_URL", "https://api.tradier.com/v1") or "").strip()
        if not base_url:
            base_url = "https://api.tradier.com/v1"
        timeout_seconds = _float_from(
            _managed_env("TRADIER_TIMEOUT_SECONDS", 10.0),
            default=10.0,
        )
        session = client if callable(getattr(client, "request", None)) else None
        return TradierBrokerAdapter(
            token=token,
            account_id=account_id,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            session=session,
        )
    if client is None:
        return None
    return AlpacaBrokerAdapter(client=client)


__all__ = [
    "AlpacaBrokerAdapter",
    "BrokerAdapter",
    "PaperBrokerAdapter",
    "TradierBrokerAdapter",
    "build_broker_adapter",
]
