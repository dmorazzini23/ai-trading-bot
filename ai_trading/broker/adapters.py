from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

OrderClass: Any
OrderSide: Any
TimeInForce: Any
QueryOrderStatus: Any
GetOrdersRequest: Any
LimitOrderRequest: Any
MarketOrderRequest: Any
StopLimitOrderRequest: Any
StopLossRequest: Any
StopOrderRequest: Any
TakeProfitRequest: Any
TrailingStopOrderRequest: Any

try:
    from alpaca.trading.enums import (
        OrderClass as _OrderClass,
        OrderSide as _OrderSide,
        QueryOrderStatus as _QueryOrderStatus,
        TimeInForce as _TimeInForce,
    )
    from alpaca.trading.requests import (
        GetOrdersRequest as _GetOrdersRequest,
        LimitOrderRequest as _LimitOrderRequest,
        MarketOrderRequest as _MarketOrderRequest,
        StopLimitOrderRequest as _StopLimitOrderRequest,
        StopLossRequest as _StopLossRequest,
        StopOrderRequest as _StopOrderRequest,
        TakeProfitRequest as _TakeProfitRequest,
        TrailingStopOrderRequest as _TrailingStopOrderRequest,
    )
except (ImportError, ModuleNotFoundError, RuntimeError, OSError):  # pragma: no cover - import guard
    OrderClass = None
    OrderSide = None
    QueryOrderStatus = None
    TimeInForce = None
    GetOrdersRequest = None
    LimitOrderRequest = None
    MarketOrderRequest = None
    StopLimitOrderRequest = None
    StopLossRequest = None
    StopOrderRequest = None
    TakeProfitRequest = None
    TrailingStopOrderRequest = None
else:
    OrderClass = _OrderClass
    OrderSide = _OrderSide
    QueryOrderStatus = _QueryOrderStatus
    TimeInForce = _TimeInForce
    GetOrdersRequest = _GetOrdersRequest
    LimitOrderRequest = _LimitOrderRequest
    MarketOrderRequest = _MarketOrderRequest
    StopLimitOrderRequest = _StopLimitOrderRequest
    StopLossRequest = _StopLossRequest
    StopOrderRequest = _StopOrderRequest
    TakeProfitRequest = _TakeProfitRequest
    TrailingStopOrderRequest = _TrailingStopOrderRequest

try:
    import requests  # type: ignore[import-untyped]
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


@dataclass
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
        if callable(lister):
            orders = lister(status=status)
        else:
            getter = getattr(self.client, "get_orders", None)
            if not callable(getter):
                return []
            if GetOrdersRequest is None or QueryOrderStatus is None:
                raise RuntimeError("alpaca-py order query models are unavailable")
            status_text = str(status or "open").strip().lower()
            status_value = getattr(QueryOrderStatus, status_text.upper(), status_text)
            orders = getter(filter=GetOrdersRequest(status=status_value))
        if orders is None:
            return []
        return list(orders)

    def submit_order(self, order_data: Mapping[str, Any]) -> Any:
        submit = getattr(self.client, "submit_order", None)
        if not callable(submit):
            raise RuntimeError("Broker client does not expose submit_order")
        request = _build_alpaca_order_request(order_data)
        return submit(order_data=request)


def _alpaca_enum_value(enum_cls: Any, value: Any, *, default: Any) -> Any:
    token = str(value or "").strip()
    if not token:
        return default
    name = token.replace("-", "_").replace(" ", "_").upper()
    member = getattr(enum_cls, name, None)
    if member is not None:
        return member
    try:
        return enum_cls(token.lower())
    except (TypeError, ValueError):
        return default


def _alpaca_bracket_leg(value: Any, *, scalar_field: str, request_cls: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return request_cls(**dict(value)) if request_cls is not None else dict(value)
    return request_cls(**{scalar_field: value}) if request_cls is not None else {scalar_field: value}


def _build_alpaca_order_request(order_data: Mapping[str, Any]) -> Any:
    if (
        OrderClass is None
        or OrderSide is None
        or TimeInForce is None
        or MarketOrderRequest is None
        or LimitOrderRequest is None
    ):
        raise RuntimeError("alpaca-py request models are unavailable")

    order_type = str(order_data.get("type") or order_data.get("order_type") or "market").strip().lower()
    symbol = str(order_data.get("symbol") or "").strip().upper()
    if not symbol:
        raise ValueError("Alpaca order requires symbol")
    side_raw = _normalize_broker_side(order_data.get("side"))
    side = OrderSide.BUY if side_raw in {"buy", "buy_to_cover"} else OrderSide.SELL
    tif = _alpaca_enum_value(
        TimeInForce,
        order_data.get("time_in_force") or "day",
        default=TimeInForce.DAY,
    )
    request_kwargs: dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "time_in_force": tif,
    }
    qty = order_data.get("qty", order_data.get("quantity"))
    notional = order_data.get("notional")
    if qty not in (None, ""):
        request_kwargs["qty"] = qty
    elif notional not in (None, ""):
        request_kwargs["notional"] = notional
    else:
        raise ValueError("Alpaca order requires qty, quantity, or notional")
    client_order_id = order_data.get("client_order_id")
    if client_order_id not in (None, ""):
        request_kwargs["client_order_id"] = str(client_order_id)
    if order_data.get("extended_hours") is not None:
        request_kwargs["extended_hours"] = bool(order_data.get("extended_hours"))
    order_class = order_data.get("order_class")
    if order_class not in (None, ""):
        request_kwargs["order_class"] = _alpaca_enum_value(
            OrderClass,
            order_class,
            default=order_class,
        )
    take_profit = _alpaca_bracket_leg(
        order_data.get("take_profit"),
        scalar_field="limit_price",
        request_cls=TakeProfitRequest,
    )
    if take_profit is not None:
        request_kwargs["take_profit"] = take_profit
    stop_loss = _alpaca_bracket_leg(
        order_data.get("stop_loss"),
        scalar_field="stop_price",
        request_cls=StopLossRequest,
    )
    if stop_loss is not None:
        request_kwargs["stop_loss"] = stop_loss

    if order_type == "market":
        return MarketOrderRequest(**request_kwargs)
    if order_type == "limit":
        limit_price = order_data.get("limit_price", order_data.get("price"))
        if limit_price in (None, ""):
            raise ValueError("Alpaca limit orders require limit_price")
        return LimitOrderRequest(limit_price=limit_price, **request_kwargs)
    if order_type == "stop":
        if StopOrderRequest is None:
            raise RuntimeError("Alpaca StopOrderRequest model is unavailable")
        stop_price = order_data.get("stop_price")
        if stop_price in (None, ""):
            raise ValueError("Alpaca stop orders require stop_price")
        return StopOrderRequest(stop_price=stop_price, **request_kwargs)
    if order_type == "stop_limit":
        if StopLimitOrderRequest is None:
            raise RuntimeError("Alpaca StopLimitOrderRequest model is unavailable")
        stop_price = order_data.get("stop_price")
        limit_price = order_data.get("limit_price", order_data.get("price"))
        if stop_price in (None, "") or limit_price in (None, ""):
            raise ValueError("Alpaca stop_limit orders require stop_price and limit_price")
        return StopLimitOrderRequest(
            stop_price=stop_price,
            limit_price=limit_price,
            **request_kwargs,
        )
    if order_type == "trailing_stop":
        if TrailingStopOrderRequest is None:
            raise RuntimeError("Alpaca TrailingStopOrderRequest model is unavailable")
        trail_price = order_data.get("trail_price")
        trail_percent = order_data.get("trail_percent")
        if trail_price not in (None, ""):
            return TrailingStopOrderRequest(trail_price=trail_price, **request_kwargs)
        if trail_percent not in (None, ""):
            return TrailingStopOrderRequest(trail_percent=trail_percent, **request_kwargs)
        raise ValueError("Alpaca trailing_stop orders require trail_price or trail_percent")
    raise ValueError(f"Unsupported Alpaca order type: {order_type}")


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


def _normalize_broker_side(side: Any) -> str:
    raw = str(side or "").strip().lower()
    if raw in {"short", "sell_short", "sellshort", "sell-short", "sell short"}:
        return "sell_short"
    if raw in {"cover", "buy_to_cover", "buy-to-cover", "buy to cover"}:
        return "buy_to_cover"
    if raw in {"buy", "sell"}:
        return raw
    raise ValueError(f"Unsupported broker order side: {side!r}")


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _normalize_broker_status(status: Any, *, default: str = "accepted") -> str:
    raw = str(status or "").strip().lower()
    if "." in raw:
        raw = raw.split(".")[-1]
    normalized = raw.replace("-", "_").replace(" ", "_")
    aliases = {
        "cancelled": "canceled",
        "complete": "filled",
        "completed": "filled",
        "error": "rejected",
    }
    return aliases.get(normalized, normalized) or default


def _normalize_adapter_order_payload(
    payload: Mapping[str, Any],
    *,
    requested_qty: Any = None,
    client_order_id: Any = None,
) -> dict[str, Any]:
    normalized = dict(payload)
    order_id = _first_present(normalized, "id", "order_id")
    if order_id not in (None, ""):
        normalized["id"] = str(order_id)
    symbol = _first_present(normalized, "symbol")
    if symbol not in (None, ""):
        normalized["symbol"] = str(symbol).strip().upper()
    side = _first_present(normalized, "side")
    if side not in (None, ""):
        normalized["side"] = _normalize_broker_side(side)
    status = _first_present(normalized, "status", "order_status")
    normalized["status"] = _normalize_broker_status(status)

    qty = _first_present(normalized, "qty", "quantity", "requested_quantity")
    if qty in (None, ""):
        qty = requested_qty
    if qty not in (None, ""):
        normalized["qty"] = qty
        normalized["quantity"] = qty

    filled_qty = _first_present(
        normalized,
        "filled_qty",
        "filled_quantity",
        "exec_quantity",
    )
    if filled_qty in (None, ""):
        filled_qty = "0"
    normalized["filled_qty"] = filled_qty
    normalized["filled_quantity"] = filled_qty

    filled_avg_price = _first_present(
        normalized,
        "filled_avg_price",
        "avg_fill_price",
        "average_fill_price",
    )
    if filled_avg_price not in (None, ""):
        normalized["filled_avg_price"] = filled_avg_price

    resolved_client_order_id = (
        _first_present(normalized, "client_order_id", "tag")
        if client_order_id in (None, "")
        else client_order_id
    )
    if resolved_client_order_id not in (None, ""):
        normalized["client_order_id"] = resolved_client_order_id
        normalized.setdefault("tag", resolved_client_order_id)
    return normalized


_MANAGED_GET_ENV_READY = False
_MANAGED_GET_ENV: Any | None = None


def _managed_env(name: str, default: Any = None) -> Any:
    """Read env via config management."""

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
            return default
    return default


@dataclass
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
            return [_normalize_adapter_order_payload(order_payload)]
        if isinstance(order_payload, list):
            normalized: list[dict[str, Any]] = []
            for item in order_payload:
                if isinstance(item, Mapping):
                    normalized.append(_normalize_adapter_order_payload(item))
            return normalized
        return []

    def submit_order(self, order_data: Mapping[str, Any]) -> dict[str, Any]:
        symbol = str(order_data.get("symbol") or "").strip().upper()
        if not symbol:
            raise ValueError("Tradier order requires symbol")
        qty = order_data.get("quantity", order_data.get("qty"))
        if qty in (None, ""):
            raise ValueError("Tradier order requires quantity")
        side_raw = _normalize_broker_side(order_data.get("side"))
        side_map = {
            "buy": "buy",
            "sell": "sell",
            "sell_short": "sell_short",
            "buy_to_cover": "buy_to_cover",
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
        client_order_id = order_data.get("client_order_id")
        if client_order_id not in (None, ""):
            request_data["tag"] = str(client_order_id)
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
                status = _normalize_broker_status(status_val, default="submitted")
        if order_id is None:
            fallback_id = payload.get("id")
            if fallback_id not in (None, ""):
                order_id = str(fallback_id)
        if not order_id:
            raise RuntimeError("Tradier submit response missing order id")
        response_payload: dict[str, Any] = {
            "id": order_id,
            "status": status,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "quantity": qty,
            "filled_qty": "0",
            "filled_quantity": "0",
            "filled_avg_price": None,
            "client_order_id": client_order_id,
            "raw": payload,
        }
        if isinstance(order_payload, Mapping):
            response_payload.update(
                _normalize_adapter_order_payload(
                    order_payload,
                    requested_qty=qty,
                    client_order_id=client_order_id,
                )
            )
            response_payload["raw"] = payload
        if response_payload.get("client_order_id") in (None, ""):
            response_payload["client_order_id"] = client_order_id or order_id
        return response_payload


@dataclass
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
        side = _normalize_broker_side(order_data.get("side"))
        order_id = f"paper-{int(datetime.now(UTC).timestamp() * 1000)}"
        payload: dict[str, Any] = {
            "id": order_id,
            "status": "accepted",
            "symbol": order_data.get("symbol"),
            "side": side,
            "qty": qty,
            "quantity": qty,
            "filled_qty": "0",
            "filled_quantity": "0",
            "filled_avg_price": None,
            "limit_price": order_data.get("limit_price"),
            "client_order_id": order_data.get("client_order_id", order_id),
            "type": order_data.get("type", order_data.get("order_type", "market")),
            "time_in_force": order_data.get("time_in_force", "day"),
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
