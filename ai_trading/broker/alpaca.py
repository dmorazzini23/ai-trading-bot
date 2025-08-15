from __future__ import annotations

from typing import Any, Iterable, Optional

try:  # AI-AGENT-REF: safe optional import for tests
    from alpaca.trading.client import TradingClient as _RealTradingClient  # type: ignore
    ALPACA_SDK_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _RealTradingClient = None
    ALPACA_SDK_AVAILABLE = False


class MockTradingClient:
    def __init__(self, *a, **k):
        pass

    def submit_order(self, *a, **k):
        return {"status": "mock"}


TradingClient = _RealTradingClient or MockTradingClient


class MockOrderSide:
    BUY = "buy"
    SELL = "sell"


class MockTimeInForce:
    DAY = "day"


class MockOrderStatus:
    NEW = "new"


class MockQueryOrderStatus:
    OPEN = "open"
    CLOSED = "closed"
    ALL = "all"


class AlpacaBroker:
    """
    Thin compatibility layer over:
      - new SDK: `alpaca-py` (`alpaca.trading.TradingClient`)
      - old SDK: `alpaca_trade_api.REST`

    Exposes a stable, minimal surface for the rest of the bot:
      - list_open_orders()
      - list_orders(...)
      - cancel_order(order_id)
      - cancel_all_orders()
      - list_open_positions()
      - get_account()
      - submit_order(...)  (market/limit/stop/stop_limit supported)

    All methods return the SDK-native objects (no shape conversion),
    to minimize downstream changes.
    """

    def __init__(self, raw_api: Any):
        self._api = raw_api
        # Detect SDK flavor
        self._is_new = False
        try:
            # New SDK: `alpaca.trading.client.TradingClient`
            from alpaca.trading.client import TradingClient  # type: ignore
            self._is_new = isinstance(raw_api, TradingClient)
        except Exception:
            self._is_new = False

        # Optionals for new SDK (import lazily when used)
        self._GetOrdersRequest = None
        self._QueryOrderStatus = None
        self._OrderSide = None
        self._OrderType = None
        self._TimeInForce = None

    # ---------- Helpers (new SDK) ----------
    def _new_imports(self):
        if not self._is_new:
            return
        if self._GetOrdersRequest is None:
            from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLimitOrderRequest  # type: ignore
            from alpaca.trading.enums import QueryOrderStatus, OrderSide, OrderType, TimeInForce  # type: ignore
            self._GetOrdersRequest = GetOrdersRequest
            self._MarketOrderRequest = MarketOrderRequest
            self._LimitOrderRequest = LimitOrderRequest
            self._StopOrderRequest = StopOrderRequest
            self._StopLimitOrderRequest = StopLimitOrderRequest
            self._QueryOrderStatus = QueryOrderStatus
            self._OrderSide = OrderSide
            self._OrderType = OrderType
            self._TimeInForce = TimeInForce

    # ---------- Orders ----------
    def list_open_orders(self) -> Iterable[Any]:
        """
        Return open orders.
        """
        if self._is_new:
            self._new_imports()
            req = self._GetOrdersRequest(status=self._QueryOrderStatus.OPEN)
            return self._api.get_orders(req)
        # old SDK
        try:
            return self._api.list_orders(status="open")
        except AttributeError:
            raise RuntimeError("Alpaca API has neither get_orders nor list_orders")

    def list_orders(self, status: Optional[str] = None, limit: Optional[int] = None) -> Iterable[Any]:
        """
        Generic order listing with optional status and limit.
        status: 'open'|'closed'|'all' (old SDK words) or maps to new enums.
        """
        if self._is_new:
            self._new_imports()
            kwargs = {}
            if status:
                map_status = {
                    "open": self._QueryOrderStatus.OPEN,
                    "closed": self._QueryOrderStatus.CLOSED,
                    "all": self._QueryOrderStatus.ALL,
                }.get(status.lower(), self._QueryOrderStatus.ALL)
                kwargs["status"] = map_status
            if limit:
                kwargs["limit"] = limit
            req = self._GetOrdersRequest(**kwargs)
            return self._api.get_orders(req)
        # old SDK
        return self._api.list_orders(status=status or "all", limit=limit)

    def cancel_order(self, order_id: str) -> Any:
        if self._is_new:
            return self._api.cancel_order_by_id(order_id)
        return self._api.cancel_order(order_id)

    def cancel_all_orders(self) -> Any:
        if self._is_new:
            return self._api.cancel_orders()
        return self._api.cancel_all_orders()

    def get_order_by_id(self, order_id: str) -> Any:
        if self._is_new:
            return self._api.get_order_by_id(order_id)
        try:
            return self._api.get_order(order_id)
        except AttributeError:
            return self._api.get_order_by_id(order_id)

    # ---------- Positions & Account ----------
    def list_open_positions(self) -> Iterable[Any]:
        if self._is_new:
            return self._api.get_all_positions()
        try:
            return self._api.list_positions()
        except AttributeError:
            # Some old SDKs used `list_positions`; keep explicit error
            raise RuntimeError("Alpaca API has neither get_all_positions nor list_positions")

    def get_account(self) -> Any:
        if self._is_new:
            return self._api.get_account()
        return self._api.get_account()


def initialize(*args, **kwargs) -> "AlpacaBroker | None":
    """Create an :class:`AlpacaBroker` if the SDK is available."""
    if not ALPACA_SDK_AVAILABLE:
        return None
    client = _RealTradingClient(*args, **kwargs)
    return AlpacaBroker(client)

    # ---------- Submit orders ----------
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,               # 'buy' | 'sell'
        type: str = "market",    # 'market'|'limit'|'stop'|'stop_limit'
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        **extras: Any,
    ) -> Any:
        if self._is_new:
            self._new_imports()
            side_enum = self._OrderSide.BUY if side.lower() == "buy" else self._OrderSide.SELL
            tif_enum = getattr(self._TimeInForce, time_in_force.upper(), self._TimeInForce.DAY)

            t = type.lower()
            if t == "market":
                req = self._MarketOrderRequest(symbol=symbol, qty=qty, side=side_enum, time_in_force=tif_enum, client_order_id=client_order_id, **extras)
            elif t == "limit":
                if limit_price is None:
                    raise ValueError("limit_price is required for limit orders")
                req = self._LimitOrderRequest(symbol=symbol, qty=qty, side=side_enum, time_in_force=tif_enum, limit_price=limit_price, client_order_id=client_order_id, **extras)
            elif t == "stop":
                if stop_price is None:
                    raise ValueError("stop_price is required for stop orders")
                req = self._StopOrderRequest(symbol=symbol, qty=qty, side=side_enum, time_in_force=tif_enum, stop_price=stop_price, client_order_id=client_order_id, **extras)
            elif t == "stop_limit":
                if stop_price is None or limit_price is None:
                    raise ValueError("stop_limit requires stop_price and limit_price")
                req = self._StopLimitOrderRequest(symbol=symbol, qty=qty, side=side_enum, time_in_force=tif_enum, stop_price=stop_price, limit_price=limit_price, client_order_id=client_order_id, **extras)
            else:
                raise ValueError(f"unsupported order type: {type!r}")

            return self._api.submit_order(req)

        # old SDK path
        # old REST signature: submit_order(symbol, qty, side, type, time_in_force, limit_price=None, stop_price=None, client_order_id=None, **kwargs)
        return self._api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            client_order_id=client_order_id,
            **extras,
        )


__all__ = [
    "TradingClient",
    "MockTradingClient",
    "AlpacaBroker",
    "MockOrderSide",
    "MockTimeInForce",
    "MockOrderStatus",
    "MockQueryOrderStatus",
]
