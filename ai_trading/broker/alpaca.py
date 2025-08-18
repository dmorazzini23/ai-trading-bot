from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ai_trading.alpaca_api import ALPACA_AVAILABLE

try:  # AI-AGENT-REF: guard Alpaca dependency
    from alpaca.common.exceptions import APIError  # type: ignore
    from alpaca.trading.client import TradingClient  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TradingClient = None  # type: ignore

    class APIError(Exception):  # AI-AGENT-REF: fallback when SDK missing
        ...


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
        self._is_new = TradingClient is not None and isinstance(raw_api, TradingClient)

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
            from alpaca.trading.enums import (  # type: ignore
                OrderSide,
                OrderType,
                QueryOrderStatus,
                TimeInForce,
            )
            from alpaca.trading.requests import (  # type: ignore
                GetOrdersRequest,
                LimitOrderRequest,
                MarketOrderRequest,
                StopLimitOrderRequest,
                StopOrderRequest,
            )

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

    def list_orders(
        self, status: str | None = None, limit: int | None = None
    ) -> Iterable[Any]:
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
            raise RuntimeError(
                "Alpaca API has neither get_all_positions nor list_positions"
            )

    def get_account(self) -> Any:
        if self._is_new:
            return self._api.get_account()
        return self._api.get_account()


def initialize(*args, **kwargs) -> AlpacaBroker | None:
    """Create an :class:`AlpacaBroker` if the SDK is available."""
    if not (ALPACA_AVAILABLE and TradingClient):
        return None
    client = TradingClient(*args, **kwargs)
    return AlpacaBroker(client)


__all__ = [
    "TradingClient",
    "AlpacaBroker",
]
