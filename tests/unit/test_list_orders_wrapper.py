from __future__ import annotations

import sys
import types

from ai_trading.core.bot_engine import _validate_trading_api, list_open_orders


class _StatusClient:
    def __init__(self):
        self.kwargs = None

    def get_orders(self, *args, **kwargs):
        self.kwargs = kwargs
        return ["ok"]


def test_list_orders_wrapper_passes_status():
    client = _StatusClient()
    assert _validate_trading_api(client)
    orders = list_open_orders(client)
    assert orders == ["ok"]
    assert client.kwargs == {"status": "open"}


def test_list_orders_wrapper_builds_filter(monkeypatch):
    enums_mod = types.ModuleType("alpaca.trading.enums")
    requests_mod = types.ModuleType("alpaca.trading.requests")

    class QueryOrderStatus:
        OPEN = "open"

    class GetOrdersRequest:
        def __init__(self, *, statuses=None):
            self.statuses = statuses

    enums_mod.QueryOrderStatus = QueryOrderStatus
    requests_mod.GetOrdersRequest = GetOrdersRequest
    monkeypatch.setitem(sys.modules, "alpaca", types.ModuleType("alpaca"))
    monkeypatch.setitem(sys.modules, "alpaca.trading", types.ModuleType("alpaca.trading"))
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", enums_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", requests_mod)

    class _FilterClient:
        def __init__(self):
            self.filter = None

        def get_orders(self, *args, filter=None, **kwargs):
            self.filter = filter
            return ["ok"]

    client = _FilterClient()
    assert _validate_trading_api(client)
    orders = list_open_orders(client)
    assert orders == ["ok"]
    assert isinstance(client.filter, GetOrdersRequest)
    assert client.filter.statuses == [QueryOrderStatus.OPEN]
