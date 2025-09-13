from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("MAX_DRAWDOWN_THRESHOLD", "0")

from ai_trading.alpaca_api import list_orders_wrapper


class _StatusClient:
    def __init__(self):
        self.kwargs = None

    def get_orders(self, *args, **kwargs):
        self.kwargs = kwargs
        return ["ok"]


def test_list_orders_wrapper_passes_status():
    client = _StatusClient()
    orders = list_orders_wrapper(client, status="open")
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
    orders = list_orders_wrapper(client, status="open")
    assert orders == ["ok"]
    assert isinstance(client.filter, GetOrdersRequest)
    assert client.filter.statuses == [QueryOrderStatus.OPEN]


def test_list_orders_wrapper_fallback_on_missing_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "alpaca", types.ModuleType("alpaca"))
    monkeypatch.setitem(sys.modules, "alpaca.trading", types.ModuleType("alpaca.trading"))
    monkeypatch.delitem(sys.modules, "alpaca.trading.enums", raising=False)
    monkeypatch.delitem(sys.modules, "alpaca.trading.requests", raising=False)

    class _FilterClient:
        def __init__(self):
            self.kwargs = None

        def get_orders(self, *args, filter=None, **kwargs):
            self.kwargs = {"filter": filter, **kwargs}
            return ["ok"]

    client = _FilterClient()
    orders = list_orders_wrapper(client, status="open")
    assert orders == ["ok"]
    assert client.kwargs == {"filter": None, "status": "open"}
