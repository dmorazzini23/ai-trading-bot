from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from ai_trading.core import bot_engine
from ai_trading.telemetry import runtime_state


@pytest.fixture(autouse=True)
def _reset_runtime_state():
    runtime_state.update_quote_status(allowed=False, reason=None, status="reset", synthetic=False)


def test_fetch_quote_unwraps_symbol_keyed_mapping(monkeypatch):
    class _Quote:
        bid_price = 100.0
        ask_price = 100.2

    class _Request:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    payload = {"AAPL": _Quote()}
    ctx = SimpleNamespace(
        data_client=SimpleNamespace(get_stock_latest_quote=lambda _req: payload)
    )
    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: True)
    monkeypatch.setattr(bot_engine, "StockLatestQuoteRequest", _Request, raising=False)

    resolved = bot_engine._fetch_quote(cast(Any, ctx), "AAPL", feed="iex")
    assert resolved is payload["AAPL"]
    assert resolved.bid_price == 100.0
    assert resolved.ask_price == 100.2


def test_fetch_quote_unwraps_quotes_mapping_container(monkeypatch):
    class _Quote:
        bid_price = 200.0
        ask_price = 200.4

    class _Request:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    payload = SimpleNamespace(quotes={"MSFT": _Quote()})
    ctx = SimpleNamespace(
        data_client=SimpleNamespace(get_stock_latest_quote=lambda _req: payload)
    )
    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: True)
    monkeypatch.setattr(bot_engine, "StockLatestQuoteRequest", _Request, raising=False)

    resolved = bot_engine._fetch_quote(cast(Any, ctx), "MSFT")
    assert resolved is payload.quotes["MSFT"]
    assert resolved.bid_price == 200.0
    assert resolved.ask_price == 200.4


def test_fetch_quote_falls_back_to_global_data_client(monkeypatch):
    class _Quote:
        bid_price = 150.0
        ask_price = 150.3

    class _Request:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _GlobalClient:
        def get_stock_latest_quote(self, _req):
            return {"AAPL": _Quote()}

    ctx = SimpleNamespace(data_client=None)
    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: True)
    monkeypatch.setattr(bot_engine, "StockLatestQuoteRequest", _Request, raising=False)
    monkeypatch.setattr(bot_engine, "data_client", _GlobalClient(), raising=False)

    resolved = bot_engine._fetch_quote(cast(Any, ctx), "AAPL", feed="iex")
    assert resolved is not None
    assert resolved.bid_price == 150.0
    assert resolved.ask_price == 150.3

