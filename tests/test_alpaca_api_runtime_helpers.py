from __future__ import annotations

import builtins
import types
from typing import Any

import pytest

from ai_trading import alpaca_api


def test_active_namespace_falls_back_when_module_entry_is_not_module(monkeypatch):
    monkeypatch.setitem(alpaca_api.sys.modules, "ai_trading.alpaca_api", object())

    namespace = alpaca_api._active_alpaca_api_namespace()

    assert namespace is alpaca_api.__dict__


def test_get_http_session_caches_lazy_session_and_raises_when_missing(monkeypatch):
    session = types.SimpleNamespace()
    monkeypatch.setattr(alpaca_api, "_HTTP_SESSION", None)
    monkeypatch.setattr(alpaca_api, "_lazy_http_session", lambda: session)

    assert alpaca_api._get_http_session() is session
    assert alpaca_api._get_http_session() is session

    monkeypatch.setattr(alpaca_api, "_HTTP_SESSION", None)
    monkeypatch.setattr(alpaca_api, "_lazy_http_session", lambda: None)
    with pytest.raises(RuntimeError, match="HTTP session is not available yet"):
        alpaca_api._get_http_session()


def test_http_shim_adds_post_method_to_minimal_session(monkeypatch):
    calls: list[tuple[str, str, float | int | None, dict[str, object]]] = []

    class MinimalSession:
        def request(self, method, url, *args, timeout=None, **kwargs):
            calls.append((method, url, timeout, kwargs))
            return {"method": method, "url": url, "timeout": timeout}

    session = MinimalSession()
    monkeypatch.setattr(alpaca_api, "_HTTP_SESSION", session)

    post = alpaca_api._HTTPShim().post
    result = post("https://example.test/orders", json={"symbol": "AAPL"}, timeout=3)

    assert result == {"method": "POST", "url": "https://example.test/orders", "timeout": 3}
    assert calls == [("POST", "https://example.test/orders", 3, {"json": {"symbol": "AAPL"}})]
    assert getattr(session, "post") is post


def test_http_shim_sets_public_attributes_on_underlying_session(monkeypatch):
    session = types.SimpleNamespace()
    monkeypatch.setattr(alpaca_api, "_HTTP_SESSION", session)

    shim = alpaca_api._HTTPShim()
    shim.headers = {"Authorization": "Bearer token"}
    shim._local = "value"

    assert session.headers == {"Authorization": "Bearer token"}
    assert shim._local == "value"


def test_managed_env_delegates_to_config_get_env(monkeypatch):
    from ai_trading.config import management as config_management

    calls: list[tuple[str, object, object, bool]] = []

    def fake_get_env(key, default=None, *, cast=None, resolve_aliases=True):
        calls.append((key, default, cast, resolve_aliases))
        return "resolved"

    monkeypatch.setattr(config_management, "get_env", fake_get_env)

    result = alpaca_api._managed_env(
        "ALPACA_DATA_FEED",
        "iex",
        cast=str,
        resolve_aliases=False,
    )

    assert result == "resolved"
    assert calls == [("ALPACA_DATA_FEED", "iex", str, False)]


def test_shadow_mode_and_env_resolver_fallbacks(monkeypatch):
    from ai_trading.config import management as config_management

    monkeypatch.setattr(config_management, "is_shadow_mode", lambda: "yes")
    assert alpaca_api.is_shadow_mode() is True

    monkeypatch.delattr(config_management, "is_shadow_mode", raising=False)
    assert alpaca_api.is_shadow_mode() is False

    monkeypatch.setattr(
        config_management,
        "_resolve_alpaca_env",
        lambda: ("key", "secret", "https://paper-api.alpaca.markets"),
        raising=False,
    )
    assert alpaca_api._resolve_alpaca_env() == (
        "key",
        "secret",
        "https://paper-api.alpaca.markets",
    )

    monkeypatch.setattr(
        config_management,
        "_resolve_alpaca_env",
        lambda: ("bad", "shape"),
        raising=False,
    )
    assert alpaca_api._resolve_alpaca_env() == alpaca_api._default_resolve_alpaca_env()


def test_make_client_order_id_uses_minute_bucket_and_uuid_prefix(monkeypatch):
    monkeypatch.setattr(alpaca_api.time, "time", lambda: 125.0)
    monkeypatch.setattr(
        alpaca_api.uuid,
        "uuid4",
        lambda: types.SimpleNamespace(hex="abcdef1234567890"),
    )

    assert alpaca_api._make_client_order_id("trade") == "trade-2-abcdef12"


def test_get_trading_client_cls_returns_unavailable_sentinel(monkeypatch):
    monkeypatch.setattr(alpaca_api, "_ensure_trading_client_cls", lambda: None)

    cls = alpaca_api.get_trading_client_cls()

    with pytest.raises(RuntimeError, match="TradingClient not available"):
        cls()


def test_get_trading_client_cls_returns_loaded_class(monkeypatch):
    class FakeTradingClient:
        pass

    monkeypatch.setattr(alpaca_api, "_ensure_trading_client_cls", lambda: FakeTradingClient)

    assert alpaca_api.get_trading_client_cls() is FakeTradingClient


def test_trading_client_adapter_proxies_helpers_and_cancel_by_id():
    class Client:
        def __init__(self):
            self.cancelled = []

        def list_orders(self):
            return ["order"]

        def list_positions(self):
            return ["position"]

        def cancel_order_by_id(self, order_id):
            self.cancelled.append(order_id)
            return {"cancelled": order_id}

    client = Client()
    adapter = alpaca_api.TradingClientAdapter(client)

    assert adapter.list_orders() == ["order"]
    assert adapter.list_positions() == ["position"]
    assert adapter.cancel_order("ord-1") == {"cancelled": "ord-1"}
    assert client.cancelled == ["ord-1"]
    assert adapter._ai_trading_wrapped_client is client
    assert getattr(adapter, "__ai_trading_adapter__") == "trading_client"


def test_trading_client_adapter_uses_cancel_orders_request_variants(monkeypatch):
    monkeypatch.setattr(alpaca_api, "AI_TRADING_FALLBACK_EXCEPTIONS", (ImportError,))
    captured: list[Any] = []

    class Client:
        def cancel_orders(self, request=None):
            captured.append(request)
            return {"ok": True, "payload": request.payload}

    result = alpaca_api.TradingClientAdapter(Client()).cancel_order("ord-2")

    assert result == {"ok": True, "payload": {"order_id": "ord-2"}}
    assert captured[0].payload == {"order_id": "ord-2"}


def test_trading_client_adapter_tries_keyword_cancel_order_signatures(monkeypatch):
    monkeypatch.setattr(alpaca_api, "AI_TRADING_FALLBACK_EXCEPTIONS", (ImportError,))
    calls: list[tuple[Any | None, Any | None]] = []

    class Client:
        def cancel_orders(self, request=None, cancel_orders_request=None):
            calls.append((request, cancel_orders_request))
            if request is not None:
                raise TypeError("request keyword unsupported")
            return {"payload": cancel_orders_request.payload}

    result = alpaca_api.TradingClientAdapter(Client()).cancel_order("ord-3")

    assert result == {"payload": {"order_id": "ord-3"}}
    assert calls[0][0].payload == {"order_id": "ord-3"}
    assert calls[1][0].payload == {"order_id": "ord-3"}
    assert calls[2][1].payload == {"order_id": "ord-3"}


def test_trading_client_adapter_reports_unadaptable_cancel_orders(monkeypatch):
    monkeypatch.setattr(alpaca_api, "AI_TRADING_FALLBACK_EXCEPTIONS", (ImportError,))

    class Client:
        def cancel_orders(self, *_args, **_kwargs):
            raise TypeError("unsupported")

    with pytest.raises(RuntimeError, match="could not adapt"):
        alpaca_api.TradingClientAdapter(Client()).cancel_order("ord-4")


def test_trading_client_adapter_requires_cancel_support():
    with pytest.raises(AttributeError, match="cancel_order not supported"):
        alpaca_api.TradingClientAdapter(object()).cancel_order("ord-5")


def test_lazy_client_fallback_classes_raise_clear_errors(monkeypatch):
    monkeypatch.setattr(alpaca_api, "AI_TRADING_FALLBACK_EXCEPTIONS", (ImportError,))
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {
            "alpaca.data.historical.stock",
            "alpaca.common.exceptions",
        }:
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    DataClient = alpaca_api.get_data_client_cls()
    data_client = DataClient()
    with pytest.raises(RuntimeError, match="StockHistoricalDataClient not available"):
        data_client.get_stock_bars()

    APIError = alpaca_api.get_api_error_cls()
    assert issubclass(APIError, Exception)
