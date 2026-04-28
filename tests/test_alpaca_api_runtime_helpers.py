from __future__ import annotations

import builtins
import types

import pytest

from ai_trading import alpaca_api


def _alpaca_api_module() -> types.ModuleType:
    """Return the real Alpaca API module even if a lazy proxy is installed."""

    real = getattr(alpaca_api, "_real", None)
    if isinstance(real, types.ModuleType):
        return real
    load = getattr(alpaca_api, "_load", None)
    if callable(load):
        loaded = load()
        if isinstance(loaded, types.ModuleType):
            return loaded
    return alpaca_api


def test_active_namespace_falls_back_when_module_entry_is_not_module(monkeypatch):
    api = _alpaca_api_module()
    monkeypatch.setitem(api.sys.modules, "ai_trading.alpaca_api", object())

    namespace = api._active_alpaca_api_namespace()

    assert namespace is api.__dict__


def test_get_http_session_caches_lazy_session_and_raises_when_missing(monkeypatch):
    api = _alpaca_api_module()
    session = types.SimpleNamespace()
    monkeypatch.setattr(api, "_HTTP_SESSION", None)
    monkeypatch.setattr(api, "_lazy_http_session", lambda: session)

    assert api._get_http_session() is session
    assert api._get_http_session() is session

    monkeypatch.setattr(api, "_HTTP_SESSION", None)
    monkeypatch.setattr(api, "_lazy_http_session", lambda: None)
    with pytest.raises(RuntimeError, match="HTTP session is not available yet"):
        api._get_http_session()


def test_http_submit_requires_native_session_post(monkeypatch):
    api = _alpaca_api_module()

    class MinimalSession:
        def request(self, method, url, *args, timeout=None, **kwargs):
            raise AssertionError("request fallback should not be used")

    monkeypatch.setattr(api, "_get_http_session", lambda: MinimalSession())
    cfg = api._AlpacaConfig(
        base_url="https://paper-api.alpaca.markets",
        key_id="key",
        secret_key="secret",
        shadow=False,
    )

    with pytest.raises(AttributeError, match="post"):
        api._http_submit(
            cfg,
            symbol="AAPL",
            qty=1,
            side="buy",
            type="market",
            time_in_force="day",
            limit_price=None,
            stop_price=None,
            idempotency_key=None,
            timeout=3,
        )


def test_managed_env_delegates_to_config_get_env(monkeypatch):
    api = _alpaca_api_module()
    from ai_trading.config import management as config_management

    calls: list[tuple[str, object, object, bool]] = []

    def fake_get_env(key, default=None, *, cast=None, resolve_aliases=True):
        calls.append((key, default, cast, resolve_aliases))
        return "resolved"

    monkeypatch.setattr(config_management, "get_env", fake_get_env)

    result = api._managed_env(
        "ALPACA_DATA_FEED",
        "iex",
        cast=str,
        resolve_aliases=False,
    )

    assert result == "resolved"
    assert calls == [("ALPACA_DATA_FEED", "iex", str, False)]


def test_shadow_mode_and_env_resolver_fallbacks(monkeypatch):
    api = _alpaca_api_module()
    from ai_trading.config import management as config_management

    monkeypatch.setattr(config_management, "is_shadow_mode", lambda: "yes")
    assert api.is_shadow_mode() is True

    monkeypatch.delattr(config_management, "is_shadow_mode", raising=False)
    assert api.is_shadow_mode() is False

    monkeypatch.setattr(
        config_management,
        "_resolve_alpaca_env",
        lambda: ("key", "secret", "https://paper-api.alpaca.markets"),
        raising=False,
    )
    assert api._resolve_alpaca_env() == (
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
    assert api._resolve_alpaca_env() == api._default_resolve_alpaca_env()


def test_make_client_order_id_uses_minute_bucket_and_uuid_prefix(monkeypatch):
    api = _alpaca_api_module()
    monkeypatch.setattr(api.time, "time", lambda: 125.0)
    monkeypatch.setattr(
        api.uuid,
        "uuid4",
        lambda: types.SimpleNamespace(hex="abcdef1234567890"),
    )

    assert api._make_client_order_id("trade") == "trade-2-abcdef12"


def test_get_trading_client_cls_fails_when_sdk_unavailable(monkeypatch):
    api = _alpaca_api_module()

    def _missing():
        raise RuntimeError("alpaca-py==0.42.1 is required")

    monkeypatch.setattr(api, "_ensure_trading_client_cls", _missing)

    with pytest.raises(RuntimeError, match="alpaca-py==0.42.1 is required"):
        api.get_trading_client_cls()


def test_get_trading_client_cls_returns_loaded_class(monkeypatch):
    api = _alpaca_api_module()

    class FakeTradingClient:
        pass

    monkeypatch.setattr(api, "_ensure_trading_client_cls", lambda: FakeTradingClient)

    assert api.get_trading_client_cls() is FakeTradingClient


def test_trading_client_adapter_is_not_exported():
    assert not hasattr(alpaca_api, "TradingClientAdapter")


def test_http_shim_is_not_exported():
    assert not hasattr(alpaca_api, "_HTTPShim")
    assert not hasattr(alpaca_api, "_HTTP")


def test_native_trading_client_methods_are_used_directly():
    class Client:
        def __init__(self):
            self.cancelled = []

        def get_orders(self):
            return ["order"]

        def get_all_positions(self):
            return ["position"]

        def cancel_order_by_id(self, order_id):
            self.cancelled.append(order_id)
            return {"cancelled": order_id}

    client = Client()

    assert client.get_orders() == ["order"]
    assert client.get_all_positions() == ["position"]
    assert client.cancel_order_by_id("ord-1") == {"cancelled": "ord-1"}
    assert client.cancelled == ["ord-1"]


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

    with pytest.raises(RuntimeError, match="alpaca-py==0.42.1 is required"):
        alpaca_api.get_data_client_cls()

    with pytest.raises(RuntimeError, match="alpaca-py==0.42.1 is required"):
        alpaca_api.get_api_error_cls()
