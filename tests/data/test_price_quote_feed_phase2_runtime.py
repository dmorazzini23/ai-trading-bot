from __future__ import annotations

import builtins
import sys
from types import ModuleType

import pytest

from ai_trading.data import price_quote_feed


@pytest.fixture(autouse=True)
def _clear_feed_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    price_quote_feed.clear()
    monkeypatch.setattr(price_quote_feed, "get_env", lambda *_args, **_kwargs: "")


def test_normalize_role_and_feed_aliases() -> None:
    assert price_quote_feed._normalize_role(" reference ") == "reference"
    assert price_quote_feed._normalize_role("anything") == "execution"
    assert price_quote_feed._normalize_feed(None) is None
    assert price_quote_feed._normalize_feed(" ALPACA_SIP ") == "sip"
    assert price_quote_feed._normalize_feed("alpaca_delayed_sip", role="reference") == "delayed_sip"
    assert price_quote_feed._normalize_feed("delayed-sip", role="reference") == "delayed_sip"
    assert price_quote_feed._normalize_feed("delayed", role="execution") is None
    assert price_quote_feed._normalize_feed("bad") is None


def test_pytest_and_sip_disabled_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(price_quote_feed, "get_env", lambda *_args, **_kwargs: "yes")
    assert price_quote_feed._pytest_mode() is True

    monkeypatch.setattr(
        price_quote_feed,
        "resolve_alpaca_feed",
        lambda _requested: (_ for _ in ()).throw(RuntimeError("bad env")),
    )
    assert price_quote_feed._sip_disabled_env(py_mode=True) is True


def test_reference_feed_resolution_cache_and_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(price_quote_feed, "resolve_alpaca_feed", lambda _requested: "iex")
    monkeypatch.setattr(
        price_quote_feed,
        "resolve_alpaca_reference_feed",
        lambda _requested: "delayed_sip",
    )

    assert price_quote_feed.ensure_entitled_feed("sip", None, role="reference") == "delayed_sip"
    assert price_quote_feed.ensure_entitled_feed("iex", None, role="reference") == "iex"
    assert price_quote_feed.ensure_entitled_feed(None, "sip", role="reference") == "sip"
    assert price_quote_feed.ensure_entitled_feed(None, None, role="reference") == "delayed_sip"
    assert price_quote_feed.resolve("AAPL", "delayed", role="reference") == "delayed_sip"
    assert price_quote_feed.get_cached("AAPL", role="reference") == "delayed_sip"
    price_quote_feed.clear("AAPL", role="reference")
    assert price_quote_feed.get_cached("AAPL", role="reference") is None


def test_execution_feed_resolution_uses_cached_and_unauthorized_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(price_quote_feed, "resolve_alpaca_feed", lambda _requested: "sip")
    monkeypatch.setattr(price_quote_feed, "_sip_unauthorized", lambda: False)

    assert price_quote_feed.ensure_entitled_feed("sip", None) == "sip"
    assert price_quote_feed.ensure_entitled_feed("iex", None) == "iex"
    assert price_quote_feed.cache("AAPL", "alpaca_sip") == "sip"
    assert price_quote_feed.resolve("AAPL", None) == "sip"
    assert price_quote_feed.get_cached("AAPL") == "sip"

    monkeypatch.setattr(price_quote_feed, "resolve_alpaca_feed", lambda _requested: "iex")
    assert price_quote_feed.ensure_entitled_feed("sip", None) == "iex"
    assert price_quote_feed.ensure_entitled_feed(None, "sip") == "iex"
    assert price_quote_feed._apply_sip_guard("iex") == "iex"
    assert price_quote_feed._apply_sip_guard("sip") == "iex"
    assert price_quote_feed._fallback_feed() == "iex"

    price_quote_feed.clear()
    assert price_quote_feed.get_cached("AAPL") is None


def test_sip_unauthorized_fetch_state_and_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    monkeypatch.setattr(price_quote_feed, "get_env", lambda *_args, **_kwargs: "")
    import ai_trading.data as data_pkg

    fake_fetch = ModuleType("ai_trading.data.fetch")
    fake_fetch._state = {"sip_unauthorized": True}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_trading.data.fetch", fake_fetch)
    monkeypatch.setattr(data_pkg, "fetch", fake_fetch)

    assert price_quote_feed._sip_unauthorized() is True

    fake_fetch._state = {}  # type: ignore[attr-defined]
    fake_fetch._SIP_UNAUTHORIZED = True  # type: ignore[attr-defined]
    assert price_quote_feed._sip_unauthorized() is True

    monkeypatch.delitem(sys.modules, "ai_trading.data.fetch", raising=False)
    monkeypatch.delattr(data_pkg, "fetch", raising=False)
    original_import = builtins.__import__
    monkeypatch.setattr(
        builtins,
        "__import__",
        lambda name, *args, **kwargs: (_ for _ in ()).throw(ImportError("bad"))
        if name == "ai_trading.data"
        else original_import(name, *args, **kwargs),
    )
    assert price_quote_feed._sip_unauthorized() is False


def test_resolve_removes_cache_when_entitlement_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    price_quote_feed._FEED_CACHE_BY_ROLE["execution"]["AAPL"] = "iex"
    monkeypatch.setattr(price_quote_feed, "ensure_entitled_feed", lambda *_args, **_kwargs: None)

    assert price_quote_feed.resolve("AAPL", None) is None
    assert price_quote_feed.cache("MSFT", None) is None
    assert price_quote_feed.get_cached("AAPL") is None
