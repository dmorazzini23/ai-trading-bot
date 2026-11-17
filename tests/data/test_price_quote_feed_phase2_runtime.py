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


