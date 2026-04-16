from __future__ import annotations

from ai_trading.data import price_quote_feed
from ai_trading.data import feed_roles


def test_reference_feed_defaults_to_delayed_sip(monkeypatch):
    monkeypatch.delenv("ALPACA_REFERENCE_FEED", raising=False)
    monkeypatch.setattr(feed_roles, "_settings_feed", lambda _attr: None)
    assert feed_roles.get_reference_feed() == "delayed_sip"


def test_execution_feed_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("ALPACA_EXECUTION_FEED", "sip")
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.setenv("ALPACA_SIP_UNAUTHORIZED", "0")
    monkeypatch.setattr(feed_roles, "_settings_feed", lambda _attr: None)
    assert feed_roles.get_execution_feed() == "sip"


def test_execution_feed_explicit_overrides_legacy_data_feed(monkeypatch):
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    monkeypatch.setenv("ALPACA_EXECUTION_FEED", "sip")
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.setenv("ALPACA_SIP_UNAUTHORIZED", "0")
    monkeypatch.setattr(feed_roles, "_settings_feed", lambda _attr: None)
    assert feed_roles.get_execution_feed() == "sip"


def test_price_quote_feed_cache_separates_execution_and_reference(monkeypatch):
    price_quote_feed.clear()
    monkeypatch.setattr(price_quote_feed, "resolve_alpaca_feed", lambda *_args, **_kwargs: "iex")
    assert price_quote_feed.resolve("AAPL", "iex", role="execution") == "iex"
    assert price_quote_feed.resolve("AAPL", "delayed_sip", role="reference") == "delayed_sip"
    assert price_quote_feed.get_cached("AAPL", role="execution") == "iex"
    assert price_quote_feed.get_cached("AAPL", role="reference") == "delayed_sip"


def test_reference_bars_feed_prefers_sip_when_entitled(monkeypatch):
    monkeypatch.setattr(feed_roles, "get_reference_feed", lambda *_args, **_kwargs: "delayed_sip")
    monkeypatch.setattr(feed_roles, "resolve_alpaca_feed", lambda *_args, **_kwargs: "sip")
    assert feed_roles.get_reference_bars_feed() == "sip"


def test_reference_bars_feed_falls_back_to_iex_when_sip_unavailable(monkeypatch):
    monkeypatch.setattr(feed_roles, "get_reference_feed", lambda *_args, **_kwargs: "delayed_sip")
    monkeypatch.setattr(feed_roles, "resolve_alpaca_feed", lambda *_args, **_kwargs: "iex")
    assert feed_roles.get_reference_bars_feed() == "iex"
