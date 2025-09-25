import logging
from types import SimpleNamespace

import ai_trading.logging.emit_once as emit_once_module
from ai_trading.data import fetch


def _reset_emit_once(monkeypatch):
    monkeypatch.setattr(emit_once_module, "_emitted", {}, raising=False)


def test_missing_alpaca_warning_suppressed_for_backup_provider(monkeypatch):
    _reset_emit_once(monkeypatch)
    monkeypatch.delenv("ALPACA_SIP_UNAUTHORIZED", raising=False)
    monkeypatch.setattr(fetch, "_is_sip_unauthorized", lambda: False)
    monkeypatch.setattr(fetch, "get_settings", lambda: SimpleNamespace(data_provider="yahoo"))
    monkeypatch.setattr(fetch, "get_data_feed_override", lambda: "yahoo")
    monkeypatch.setattr(fetch, "resolve_alpaca_feed", lambda _requested=None: None)

    should_warn, extra = fetch._missing_alpaca_warning_context()

    assert should_warn is False
    assert extra.get("provider") == "yahoo"


def test_missing_alpaca_warning_suppressed_when_sip_locked(monkeypatch):
    _reset_emit_once(monkeypatch)
    monkeypatch.setenv("ALPACA_SIP_UNAUTHORIZED", "1")
    monkeypatch.setattr(fetch, "_is_sip_unauthorized", lambda: True)
    monkeypatch.setattr(fetch, "get_settings", lambda: SimpleNamespace(data_provider="alpaca"))
    monkeypatch.setattr(fetch, "get_data_feed_override", lambda: None)
    monkeypatch.setattr(fetch, "resolve_alpaca_feed", lambda _requested=None: "sip")

    should_warn, extra = fetch._missing_alpaca_warning_context()

    assert should_warn is False
    assert extra.get("sip_locked") is True
    monkeypatch.delenv("ALPACA_SIP_UNAUTHORIZED", raising=False)


def test_warn_missing_alpaca_emits_once(monkeypatch, caplog):
    _reset_emit_once(monkeypatch)
    monkeypatch.delenv("ALPACA_SIP_UNAUTHORIZED", raising=False)
    monkeypatch.setattr(fetch, "_is_sip_unauthorized", lambda: False)
    monkeypatch.setattr(fetch, "get_settings", lambda: SimpleNamespace(data_provider="alpaca"))
    monkeypatch.setattr(fetch, "get_data_feed_override", lambda: None)
    monkeypatch.setattr(fetch, "resolve_alpaca_feed", lambda _requested=None: "iex")

    caplog.set_level(logging.WARNING)

    fetch._warn_missing_alpaca("FAKE", "1Min")
    fetch._warn_missing_alpaca("FAKE", "1Min")

    messages = [record.message for record in caplog.records]
    assert messages.count("ALPACA_API_KEY_MISSING") == 1
