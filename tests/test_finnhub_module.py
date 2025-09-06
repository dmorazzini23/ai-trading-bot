import importlib
import logging
import types
import sys

import pytest


def _reload_module():
    import ai_trading.data.finnhub as fh
    return importlib.reload(fh)


def test_disabled_logs_once(monkeypatch, caplog):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setenv("ENABLE_FINNHUB", "0")
    sys.modules.pop("finnhub", None)
    with caplog.at_level(logging.DEBUG):
        mod = _reload_module()
    assert getattr(mod.fh_fetcher, "is_stub", False)
    assert mod._SENT_DEPS_LOGGED == {"finnhub"}
    assert any(r.message == "FINNHUB_DISABLED" for r in caplog.records)


def test_enabled_fetcher(monkeypatch):
    monkeypatch.setenv("ENABLE_FINNHUB", "1")
    monkeypatch.setenv("FINNHUB_API_KEY", "test")
    finnhub_stub = types.ModuleType("finnhub")
    finnhub_stub.Client = lambda key: object()
    monkeypatch.setitem(sys.modules, "finnhub", finnhub_stub)
    mod = _reload_module()
    assert not getattr(mod.fh_fetcher, "is_stub", False)
    assert mod._SENT_DEPS_LOGGED == set()
