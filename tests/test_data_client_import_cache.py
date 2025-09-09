import importlib

import pytest


def test_data_client_import_failure_cached(monkeypatch):
    be = importlib.reload(
        __import__("ai_trading.core.bot_engine", fromlist=["dummy"])
    )
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ImportError("boom")

    monkeypatch.setattr(be, "get_data_client_cls", boom)
    monkeypatch.setattr(be, "_ALPACA_DATA_CLIENT_AVAILABLE", True)

    with pytest.raises(ImportError):
        be._get_data_client_cls_cached()
    with pytest.raises(ImportError):
        be._get_data_client_cls_cached()
    assert calls["n"] == 1
