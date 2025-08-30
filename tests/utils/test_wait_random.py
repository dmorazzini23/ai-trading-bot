from __future__ import annotations

import builtins
import importlib
import sys


def test_wait_random_positional_without_tenacity(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tenacity":
            raise ImportError("tenacity not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "tenacity", raising=False)
    monkeypatch.delitem(sys.modules, "ai_trading.utils.retry", raising=False)

    retry_mod = importlib.import_module("ai_trading.utils.retry")
    assert retry_mod.HAS_TENACITY is False

    wait = retry_mod.wait_random(0, 5)
    value = wait(1)
    assert 0 <= value <= 5
