from __future__ import annotations

import builtins
import importlib
import sys


def test_import_system_health_without_psutil(monkeypatch):
    """Import system_health when psutil is missing should not raise ImportError."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("psutil not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "psutil", raising=False)
    monkeypatch.delitem(sys.modules, "ai_trading.monitoring.system_health", raising=False)

    module = importlib.import_module("ai_trading.monitoring.system_health")
    assert module.snapshot_basic()["has_psutil"] is False
