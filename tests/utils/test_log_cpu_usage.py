import builtins
import importlib
import sys

import logging

from ai_trading.logging import get_logger


def test_import_base_without_system_health(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "psutil":
            raise ImportError("psutil missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("ai_trading.monitoring.system_health", None)
    sys.modules.pop("ai_trading.utils.base", None)
    base = importlib.import_module("ai_trading.utils.base")
    assert "ai_trading.monitoring.system_health" not in sys.modules
    assert base.logger


def test_log_cpu_usage_without_psutil(monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "psutil":
            raise ImportError("psutil missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("ai_trading.monitoring.system_health", None)
    from ai_trading.utils import base

    lg = get_logger("test")
    with caplog.at_level(logging.DEBUG):
        base.log_cpu_usage(lg)
    assert not any("CPU_USAGE" in r.message for r in caplog.records)
