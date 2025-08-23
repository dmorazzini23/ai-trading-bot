from __future__ import annotations

import builtins
import importlib


def test_indicators_import_without_numba(monkeypatch):
    """Ensure indicators module loads when numba is missing."""
    # AI-AGENT-REF: simulate missing numba to test optional import path
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "numba":
            raise ModuleNotFoundError("numba")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Reload module to hit guarded import
    importlib.sys.modules.pop("ai_trading.indicators", None)
    mod = importlib.import_module("ai_trading.indicators")

    assert hasattr(mod, "jit")
    assert getattr(mod, "_numba_jit", None) is None

