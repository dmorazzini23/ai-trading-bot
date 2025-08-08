import importlib
import sys
from types import ModuleType

def test_engine_imports_without_meta_learning(monkeypatch):
    # Simulate missing module to ensure optional behavior works
    class _Missing(ModuleType): pass
    sys.modules.pop("ai_trading.meta_learning", None)
    monkeypatch.setitem(sys.modules, "ai_trading.meta_learning", _Missing("ai_trading.meta_learning"))
    # Import should not crash
    eng = importlib.import_module("ai_trading.core.bot_engine")
    assert hasattr(eng, "logger")

def test_optimize_signals_fallback(monkeypatch):
    # Force missing meta module -> fallback should return input unchanged
    sys.modules.pop("ai_trading.meta_learning", None)
    import importlib
    eng = importlib.import_module("ai_trading.core.bot_engine")
    dummy = [{"sym":"AAPL","score":0.5}, {"sym":"MSFT","score":0.4}]
    out = eng.optimize_signals(dummy)  # type: ignore[attr-defined]
    assert out == dummy