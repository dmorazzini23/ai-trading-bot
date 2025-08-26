import importlib
import builtins
import sys
import types
import pytest

@pytest.mark.parametrize(
    "module",
    [
        "ai_trading.features.pipeline",
        "ai_trading.portfolio.sizing",
        "ai_trading.monitoring.metrics",
    ],
)
def test_module_import_without_pandas(monkeypatch, module):
    """Modules should import even if pandas is unavailable."""
    pytest.importorskip("pandas")
    if module == "ai_trading.features.pipeline":
        pytest.importorskip("sklearn")
    for key in ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL", "WEBHOOK_SECRET"]:
        monkeypatch.setenv(key, "test")
    monkeypatch.delitem(sys.modules, "pandas", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("pandas"):
            raise ModuleNotFoundError("No module named 'pandas'")
        try:
            return real_import(name, *args, **kwargs)
        except ModuleNotFoundError:
            return types.ModuleType(name)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, module, raising=False)
    importlib.import_module(module)
