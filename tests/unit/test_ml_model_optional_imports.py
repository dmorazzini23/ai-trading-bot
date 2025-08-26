import builtins
import importlib
import sys


def test_ml_model_import_without_heavy_deps(monkeypatch):
    missing = {"joblib", "numpy", "pandas"}
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in missing:
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for mod in ["ai_trading.ml_model", *missing]:
        monkeypatch.delitem(sys.modules, mod, raising=False)

    module = importlib.import_module("ai_trading.ml_model")
    assert hasattr(module, "MLModel")
