import importlib
import sys


def test_package_import_has_no_cli_side_effects(monkeypatch):
    for name in list(sys.modules):
        if name.startswith("ai_trading"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.import_module("ai_trading")
    heavy_modules = {
        "ai_trading.__main__",
        "ai_trading.app",
        "ai_trading.main",
        "ai_trading.production_system",
        "ai_trading.core.run_all_trades",
    }
    assert heavy_modules.isdisjoint(sys.modules)
