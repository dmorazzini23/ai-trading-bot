import importlib
import sys


def test_package_import_has_no_cli_side_effects(monkeypatch):
    for name in list(sys.modules):
        if name.startswith("ai_trading"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.import_module("ai_trading")
    assert "ai_trading.__main__" not in sys.modules
