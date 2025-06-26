# This test modifies sys.modules and should be run serially. Use: pytest -n0 tests/alias/test_main_alias.py
import importlib
import runpy
import types
import sys
from conftest import reload_module


def test_main_aliases(monkeypatch):
    run_mod = types.ModuleType("run")
    run_mod.create_flask_app = lambda: "app"
    run_mod.run_flask_app = lambda port: port
    run_mod.run_bot = lambda v, s: 0
    run_mod.validate_environment = lambda: None
    run_mod.main = lambda: "main"
    run_mod.__spec__ = importlib.util.spec_from_loader("run", loader=None)
    monkeypatch.setitem(sys.modules, "run", run_mod)
    original_reload = importlib.reload
    importlib.reload = lambda mod: mod
    try:
        import main as main_mod
        main_mod = reload_module(main_mod)
        assert main_mod.create_flask_app() == "app"
        assert main_mod.main() == "main"
    finally:
        importlib.reload = original_reload


def test_main_executes_run(monkeypatch):
    run_mod = types.ModuleType("run")
    called = []
    run_mod.main = lambda: called.append(True)
    run_mod.create_flask_app = lambda: None
    run_mod.run_flask_app = lambda port: None
    run_mod.run_bot = lambda v, s: 0
    run_mod.validate_environment = lambda: None
    run_mod.__spec__ = importlib.util.spec_from_loader("run", loader=None)
    monkeypatch.setitem(sys.modules, "run", run_mod)
    original_reload = importlib.reload
    importlib.reload = lambda mod: mod
    try:
        runpy.run_module("main", run_name="__main__")
    finally:
        importlib.reload = original_reload
    assert called == [True]
