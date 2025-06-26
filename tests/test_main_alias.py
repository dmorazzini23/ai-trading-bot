import importlib
import runpy
import sys
import types

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
    monkeypatch.setattr(importlib, "reload", lambda mod: mod)
    import main as main_mod
    main_mod = reload_module(main_mod)
    assert main_mod.create_flask_app() == "app"
    assert main_mod.main() == "main"


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
    monkeypatch.setattr(importlib, "reload", lambda mod: mod)
    runpy.run_module("main", run_name="__main__")
    assert called == [True]
