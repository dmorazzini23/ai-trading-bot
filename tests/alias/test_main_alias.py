# This test modifies sys.modules and should be run serially. Use: pytest -n0 tests/alias/test_main_alias.py
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
    dummy_client = types.ModuleType("alpaca.trading.client")
    dummy_client.TradingClient = object
    monkeypatch.setitem(sys.modules, "alpaca", types.ModuleType("alpaca"))
    trading_mod = types.ModuleType("alpaca.trading")
    trading_mod.__path__ = []
    monkeypatch.setitem(sys.modules, "alpaca.trading", trading_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", dummy_client)
    dummy_requests = types.ModuleType("requests")
    dummy_requests.get = lambda *a, **k: None
    exc_mod = types.ModuleType("requests.exceptions")
    exc_mod.HTTPError = Exception
    dummy_requests.exceptions = exc_mod
    monkeypatch.setitem(sys.modules, "requests.exceptions", exc_mod)
    monkeypatch.setitem(sys.modules, "requests", dummy_requests)
    dummy_stream = types.ModuleType("alpaca.trading.stream")
    dummy_stream.TradingStream = object
    monkeypatch.setitem(sys.modules, "alpaca.trading.stream", dummy_stream)
    original_reload = importlib.reload
    importlib.reload = lambda mod: mod
    try:
        import main as main_mod
        main_mod = reload_module(main_mod)
        assert isinstance(main_mod.create_flask_app(), (str, object))
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
    dummy_client = types.ModuleType("alpaca.trading.client")
    dummy_client.TradingClient = object
    monkeypatch.setitem(sys.modules, "alpaca", types.ModuleType("alpaca"))
    trading_mod = types.ModuleType("alpaca.trading")
    trading_mod.__path__ = []
    monkeypatch.setitem(sys.modules, "alpaca.trading", trading_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", dummy_client)
    dummy_requests = types.ModuleType("requests")
    dummy_requests.get = lambda *a, **k: None
    exc_mod = types.ModuleType("requests.exceptions")
    exc_mod.HTTPError = Exception
    dummy_requests.exceptions = exc_mod
    monkeypatch.setitem(sys.modules, "requests.exceptions", exc_mod)
    monkeypatch.setitem(sys.modules, "requests", dummy_requests)
    dummy_stream = types.ModuleType("alpaca.trading.stream")
    dummy_stream.TradingStream = object
    monkeypatch.setitem(sys.modules, "alpaca.trading.stream", dummy_stream)
    original_reload = importlib.reload
    importlib.reload = lambda mod: mod
    try:
        runpy.run_module("main", run_name="__main__")
    finally:
        importlib.reload = original_reload
    assert called == [True]
