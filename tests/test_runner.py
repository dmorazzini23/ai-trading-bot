import runpy
import sys
import types

import pytest
import requests

from tests.conftest import load_runner


def test_handle_signal_sets_shutdown(monkeypatch):
    mod = load_runner(monkeypatch)
    mod._shutdown = False
    mod._handle_signal(15, None)
    assert mod._shutdown


def test_run_forever_exit(monkeypatch):
    mod = load_runner(monkeypatch)
    monkeypatch.setattr(mod, "main", lambda: (_ for _ in ()).throw(SystemExit(0)))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    mod._shutdown = False
    mod._run_forever()


def test_run_forever_exception(monkeypatch):
    mod = load_runner(monkeypatch)
    monkeypatch.setattr(mod, "main", lambda: (_ for _ in ()).throw(ValueError("bad")))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    mod._shutdown = False
    with pytest.raises(ValueError):
        mod._run_forever()


def test_run_forever_request_exception(monkeypatch):
    mod = load_runner(monkeypatch)
    monkeypatch.setattr(mod, "main", lambda: (_ for _ in ()).throw(requests.exceptions.RequestException("boom")))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    mod._shutdown = False
    with pytest.raises(requests.exceptions.RequestException):
        mod._run_forever()


def test_run_forever_system_exit_nonzero(monkeypatch):
    mod = load_runner(monkeypatch)
    seq = [SystemExit(1), SystemExit(0)]

    def side():
        exc = seq.pop(0)
        raise exc

    monkeypatch.setattr(mod, "main", side)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    mod._shutdown = False
    mod._run_forever()
    assert not seq


def test_runner_as_main(monkeypatch):
    mod = load_runner(monkeypatch)
    monkeypatch.setattr(mod, "main", lambda: (_ for _ in ()).throw(SystemExit(0)))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    mod._shutdown = False
    runpy.run_module("runner", run_name="__main__")


def test_runner_import_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "bot", None)
    bot_engine_mod = types.ModuleType("bot_engine")
    bot_engine_mod.main = lambda: None
    monkeypatch.setitem(sys.modules, "bot_engine", bot_engine_mod)
    import importlib
    r = importlib.reload(importlib.import_module("runner"))
    assert r.main is bot_engine_mod.main
