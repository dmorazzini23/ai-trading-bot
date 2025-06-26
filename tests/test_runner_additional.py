import importlib
import types

import pytest
import requests

import runner


def test_handle_signal_sets_shutdown():
    importlib.reload(runner)
    runner._shutdown = False
    runner._handle_signal(15, None)
    assert runner._shutdown


def test_run_forever_request_exception(monkeypatch):
    importlib.reload(runner)
    monkeypatch.setattr(runner, "main", lambda: (_ for _ in ()).throw(requests.exceptions.RequestException("boom")))
    with pytest.raises(requests.exceptions.RequestException):
        runner._run_forever()


def test_run_forever_unexpected(monkeypatch):
    importlib.reload(runner)
    monkeypatch.setattr(runner, "main", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        runner._run_forever()
