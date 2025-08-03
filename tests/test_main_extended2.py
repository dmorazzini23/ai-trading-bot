import logging
import sys
import types

import pytest

flask_mod = types.ModuleType("flask")
class Flask:
    def __init__(self, *a, **k):
        pass
    def route(self, *a, **k):
        def deco(f):
            return f
        return deco
    def run(self, *a, **k):
        pass
flask_mod.Flask = Flask
flask_mod.jsonify = lambda *a, **k: {}
sys.modules["flask"] = flask_mod
import ai_trading.main as main
import ai_trading.app as app


def test_run_flask_app(monkeypatch):
    """Flask app runs on provided port."""
    called = {}

    class App:
        def run(self, host, port):
            called["args"] = (host, port)

    monkeypatch.setattr(app, "create_app", lambda: App())
    main.run_flask_app(1234)
    assert called["args"] == ("0.0.0.0", 1234)


def test_run_flask_app_port_in_use(monkeypatch):
    """Port conflict triggers fallback port."""
    called = []

    class App:
        def run(self, host, port):
            called.append(port)

    monkeypatch.setattr(app, "create_app", lambda: App())
    monkeypatch.setattr(main.utils, "get_pid_on_port", lambda p: 111)
    monkeypatch.setattr(main.utils, "get_free_port", lambda *a, **k: 5678)
    main.run_flask_app(1234)
    assert called == [5678]


def test_run_bot_calls_cycle(monkeypatch):
    """run_bot executes a trading cycle in-process."""
    called = {}

    monkeypatch.setattr(
        main, "run_cycle", lambda: called.setdefault("ran", True)
    )
    assert main.run_bot() == 0
    assert called["ran"]


def test_validate_environment_missing(monkeypatch):
    """validate_environment errors when secret missing."""
    monkeypatch.setattr(main.config, 'WEBHOOK_SECRET', '', raising=False)
    with pytest.raises(RuntimeError):
        main.validate_environment()


def test_main_runs_once(monkeypatch):
    """main executes a single cycle when configured."""
    monkeypatch.setenv("SCHEDULER_ITERATIONS", "1")
    called = {}

    # AI-AGENT-REF: Fix lambda signature to accept ready_signal parameter and set it
    def mock_start_api(ready_signal=None):
        called.setdefault("api", True)
        if ready_signal:
            ready_signal.set()  # Important: signal that API is ready
    monkeypatch.setattr(main, "start_api", mock_start_api)
    def _cycle():
        called["cycle"] = called.get("cycle", 0) + 1
    monkeypatch.setattr(main, "run_cycle", _cycle)
    monkeypatch.setattr(main.time, "sleep", lambda s: None)
    main.main()
    assert called.get("api")
    assert called.get("cycle") == 1
