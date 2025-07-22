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


def test_run_flask_app(monkeypatch):
    """Flask app runs on provided port."""
    called = {}

    class App:
        def run(self, host, port):
            called["args"] = (host, port)

    monkeypatch.setattr(main, "create_flask_app", lambda: App())
    main.run_flask_app(1234)
    assert called["args"] == ("0.0.0.0", 1234)


def test_run_flask_app_port_in_use(monkeypatch):
    """Port conflict triggers fallback port."""
    called = []

    class App:
        def run(self, host, port):
            called.append(port)

    monkeypatch.setattr(main, "create_flask_app", lambda: App())
    monkeypatch.setattr(main.utils, "get_pid_on_port", lambda p: 111)
    monkeypatch.setattr(main.utils, "get_free_port", lambda *a, **k: 5678)
    main.run_flask_app(1234)
    assert called == [5678]


def test_run_bot_missing_exec(monkeypatch):
    """run_bot raises when python executable missing."""
    monkeypatch.setattr(main.os.path, "isfile", lambda p: False)
    with pytest.raises(RuntimeError):
        main.run_bot([])


def test_run_bot_success(monkeypatch):
    """Subprocess is invoked when executable exists."""
    monkeypatch.setattr(main.os.path, "isfile", lambda p: True)
    called = {}
    def fake_call(cmd):
        called["cmd"] = cmd
        return 7
    monkeypatch.setattr(main.subprocess, "call", fake_call)
    ret = main.run_bot(["--test"], service=False)
    assert ret == 7
    assert called["cmd"][:3] == [sys.executable, "-m", "ai_trading.main"]


def test_validate_environment_missing(monkeypatch):
    """validate_environment errors when secret missing."""
    monkeypatch.setattr(main.config, 'WEBHOOK_SECRET', '', raising=False)
    with pytest.raises(RuntimeError):
        main.validate_environment()


def test_main_bot_only(monkeypatch):
    """main runs bot and exits with its return code."""
    monkeypatch.setattr(sys, 'argv', ['ai_trading', '--bot-only'])
    monkeypatch.setattr(main, 'run_bot', lambda argv=None, service=False: 5)
    monkeypatch.setattr(main, 'run_flask_app', lambda port: None)
    monkeypatch.setattr(main, 'setup_logging', lambda *a, **k: logging.getLogger('t'))
    monkeypatch.setattr(main, 'load_dotenv', lambda *a, **k: None)
    monkeypatch.setattr(main, 'validate_environment', lambda: None)
    exits = []
    monkeypatch.setattr(sys, 'exit', lambda code=0: exits.append(code))
    main.main()
    assert exits == [5]
