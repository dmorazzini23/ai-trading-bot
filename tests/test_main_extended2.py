import os
import sys
import types
from typing import Any

import errno
import socket

import pytest

os.environ.setdefault("PYTEST_RUNNING", "1")


if "numpy" not in sys.modules:
    class _RandomStub:
        def seed(self, *_args, **_kwargs):
            return None

    class _NumpyStub(types.ModuleType):
        def __init__(self) -> None:
            super().__init__("numpy")
            self.random = _RandomStub()
            self.nan = float("nan")
            self.NaN = self.nan
            self.ndarray = object

        def __getattr__(self, _name):  # type: ignore[override]
            def _stub(*_args, **_kwargs):
                return 0

            return _stub

    sys.modules["numpy"] = _NumpyStub()

if "portalocker" not in sys.modules:
    portalocker_stub = types.ModuleType("portalocker")
    portalocker_stub.LOCK_EX = 1
    portalocker_stub.lock = lambda *_args, **_kwargs: None
    portalocker_stub.unlock = lambda *_args, **_kwargs: None
    sys.modules["portalocker"] = portalocker_stub

if "bs4" not in sys.modules:
    bs4_stub = types.ModuleType("bs4")

    class _Soup:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def find_all(self, *_args, **_kwargs):
            return []

        def find_parent(self, *_args, **_kwargs):
            return None

        def get_text(self, *_args, **_kwargs):
            return ""

    bs4_stub.BeautifulSoup = lambda *_args, **_kwargs: _Soup()
    sys.modules["bs4"] = bs4_stub

flask_mod: Any = types.ModuleType("flask")


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
from ai_trading import app, main


def test_run_flask_app(monkeypatch):
    """Flask app runs on provided port and forwards kwargs."""
    called = {}

    class App:
        def run(self, host, port, debug=False, **kwargs):
            called["args"] = (host, port)
            called["debug"] = debug
            called["kwargs"] = kwargs

    monkeypatch.setattr(app, "create_app", lambda: App())
    main.run_flask_app(1234, debug=True, extra=1)
    assert called["args"] == ("0.0.0.0", 1234)
    assert called["debug"] is True
    assert called["kwargs"] == {"extra": 1}


def test_run_flask_app_port_in_use(monkeypatch):
    """OSError EADDRINUSE triggers retry on next port."""
    called = []

    class App:
        def run(self, host, port, debug=False, **kwargs):
            called.append(port)
            if len(called) == 1:
                raise OSError(errno.EADDRINUSE, "address in use")
            raise SystemExit

    monkeypatch.setattr(app, "create_app", lambda: App())
    monkeypatch.setattr(main, "get_pid_on_port", lambda p: None)
    with pytest.raises(SystemExit):
        main.run_flask_app(1234)
    assert called == [1234, 1235]


def test_run_flask_app_skips_ipv6_port(monkeypatch):
    """IPv6-bound port is skipped in favor of a free one."""
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    s.bind(("::", 0))
    s.listen(1)
    port = s.getsockname()[1]
    called = []

    class App:
        def run(self, host, port, debug=False, **kwargs):
            called.append(port)
            raise SystemExit

    monkeypatch.setattr(app, "create_app", lambda: App())
    with pytest.raises(SystemExit):
        main.run_flask_app(port)
    s.close()
    assert called == [port + 1]


def test_run_bot_calls_cycle(monkeypatch):
    """run_bot executes a trading cycle in-process."""
    called = {}
    trade_log_calls = {"count": 0}

    def _fake_ensure():
        trade_log_calls["count"] += 1
        main._TRADE_LOG_INITIALIZED = True

    monkeypatch.setattr(main, "run_cycle", lambda: called.setdefault("ran", True))
    monkeypatch.setattr(main, "ensure_trade_log_path", _fake_ensure)
    monkeypatch.setenv("IMPORT_PREFLIGHT_DISABLED", "1")
    main._TRADE_LOG_INITIALIZED = False

    assert main.run_bot() == 0
    assert called["ran"]
    assert trade_log_calls["count"] == 1


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


def test_main_allows_blank_env_under_test(monkeypatch):
    """Blank secrets under pytest should not abort startup."""

    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv("SCHEDULER_ITERATIONS", "1")
    for key in (
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "ALPACA_DATA_FEED",
        "WEBHOOK_SECRET",
        "CAPITAL_CAP",
        "DOLLAR_RISK_LIMIT",
        "ALPACA_API_URL",
        "ALPACA_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    def _start_api(ready_signal=None):
        if ready_signal:
            ready_signal.set()

    call_order: list[str] = []

    def _cycle():
        call_order.append("cycle")

    monkeypatch.setattr(main, "start_api", _start_api)
    monkeypatch.setattr(main, "run_cycle", _cycle)
    monkeypatch.setattr(main, "_init_http_session", lambda _cfg: True)
    monkeypatch.setattr(main.time, "sleep", lambda _s: None)

    main.main([])

    assert call_order, "run_cycle should be invoked at least once"
    assert len(call_order[1:]) == 1
