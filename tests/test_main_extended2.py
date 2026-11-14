import time
from typing import Any
import threading

import errno
import socket

import pytest

from ai_trading import app, main


@pytest.fixture(autouse=True)
def offline_equity(monkeypatch):
    monkeypatch.setattr(main, "_get_equity_from_alpaca", lambda cfg: 100000.)


def test_run_flask_app(monkeypatch):
    """Flask app runs on provided port and forwards kwargs."""
    called: dict[str, Any] = {}

    class App:
        def run(self, host, port, debug=False, **kwargs):
            called["args"] = (host, port)
            called["debug"] = debug
            called["kwargs"] = kwargs

    monkeypatch.setattr(app, "create_app", lambda: App())
    main.run_flask_app(1234, debug=True, extra=1)
    assert called["args"] == ("0.0.0.0", 1234)
    assert called["debug"] is True
    assert called["kwargs"] == {"extra": 1, "use_reloader": False}


def test_run_flask_app_suppresses_flask_banner(monkeypatch):
    called = {"suppressed": False}

    class App:
        def run(self, host, port, debug=False, **kwargs):
            raise SystemExit

    monkeypatch.setattr(app, "create_app", lambda: App())
    monkeypatch.setattr(app, "suppress_flask_startup_noise", lambda: called.__setitem__("suppressed", True))
    with pytest.raises(SystemExit):
        main.run_flask_app(1234)
    assert called["suppressed"] is True


def test_run_flask_app_signals_ready_only_after_probe(monkeypatch):
    ready_signal = threading.Event()
    state = {"serving": False}

    class App:
        def run(self, host, port, debug=False, **kwargs):
            assert host == "0.0.0.0"
            assert not ready_signal.is_set()
            state["serving"] = True
            deadline = time.monotonic() + 1.0
            while not ready_signal.is_set() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready_signal.is_set()
            raise SystemExit

    monkeypatch.setattr(app, "create_app", lambda: App())
    monkeypatch.setattr(
        main,
        "_probe_local_api_socket",
        lambda _port, **_kwargs: bool(state["serving"]),
    )
    monkeypatch.setattr(main, "get_pid_on_port", lambda _port: None)

    with pytest.raises(SystemExit):
        main.run_flask_app(1234, ready_signal=ready_signal)


def test_run_flask_app_port_in_use(monkeypatch):
    """OSError EADDRINUSE fails instead of moving off the configured port."""
    called = []

    class App:
        def run(self, host, port, debug=False, **kwargs):
            called.append(port)
            raise OSError(errno.EADDRINUSE, "address in use")

    monkeypatch.setattr(app, "create_app", lambda: App())
    monkeypatch.setattr(main, "get_pid_on_port", lambda p: None)
    with pytest.raises(main.PortInUseError) as excinfo:
        main.run_flask_app(1234)
    assert excinfo.value.port == 1234
    assert called == [1234]


def test_run_flask_app_skips_ipv6_port(monkeypatch):
    """IPv6-bound configured ports fail instead of walking upward."""
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
    with pytest.raises(main.PortInUseError) as excinfo:
        main.run_flask_app(port)
    s.close()
    assert excinfo.value.port == port
    assert called == []






def test_main_runs_once(monkeypatch):
    """main executes a single cycle when configured."""
    monkeypatch.setenv("SCHEDULER_ITERATIONS", "1")
    called: dict[str, Any] = {}

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
