import errno
import socket

import pytest

from ai_trading import main


class DummySettings:
    def __init__(
        self,
        api_port: int,
        api_port_wait_seconds: float = 0.0,
        healthcheck_port: int | None = None,
    ):
        self.api_port = api_port
        self.api_port_wait_seconds = api_port_wait_seconds
        self.healthcheck_port = api_port if healthcheck_port is None else healthcheck_port


def test_start_api_raises_when_port_busy(monkeypatch):
    """Configured API port conflicts should raise an error instead of retrying."""
    probe = socket.socket()
    probe.bind(("0.0.0.0", 0))
    start_port = probe.getsockname()[1]
    probe.close()

    blocker = socket.socket()
    blocker.bind(("0.0.0.0", start_port))
    blocker.listen(1)

    monkeypatch.setattr(main, "get_settings", lambda: DummySettings(start_port, 0.0))
    monkeypatch.setattr(main, "ensure_dotenv_loaded", lambda: None)

    def _fail(*_args, **_kwargs):
        raise AssertionError("run should not be invoked")

    monkeypatch.setattr(main, "run_flask_app", _fail)

    try:
        with pytest.raises(main.PortInUseError) as excinfo:
            main.start_api()
    finally:
        blocker.close()

    assert excinfo.value.port == start_port


def test_start_api_waits_for_transient_port_conflicts(monkeypatch):
    """start_api retries briefly when port is busy without an owning PID."""

    probe = socket.socket()
    probe.bind(("0.0.0.0", 0))
    test_port = probe.getsockname()[1]
    probe.close()

    original_socket = socket.socket

    class RetrySocket(socket.socket):
        failures_remaining = 2

        def bind(self, address):  # type: ignore[override]
            if RetrySocket.failures_remaining > 0:
                RetrySocket.failures_remaining -= 1
                err = OSError(errno.EADDRINUSE, "Address already in use")
                err.errno = errno.EADDRINUSE
                raise err
            return super().bind(address)

    monkeypatch.setattr(socket, "socket", RetrySocket)
    monkeypatch.setattr(main.socket, "socket", RetrySocket)
    monkeypatch.setattr(main, "get_settings", lambda: DummySettings(test_port, 2.0))
    monkeypatch.setattr(main, "ensure_dotenv_loaded", lambda: None)
    monkeypatch.setattr(main, "get_pid_on_port", lambda _port: None)

    fake_time = [0.0]

    def fake_monotonic():
        return fake_time[0]

    def fake_sleep(seconds: float):
        fake_time[0] += seconds

    monkeypatch.setattr(main.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(main.time, "sleep", fake_sleep)

    called = {}

    def capture_run(port: int, *_args, **_kwargs):
        called["port"] = port

    monkeypatch.setattr(main, "run_flask_app", capture_run)

    try:
        main.start_api()
    finally:
        monkeypatch.setattr(socket, "socket", original_socket)
        monkeypatch.setattr(main.socket, "socket", original_socket)

    assert called["port"] == test_port
    assert RetrySocket.failures_remaining == 0


def test_start_api_aborts_when_existing_api_healthy(monkeypatch):
    """Healthy existing API instances should halt the new startup."""

    probe = socket.socket()
    probe.bind(("0.0.0.0", 0))
    test_port = probe.getsockname()[1]
    probe.close()

    blocker = socket.socket()
    blocker.bind(("0.0.0.0", test_port))
    blocker.listen(1)

    monkeypatch.setattr(main, "get_settings", lambda: DummySettings(test_port))
    monkeypatch.setattr(main, "ensure_dotenv_loaded", lambda: None)
    monkeypatch.setattr(main, "get_pid_on_port", lambda _port: None)
    monkeypatch.setattr(main, "_probe_local_api_health", lambda _port: True)

    try:
        with pytest.raises(main.ExistingApiDetected) as excinfo:
            main.start_api()
    finally:
        blocker.close()

    assert excinfo.value.port == test_port


def test_start_api_fails_before_logging_aux_health_started(monkeypatch, caplog):
    api_probe = socket.socket()
    api_probe.bind(("0.0.0.0", 0))
    api_port = api_probe.getsockname()[1]
    api_probe.close()

    health_blocker = socket.socket()
    health_blocker.bind(("0.0.0.0", 0))
    health_port = health_blocker.getsockname()[1]
    health_blocker.listen(1)

    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: DummySettings(api_port, 0.0, healthcheck_port=health_port),
    )
    monkeypatch.setattr(main, "ensure_dotenv_loaded", lambda: None)
    monkeypatch.setattr(main, "run_flask_app", lambda *_args, **_kwargs: None)

    try:
        with caplog.at_level("INFO"):
            with pytest.raises(OSError):
                main.start_api()
    finally:
        health_blocker.close()

    assert "HEALTHCHECK_PORT_CONFLICT" in caplog.text
    assert "HEALTH_SERVER_STARTED" not in caplog.text
