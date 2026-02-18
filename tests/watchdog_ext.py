import contextlib
import faulthandler
import importlib
import os
import socket
import sys
import time

import pytest
try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - optional dependency in CI
    requests = None


@pytest.fixture(scope="session", autouse=True)
def _watchdog():
    faulthandler.enable()
    watchdog_seconds_raw = os.getenv("PYTEST_WATCHDOG_SECONDS", "0")
    watchdog_repeat_raw = os.getenv("PYTEST_WATCHDOG_REPEAT", "0")
    with contextlib.suppress(ValueError):
        watchdog_seconds = int(float(watchdog_seconds_raw))
        watchdog_repeat = watchdog_repeat_raw.strip().lower() in {"1", "true", "yes", "on"}
        if watchdog_seconds > 0:
            faulthandler.dump_traceback_later(watchdog_seconds, repeat=watchdog_repeat)
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            faulthandler.cancel_dump_traceback_later()


@pytest.fixture(autouse=True)
def _short_sleep(monkeypatch):
    orig_sleep = time.sleep

    def fast_sleep(s):
        if s <= 0:
            return orig_sleep(0)
        if s >= 0.1:
            return orig_sleep(s)
        return orig_sleep(min(s, 0.02))

    monkeypatch.setattr(time, "sleep", fast_sleep)
    yield


def _resolve_requests_session():
    """Locate requests.Session in a robust way."""  # AI-AGENT-REF: fallback lookup
    if requests is None:
        return None, None
    Session = getattr(requests, "Session", None)
    if Session is None:
        try:
            sess_mod = importlib.import_module("requests.sessions")
            Session = getattr(sess_mod, "Session", None)
        except (TimeoutError, Exception):  # pragma: no cover - defensive
            Session = None
    return Session, requests


@pytest.fixture(scope="session", autouse=True)
def _requests_default_timeout():
    Session, _ = _resolve_requests_session()
    if Session is None:
        sys.stderr.write(
            "[watchdog_ext] Warning: could not locate requests.Session; "
            "default-timeout patch disabled. Check for local module shadowing.\n"
        )
        yield
        return

    default_timeout = float(os.getenv("AI_HTTP_TIMEOUT", "10") or 10)

    orig_request = getattr(Session, "request", None)
    if orig_request is None:
        yield
        return

    def request_with_default_timeout(self, method, url, **kwargs):
        if "timeout" not in kwargs or kwargs["timeout"] is None:
            kwargs["timeout"] = default_timeout
        return orig_request(self, method, url, **kwargs)

    Session.request = (
        request_with_default_timeout  # AI-AGENT-REF: manual patch to inject default timeout
    )
    try:
        yield
    finally:
        Session.request = orig_request  # AI-AGENT-REF: restore original request method


@pytest.fixture(scope="session", autouse=True)
def _block_external_network():
    if os.getenv("ALLOW_EXTERNAL_NETWORK", "0") == "1":
        yield
        return

    orig_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        try:
            ip = socket.gethostbyname(host)
        except (TimeoutError, Exception):
            ip = str(host)
        if ip.startswith("127.") or host in ("::1", "localhost"):
            return orig_connect(self, address)
        raise RuntimeError(
            f"External network blocked in tests (host={host}). "
            f"Set ALLOW_EXTERNAL_NETWORK=1 to override."
        )

    socket.socket.connect = guarded_connect  # AI-AGENT-REF: manual patch to block external network
    try:
        yield
    finally:
        socket.socket.connect = orig_connect  # AI-AGENT-REF: restore network connect


@pytest.fixture(scope="session", autouse=True)
def _test_env():
    os.environ.setdefault("TESTING", "1")
    os.environ.setdefault("CPU_ONLY", "1")
    os.environ.setdefault("AI_TRADING_HEALTH_TICK_SECONDS", "30")
    os.environ.setdefault("AI_HTTP_TIMEOUT", "10")
    os.environ.setdefault("FLASK_PORT", "0")  # avoid port collisions
    yield


# Reset HTTP timeout coupling across tests to avoid order sensitivity
@pytest.fixture(autouse=True)
def _reset_http_timeout_env(monkeypatch):
    # Ensure a clean timeout env at the start of each test; avoid reloading timing here
    monkeypatch.delenv("HTTP_TIMEOUT", raising=False)
    yield
