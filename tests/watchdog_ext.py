import contextlib
import faulthandler
import importlib
import os
import socket
import sys
import time
import contextlib
import faulthandler
import os
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

import pytest


@pytest.fixture(scope="session", autouse=True)
def _watchdog():
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            faulthandler.cancel_dump_traceback_later()


@pytest.fixture(autouse=True)
def _short_sleep(monkeypatch):
    orig_sleep = time.sleep

    def fast_sleep(s):
        return orig_sleep(0 if s <= 0 else min(s, 0.02))

    monkeypatch.setattr(time, "sleep", fast_sleep)
    yield


def _resolve_requests_session():
    """Locate requests.Session in a robust way."""  # AI-AGENT-REF: fallback lookup
    try:
        import requests  # noqa: F401
    except (requests.RequestException, TimeoutError):
        return None, None
    Session = getattr(sys.modules["requests"], "Session", None)
    if Session is None:
        try:
            sess_mod = importlib.import_module("requests.sessions")
            Session = getattr(sess_mod, "Session", None)
        except (requests.RequestException, TimeoutError):  # pragma: no cover - defensive
            Session = None
    return Session, sys.modules.get("requests")


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

    orig_request = Session.request

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
    os.environ.setdefault("AI_TRADER_HEALTH_TICK_SECONDS", "2")
    os.environ.setdefault("AI_HTTP_TIMEOUT", "10")
    os.environ.setdefault("FLASK_PORT", "0")  # avoid port collisions
    yield
