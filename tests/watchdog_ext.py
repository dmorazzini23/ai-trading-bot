import faulthandler
import os
import socket
import time

import pytest


@pytest.fixture(scope="session", autouse=True)
def _watchdog():
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.fixture(autouse=True)
def _short_sleep(monkeypatch):
    orig = time.sleep

    def fast_sleep(s):
        return orig(0 if s <= 0 else min(s, 0.02))

    monkeypatch.setattr(time, "sleep", fast_sleep)
    yield


@pytest.fixture(scope="session", autouse=True)
def _requests_default_timeout(monkeypatch):
    try:
        import requests
    except Exception:
        return
    default_timeout = float(os.getenv("HTTP_TIMEOUT_S", "10") or 10)

    # Robust: patch the commonly exposed class attribute
    # (some environments don't have `requests.sessions` attr)
    orig_request = requests.Session.request

    def _request_with_default_timeout(self, method, url, **kwargs):
        if "timeout" not in kwargs or kwargs["timeout"] is None:
            kwargs["timeout"] = default_timeout
        return orig_request(self, method, url, **kwargs)

    monkeypatch.setattr(requests.Session, "request", _request_with_default_timeout)
    yield


@pytest.fixture(scope="session", autouse=True)
def _block_external_network(monkeypatch):
    if os.getenv("ALLOW_EXTERNAL_NETWORK", "0") == "1":
        return
    orig_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        try:
            ip = socket.gethostbyname(host)
        except Exception:
            ip = host
        if ip.startswith("127.") or ip in ("::1", "localhost"):
            return orig_connect(self, address)
        raise RuntimeError(
            f"External network blocked in tests (host={host}). "
            "Set ALLOW_EXTERNAL_NETWORK=1 to override."
        )

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    yield


@pytest.fixture(scope="session", autouse=True)
def _test_env():
    os.environ.setdefault("TESTING", "1")
    os.environ.setdefault("CPU_ONLY", "1")
    os.environ.setdefault("AI_TRADER_HEALTH_TICK_SECONDS", "2")
    os.environ.setdefault("HTTP_TIMEOUT_S", "10")
    os.environ.setdefault("FLASK_PORT", "0")
    yield
