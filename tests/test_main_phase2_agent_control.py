from __future__ import annotations

import http.client
import types

import ai_trading.main as main


def test_probe_local_api_health_accepts_canonical_service(monkeypatch) -> None:
    closed: list[str] = []

    class _Response:
        status = 200

        def read(self) -> bytes:
            return b'{"service": "ai-trading", "ok": true}'

        def close(self) -> None:
            closed.append("response")

    class _Connection:
        def __init__(self, host: str, port: int, timeout: float) -> None:
            assert (host, port, timeout) == ("127.0.0.1", 9001, 1.5)

        def request(self, method: str, path: str) -> None:
            assert (method, path) == ("GET", "/healthz")

        def getresponse(self) -> _Response:
            return _Response()

        def close(self) -> None:
            closed.append("connection")

    monkeypatch.setattr(http.client, "HTTPConnection", _Connection)

    assert main._probe_local_api_health(9001) is True
    assert closed == ["response", "connection"]


def test_probe_local_api_health_rejects_bad_json(monkeypatch) -> None:
    class _Response:
        status = 200

        def read(self) -> bytes:
            return b"not-json"

        def close(self) -> None:
            return None

    class _Connection:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def request(self, *args, **kwargs) -> None:
            return None

        def getresponse(self) -> _Response:
            return _Response()

        def close(self) -> None:
            return None

    monkeypatch.setattr(http.client, "HTTPConnection", _Connection)

    assert main._probe_local_api_health(9001) is False


def test_init_http_session_retries_and_applies_host_profile(monkeypatch) -> None:
    attempts: list[dict[str, float]] = []
    mounted: list[dict[str, object]] = []
    slept: list[float] = []
    session = object()

    def _build_retrying_session(**kwargs):
        attempts.append(dict(kwargs))
        if len(attempts) == 1:
            raise RuntimeError("temporary init failure")
        return session

    monkeypatch.setattr(main, "should_stop", lambda: False)
    monkeypatch.setattr(main, "build_retrying_session", _build_retrying_session)
    monkeypatch.setattr(main, "set_global_session", lambda value: mounted.append({"global": value}))
    monkeypatch.setattr(
        main,
        "mount_host_retry_profile",
        lambda sess, host, **kwargs: mounted.append(
            {"session": sess, "host": host, **kwargs}
        ),
    )
    monkeypatch.setattr(
        main,
        "_managed_env",
        lambda name, default=None: {
            "HTTP_RETRIES_paper-api_example_com": "4",
            "HTTP_BACKOFF_paper-api_example_com": "0.7",
        }.get(name, default),
    )
    monkeypatch.setattr(main, "_interruptible_sleep", lambda seconds: slept.append(seconds))

    cfg = types.SimpleNamespace(
        http_connect_timeout=2.0,
        http_read_timeout=4.0,
        http_pool_maxsize=7,
        http_total_retries=2,
        http_backoff_factor=0.1,
        alpaca_base_url="https://paper-api.example.com/v2",
    )

    assert main._init_http_session(cfg, retries=2, delay=0.25) is True
    assert len(attempts) == 2
    assert slept == [0.25]
    assert {"global": session} in mounted
    assert {
        "session": session,
        "host": "paper-api.example.com",
        "total_retries": 4,
        "backoff_factor": 0.7,
        "pool_maxsize": 7,
    } in mounted


def test_init_http_session_aborts_when_shutdown_requested(monkeypatch) -> None:
    monkeypatch.setattr(main, "should_stop", lambda: True)

    assert main._init_http_session(types.SimpleNamespace(), retries=3, delay=0.0) is False
