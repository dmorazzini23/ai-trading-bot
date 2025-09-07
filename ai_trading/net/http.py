from __future__ import annotations

import os
from typing import cast

try:
    import requests
except Exception:  # pragma: no cover - fallback when requests missing
    class _RequestsFallback:
        Session = None
        get = None

    requests = _RequestsFallback()  # type: ignore[assignment]
try:
    from requests.adapters import HTTPAdapter
except Exception:  # pragma: no cover - requests missing
    HTTPAdapter = cast("type[object]", object)
from urllib3.util.retry import Retry
from ai_trading.utils import clamp_request_timeout
from urllib.parse import urlparse

try:  # handle TLS validation clock skew warnings gracefully
    import urllib3
    urllib3.disable_warnings(
        getattr(urllib3.exceptions, "SystemTimeWarning", urllib3.exceptions.HTTPWarning)
    )
except Exception:  # pragma: no cover - urllib3 missing or misbehaving
    pass


_SessionBase = cast(
    "type[object]", requests.Session if getattr(requests, "Session", None) else object
)


class TimeoutSession(_SessionBase):
    """Requests ``Session`` that injects a default timeout."""

    def __init__(self, default_timeout: tuple[float, float] = (5.0, 10.0)) -> None:
        super().__init__()
        self._default_timeout = cast(tuple[float, float], clamp_request_timeout(default_timeout))

    def request(self, method, url, **kwargs):  # type: ignore[override]
        if "timeout" not in kwargs or kwargs["timeout"] is None:
            kwargs["timeout"] = self._default_timeout
        else:
            kwargs["timeout"] = clamp_request_timeout(kwargs["timeout"])
        return super().request(method, url, **kwargs)

    def get(self, url, **kwargs):  # type: ignore[override]
        """Issue a GET request with a test-friendly fast path.

        In test environments (TESTING=1), route through the top-level
        ``requests.get`` function so test suites that monkeypatch
        ``requests.get`` can intercept calls deterministically. Otherwise,
        defer to the parent ``Session.get`` implementation which preserves
        connection pooling and retry adapters.
        """
        if os.getenv("TESTING", "0") == "1":
            if "timeout" not in kwargs or kwargs["timeout"] is None:
                kwargs["timeout"] = self._default_timeout
            else:
                kwargs["timeout"] = clamp_request_timeout(kwargs["timeout"])
            return requests.get(url, **kwargs)
        return super().get(url, **kwargs)


# Public alias used throughout the codebase
HTTPSession = TimeoutSession

_GLOBAL_SESSION: TimeoutSession | None = None


def build_retrying_session(
    *,
    pool_maxsize: int = 32,
    total_retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
    connect_timeout: float = 5.0,
    read_timeout: float = 10.0,
) -> TimeoutSession:
    """Create a session with urllib3 ``Retry`` and default timeout."""

    connect_timeout_f = cast(float, clamp_request_timeout(connect_timeout))
    read_timeout_f = cast(float, clamp_request_timeout(read_timeout))
    s = TimeoutSession(default_timeout=(connect_timeout_f, read_timeout_f))
    if HTTPAdapter is object:  # pragma: no cover - requests missing
        return s
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset({"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=pool_maxsize,
        pool_maxsize=pool_maxsize,
    )
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def set_global_session(s: TimeoutSession) -> None:
    """Register global session singleton."""

    global _GLOBAL_SESSION
    _GLOBAL_SESSION = s


def mount_host_retry_profile(
    s: TimeoutSession,
    host_or_url: str,
    *,
    total_retries: int,
    backoff_factor: float,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
    pool_maxsize: int | None = None,
) -> None:
    """Mount a host-specific retry profile on an existing session.

    host_or_url may be a bare host (e.g. "paper-api.alpaca.markets") or a URL.
    """
    parsed = urlparse(host_or_url if "://" in host_or_url else f"https://{host_or_url}")
    host = parsed.netloc or parsed.path
    if not host:
        return
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset({"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=pool_maxsize or 32,
        pool_maxsize=pool_maxsize or 32,
    )
    s.mount(f"https://{host}/", adapter)
    s.mount(f"http://{host}/", adapter)


def get_global_session() -> TimeoutSession:
    """Return the global session, building a default if missing."""

    if _GLOBAL_SESSION is None:
        set_global_session(build_retrying_session())
    return _GLOBAL_SESSION


def get_http_session() -> HTTPSession:
    """Return process-wide HTTP session singleton."""

    return get_global_session()


__all__ = [
    "HTTPSession",
    "TimeoutSession",
    "build_retrying_session",
    "set_global_session",
    "get_global_session",
    "get_http_session",
    "mount_host_retry_profile",
]
