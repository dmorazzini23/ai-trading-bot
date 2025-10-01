from __future__ import annotations

import importlib
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


def ensure_urllib3_disable_warnings() -> None:
    """Re-import ``urllib3`` and guarantee a callable ``disable_warnings`` attribute."""

    try:
        urllib3 = importlib.import_module("urllib3")
    except Exception:  # pragma: no cover - urllib3 not installed
        return

    globals()["urllib3"] = urllib3

    disable_warnings = getattr(urllib3, "disable_warnings", None)
    if not callable(disable_warnings):

        def _noop_disable_warnings(*args, **kwargs):
            return None

        try:
            setattr(urllib3, "disable_warnings", _noop_disable_warnings)
        except Exception:  # pragma: no cover - attribute assignment failed
            return
        disable_warnings = getattr(urllib3, "disable_warnings", None)
        if not callable(disable_warnings):  # pragma: no cover - custom module rejected shim
            return

    warning_category = Warning
    exceptions = getattr(urllib3, "exceptions", None)
    if exceptions is not None:
        warning_category = getattr(exceptions, "SystemTimeWarning", warning_category)
        if warning_category is Warning:
            warning_category = getattr(exceptions, "HTTPWarning", warning_category)

    try:
        disable_warnings(warning_category)
    except Exception:  # pragma: no cover - downstream failure should not break imports
        pass


ensure_urllib3_disable_warnings()


_SessionBase = cast(
    "type[object]", requests.Session if getattr(requests, "Session", None) else object
)


class TimeoutSession(_SessionBase):
    """Requests ``Session`` that injects a default timeout."""

    def __init__(self, default_timeout: tuple[float, float] = (5.0, 10.0)) -> None:
        ensure_urllib3_disable_warnings()
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
                timeout = self._default_timeout
            else:
                timeout = clamp_request_timeout(kwargs["timeout"])
            kwargs.pop("timeout", None)
            return requests.get(url, timeout=timeout, **kwargs)
        return super().get(url, **kwargs)


# Public alias used throughout the codebase
HTTPSession = TimeoutSession


class HostLimitController:
    """Manage HTTPAdapter pool sizing based on environment overrides."""

    def __init__(self) -> None:
        self._last_limit: int | None = None

    @staticmethod
    def _parse_limit(value: str | None) -> int | None:
        if value is None or value.strip() == "":
            return None
        try:
            limit = int(value)
        except (TypeError, ValueError):
            return None
        return max(limit, 1)

    def current_limit(self) -> int | None:
        """Return the active pool limit honoring legacy environment names."""

        for env_var in ("HTTP_MAX_PER_HOST", "AI_TRADING_HTTP_HOST_LIMIT", "AI_TRADING_HOST_LIMIT"):
            raw = os.getenv(env_var)
            parsed = self._parse_limit(raw)
            if parsed is not None:
                return parsed
        return None

    def apply(self, session: TimeoutSession) -> None:
        limit = self.current_limit()
        if limit is None or HTTPAdapter is object:  # pragma: no cover - requests missing
            return
        existing_https = session.adapters.get("https://") if hasattr(session, "adapters") else None
        max_retries = getattr(existing_https, "max_retries", Retry(total=0))
        adapter = HTTPAdapter(
            max_retries=max_retries,
            pool_connections=limit,
            pool_maxsize=limit,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

    def reload_if_changed(self, session: TimeoutSession) -> None:
        limit = self.current_limit()
        if limit == self._last_limit:
            return
        self._last_limit = limit
        if session is not None:
            self.apply(session)


_GLOBAL_SESSION: TimeoutSession | None = None
_HOST_LIMIT_CONTROLLER = HostLimitController()


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

    ensure_urllib3_disable_warnings()

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
    _HOST_LIMIT_CONTROLLER.reload_if_changed(s)
    return s


def set_global_session(s: TimeoutSession) -> None:
    """Register global session singleton."""

    ensure_urllib3_disable_warnings()
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
    ensure_urllib3_disable_warnings()
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

    ensure_urllib3_disable_warnings()
    if _GLOBAL_SESSION is None:
        set_global_session(build_retrying_session())
    reload_host_limit_if_env_changed(_GLOBAL_SESSION)
    return _GLOBAL_SESSION


def get_http_session() -> HTTPSession:
    """Return process-wide HTTP session singleton."""

    ensure_urllib3_disable_warnings()
    return get_global_session()


def reload_host_limit_if_env_changed(session: TimeoutSession | None = None) -> None:
    """Reapply host connection limits when the environment override changes."""

    target = session or _GLOBAL_SESSION
    if target is None:
        return
    try:
        from ai_trading.http import pooling as _pooling
    except Exception:  # pragma: no cover - pooling optional during stubbed tests
        pass
    else:
        try:
            _pooling.reload_host_limit_if_env_changed()
        except Exception:
            pass
    _HOST_LIMIT_CONTROLLER.reload_if_changed(target)


__all__ = [
    "HTTPSession",
    "TimeoutSession",
    "build_retrying_session",
    "set_global_session",
    "get_global_session",
    "get_http_session",
    "mount_host_retry_profile",
    "ensure_urllib3_disable_warnings",
    "reload_host_limit_if_env_changed",
]
