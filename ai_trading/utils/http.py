from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

try:  # requests is optional
    import requests  # type: ignore
    from requests.adapters import HTTPAdapter  # type: ignore
    from requests.exceptions import RequestException as RequestsRequestException  # type: ignore

    REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover - requests missing
    requests = None  # type: ignore
    HTTPAdapter = None  # type: ignore

    class RequestsRequestException(Exception):  # type: ignore
        pass

    REQUESTS_AVAILABLE = False

    class _StubSession:
        def request(self, *args, **kwargs):
            raise RuntimeError("requests library is required for HTTP operations")

        def get(self, *args, **kwargs):
            return self.request(*args, **kwargs)

        def post(self, *args, **kwargs):
            return self.request(*args, **kwargs)

        def put(self, *args, **kwargs):
            return self.request(*args, **kwargs)

        def delete(self, *args, **kwargs):
            return self.request(*args, **kwargs)

    class _StubResponse:
        pass

    class _RequestsStub:
        Session = _StubSession
        Response = _StubResponse
        exceptions = type("exc", (), {"RequestException": RequestsRequestException})

    requests = _RequestsStub()  # type: ignore

try:  # urllib3 is only needed when requests is available
    from urllib3.util.retry import Retry  # type: ignore
except ImportError:  # pragma: no cover - fallback when urllib3 missing

    class Retry:  # type: ignore
        def __init__(self, *a, **k):
            pass


from contextlib import AbstractAsyncContextManager, contextmanager
from typing import Iterator
from ai_trading.exc import TRANSIENT_HTTP_EXC, JSONDecodeError, RequestException


def _strip_inline_comment(value: str) -> str:
    """Return ``value`` without trailing comments introduced with ``#``."""

    for idx, ch in enumerate(value):
        if ch == "#" and (idx == 0 or value[idx - 1].isspace()):
            return value[:idx].rstrip()
    return value.rstrip()


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(_strip_inline_comment(raw))
    except Exception:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(_strip_inline_comment(raw))
    except Exception:
        return default
from ai_trading.logging import get_logger
from ai_trading.utils.retry import retry_call
from .timing import clamp_timeout, sleep

_log = get_logger(__name__)
_session = None
_session_lock = threading.Lock()
_pool_stats = {
    "workers": _int_env("HTTP_POOL_WORKERS", _int_env("HTTP_MAX_WORKERS", 8)),
    "per_host": _int_env("HTTP_MAX_PER_HOST", 6),
    "pool_maxsize": _int_env("HTTP_POOL_MAXSIZE", 32),
    "requests": 0,
    "responses": 0,
    "errors": 0,
}


class _HostLimiterState:
    __slots__ = ("limit", "semaphore", "inflight", "peak")

    def __init__(self, limit: int) -> None:
        normalized = max(1, int(limit))
        self.limit = normalized
        self.semaphore = threading.BoundedSemaphore(normalized)
        self.inflight = 0
        self.peak = 0

    def set_limit(self, new_limit: int) -> None:
        normalized = max(1, int(new_limit))
        if normalized == self.limit:
            return
        adjusted = max(normalized, self.inflight if self.inflight > 0 else 1)
        semaphore = threading.BoundedSemaphore(adjusted)
        permits = min(self.inflight, adjusted)
        for _ in range(permits):
            semaphore.acquire(blocking=False)
        self.limit = adjusted
        self.semaphore = semaphore


_HOST_LIMITERS: dict[str, _HostLimiterState] = {}
_HOST_LIMIT_LOCK = threading.RLock()
_DEFAULT_HOST_LIMIT = max(1, _pool_stats["per_host"])


def _coerce_host_limit(value: object | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return max(1, int(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _resolve_host_limit() -> int:
    candidate: object | None = None
    try:
        from ai_trading.config.management import get_env as _get_env  # type: ignore
    except Exception:  # pragma: no cover - during early bootstrap
        _get_env = None
    if _get_env is not None:
        try:
            candidate = _get_env("HTTP_MAX_CONNS_PER_HOST", None)
        except Exception:
            candidate = None
    if candidate in (None, ""):
        candidate = os.getenv("HTTP_MAX_CONNS_PER_HOST")
    limit = _coerce_host_limit(candidate)
    return limit if limit is not None else _DEFAULT_HOST_LIMIT


def _host_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.netloc or ""
    except Exception:
        host = ""
    host = host.strip().lower()
    return host or "default"


def _get_host_state(host: str, desired_limit: int) -> _HostLimiterState:
    with _HOST_LIMIT_LOCK:
        state = _HOST_LIMITERS.get(host)
        if state is None:
            state = _HostLimiterState(desired_limit)
            _HOST_LIMITERS[host] = state
        else:
            state.set_limit(desired_limit)
        return state


@contextmanager
def host_slot(host: str | None) -> Iterator[None]:
    """Limit concurrent requests to *host* using a bounded semaphore."""

    host_key = (host or "default").strip().lower() or "default"
    desired_limit = _resolve_host_limit()
    state = _get_host_state(host_key, desired_limit)
    state.semaphore.acquire()
    try:
        with _HOST_LIMIT_LOCK:
            state.inflight += 1
            if state.inflight > state.peak:
                state.peak = state.inflight
        yield
    finally:
        with _HOST_LIMIT_LOCK:
            state.inflight = max(0, state.inflight - 1)
        state.semaphore.release()


def reset_host_limit_state() -> None:
    """Reset host limiter bookkeeping (intended for tests)."""

    with _HOST_LIMIT_LOCK:
        _HOST_LIMITERS.clear()


def host_limit_snapshot(host: str | None = None) -> dict[str, int]:
    """Return ``{"limit": int, "inflight": int, "peak": int}`` for *host*."""

    host_key = (host or "default").strip().lower() or "default"
    with _HOST_LIMIT_LOCK:
        state = _HOST_LIMITERS.get(host_key)
        if state is None:
            limit = _resolve_host_limit()
            return {"limit": limit, "inflight": 0, "peak": 0}
        return {"limit": state.limit, "inflight": state.inflight, "peak": state.peak}


def reload_host_limit() -> int:
    """Refresh all host limiters using the latest environment configuration."""

    new_limit = _resolve_host_limit()
    with _HOST_LIMIT_LOCK:
        for state in _HOST_LIMITERS.values():
            state.set_limit(new_limit)
    return new_limit


def _get_session_timeout() -> float | int | None:
    try:
        from ai_trading.http.timeouts import get_session_timeout as _get

        return _get()
    except ImportError:
        return clamp_timeout(None)


def clamp_request_timeout(
    timeout: float | int | tuple[float | int, float | int] | None,
) -> float | tuple[float, float] | None:
    """Normalize ``requests``-style timeout values using :func:`clamp_timeout`.

    Accepts either a single timeout (applied to both connect/read) or a
    ``(connect, read)`` tuple. ``None`` is propagated so caller defaults can
    take effect.
    """

    if timeout is None:
        return None
    if isinstance(timeout, tuple):
        return tuple(clamp_timeout(t) for t in timeout)
    return clamp_timeout(timeout)


if REQUESTS_AVAILABLE:

    class HTTPSession(requests.Session):
        """Session with sane connection pooling and timeout defaults.

        Parameters
        ----------
        timeout:
            Default timeout (in seconds) applied to requests that do not
            provide a ``timeout``. The value is normalized via
            :func:`ai_trading.utils.timing.clamp_timeout`.
        """

        def __init__(self, timeout: float | int | None = None) -> None:
            super().__init__()
            if timeout is None:
                timeout = _get_session_timeout()
            self._timeout = clamp_request_timeout(timeout)
            _pool_stats["per_host"] = _int_env("HTTP_MAX_PER_HOST", _pool_stats["per_host"])
            _pool_stats["workers"] = _int_env(
                "HTTP_POOL_WORKERS",
                _int_env("HTTP_MAX_WORKERS", _pool_stats["workers"]),
            )
            retries = Retry(
                total=3,
                backoff_factor=0.3,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(
                pool_connections=_pool_stats["per_host"],
                pool_maxsize=_pool_stats["pool_maxsize"],
                max_retries=retries,
            )
            self.mount("http://", adapter)
            self.mount("https://", adapter)

        def request(self, method: str, url: str, **kwargs) -> requests.Response:  # type: ignore[override]
            timeout = kwargs.get("timeout")
            if timeout is None:
                timeout = self._timeout
            kwargs["timeout"] = clamp_request_timeout(timeout)
            return super().request(method, url, **kwargs)

else:  # pragma: no cover - exercised in tests

    class HTTPSession(requests.Session):
        """Stub session used when :mod:`requests` is unavailable."""

        def __init__(self, *a, **k):
            super().__init__()


def _build_session() -> HTTPSession:
    return HTTPSession()


def _get_session() -> HTTPSession:
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = _build_session()
    return _session


def _with_timeout(kwargs: dict, default_timeout: float | int | None = None) -> dict:
    """Clamp provided timeout while allowing session defaults."""
    if "timeout" in kwargs:
        kwargs["timeout"] = clamp_request_timeout(kwargs["timeout"])
    else:
        # Ensure callers without explicit timeout still get a sane default.
        # Prefer the session default when available; otherwise fall back to
        # ai_trading.utils.timing defaults used across tests.
        if default_timeout is not None:
            kwargs["timeout"] = clamp_request_timeout(default_timeout)
        else:
            from .timing import clamp_timeout as _clamp

            kwargs["timeout"] = _clamp(None)
    return kwargs


def _retry_config() -> tuple[int, float, float, float]:
    """Load retry knobs from settings if available."""
    retries, backoff, max_backoff, jitter = (3, 0.1, 2.0, 0.1)
    try:
        from ai_trading.config import get_settings

        s = get_settings()
        retries = int(getattr(s, "RETRY_MAX_ATTEMPTS", retries))
        backoff = float(getattr(s, "RETRY_BASE_DELAY", backoff))
        max_backoff = float(getattr(s, "RETRY_MAX_DELAY", max_backoff))
        jitter = float(getattr(s, "RETRY_JITTER", jitter))
    except (AttributeError, TypeError, ValueError, ImportError):
        pass
    return (retries, backoff, max_backoff, jitter)


def request(method: str, url: str, **kwargs) -> requests.Response:
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("requests library is required for HTTP operations")
    sess = _get_session()
    sess_default = getattr(sess, "_timeout", None)
    kwargs = _with_timeout(kwargs, sess_default)
    retries, backoff, max_backoff, jitter = _retry_config()
    excs = (RequestException, RequestsRequestException, JSONDecodeError, TimeoutError, OSError, ValueError)
    attempt = {"n": 0}
    host_key = _host_from_url(url)

    def _do_request() -> requests.Response:
        try:
            with host_slot(host_key):
                return sess.request(method, url, **kwargs)
        except excs as e:
            attempt["n"] += 1
            log_fn = _log.warning if attempt["n"] == 1 else _log.debug
            log_fn("HTTP_RETRY", extra={"attempt": attempt["n"], "attempts": retries, "error": str(e)})
            raise

    _pool_stats["requests"] += 1
    try:
        resp = retry_call(
            _do_request, exceptions=excs, retries=retries, backoff=backoff, max_backoff=max_backoff, jitter=jitter
        )
        _pool_stats["responses"] += 1
        return resp
    except excs as e:
        _pool_stats["errors"] += 1
        _log.error("HTTP_GIVEUP", extra={"attempts": retries, "error": str(e)})
        raise


async def async_request(method: str, url: str, **kwargs) -> requests.Response:
    """Asynchronously execute :func:`request` while honouring host limits."""

    if not REQUESTS_AVAILABLE:
        raise RuntimeError("requests library is required for HTTP operations")
    try:
        from ai_trading.http.pooling import AsyncHostLimiter
    except ImportError:  # pragma: no cover - defensive fallback
        AsyncHostLimiter = None  # type: ignore[assignment]

    if AsyncHostLimiter is None:
        return await asyncio.to_thread(request, method, url, **kwargs)

    async with AsyncHostLimiter.from_url(url):
        return await asyncio.to_thread(request, method, url, **kwargs)


_ERROR_LOGGED: set[str] = set()


def request_json(
    method: str,
    url: str,
    *,
    timeout: tuple[float, float] | float | None = (3.0, 10.0),
    retries: int = 3,
    backoff: float = 0.5,
    status_forcelist: set[int] | None = None,
    headers: dict | None = None,
    params: dict | None = None,
) -> dict:
    """Perform HTTP request and return decoded JSON with bounded retries."""
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("requests library is required for HTTP operations")
    status_forcelist = status_forcelist or {429, 500, 502, 503, 504}
    timeout = clamp_request_timeout(timeout)
    sess = _get_session()

    host_key = _host_from_url(url)

    def _fetch() -> requests.Response:
        with host_slot(host_key):
            return sess.request(method, url, headers=headers, params=params, timeout=timeout)

    for attempt in range(1, retries + 1):
        try:
            resp = _fetch()
            if resp.status_code in status_forcelist and attempt < retries:
                raise RequestsRequestException(f"status {resp.status_code}")
            try:
                return resp.json()
            except ValueError:
                text = resp.text.strip()
                return {"text": text}
        except (requests.RequestException, TimeoutError) as exc:
            key = f"{method}:{url}:{getattr(exc, 'args', '')}"
            log_fn = _log.warning if key not in _ERROR_LOGGED else _log.debug
            log_fn("HTTP_RETRY", extra={"attempt": attempt, "attempts": retries, "error": str(exc)})
            _ERROR_LOGGED.add(key)
            if attempt >= retries:
                raise
            sleep(backoff * attempt)


def get(url: str, **kwargs) -> requests.Response:
    return request("GET", url, **kwargs)


async def async_get(url: str, **kwargs) -> requests.Response:
    return await async_request("GET", url, **kwargs)


def post(url: str, **kwargs) -> requests.Response:
    return request("POST", url, **kwargs)


async def async_post(url: str, **kwargs) -> requests.Response:
    return await async_request("POST", url, **kwargs)


def put(url: str, **kwargs) -> requests.Response:
    return request("PUT", url, **kwargs)


async def async_put(url: str, **kwargs) -> requests.Response:
    return await async_request("PUT", url, **kwargs)


def delete(url: str, **kwargs) -> requests.Response:
    return request("DELETE", url, **kwargs)


async def async_delete(url: str, **kwargs) -> requests.Response:
    return await async_request("DELETE", url, **kwargs)


def pool_stats() -> dict:
    return dict(_pool_stats)


def _fetch_one(url: str, timeout: float | None = None) -> tuple[str, int, bytes]:
    r = get(url, timeout=timeout)
    return (url, r.status_code, r.content)


def map_get(
    urls: list[str], *, timeout: float | None = None
) -> list[tuple[tuple[str, int, bytes] | None, Exception | None]]:
    """Concurrent GET for a list of URLs."""
    if not urls:
        return []
    timeout = clamp_request_timeout(timeout)
    workers = _pool_stats["workers"]
    SAFE_EXC = TRANSIENT_HTTP_EXC + (ValueError, TypeError, JSONDecodeError)
    results: list[tuple[tuple[str, int, bytes] | None, Exception | None]] = [(None, None)] * len(urls)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_idx = {ex.submit(_fetch_one, url, timeout): i for i, url in enumerate(urls)}
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                results[i] = (fut.result(), None)
            except SAFE_EXC as e:
                results[i] = (None, e)
    return results


__all__ = [
    "HTTPSession",
    "request",
    "async_request",
    "request_json",
    "get",
    "async_get",
    "post",
    "async_post",
    "put",
    "async_put",
    "delete",
    "async_delete",
    "pool_stats",
    "map_get",
    "clamp_request_timeout",
    "host_slot",
    "host_limit_snapshot",
    "reset_host_limit_state",
    "reload_host_limit",
]
