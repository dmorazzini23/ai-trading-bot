from __future__ import annotations
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
try:  # requests is optional
    import requests  # type: ignore
    from requests.adapters import HTTPAdapter  # type: ignore
    from requests.exceptions import RequestException as RequestsRequestException  # type: ignore
    REQUESTS_AVAILABLE = True
except Exception:  # pragma: no cover - requests missing
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

    class _StubResponse:
        pass

    class _RequestsStub:
        Session = _StubSession
        Response = _StubResponse
        exceptions = type("exc", (), {"RequestException": RequestsRequestException})

    requests = _RequestsStub()  # type: ignore

try:  # urllib3 is only needed when requests is available
    from urllib3.util.retry import Retry  # type: ignore
except Exception:  # pragma: no cover - fallback when urllib3 missing
    class Retry:  # type: ignore
        def __init__(self, *a, **k):
            pass
from ai_trading.exc import TRANSIENT_HTTP_EXC, JSONDecodeError, RequestException
from ai_trading.logging import get_logger
from ai_trading.utils.retry import retry_call
from .timing import HTTP_TIMEOUT, clamp_timeout, sleep

_log = get_logger(__name__)
_session = None
_session_lock = threading.Lock()
_pool_stats = {
    "workers": int(os.getenv("HTTP_POOL_WORKERS", os.getenv("HTTP_MAX_WORKERS", "8"))),
    "per_host": int(os.getenv("HTTP_MAX_PER_HOST", "6")),
    "pool_maxsize": 32,
    "requests": 0,
    "responses": 0,
    "errors": 0,
}


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

        def __init__(self, timeout: float | int | None = HTTP_TIMEOUT) -> None:
            super().__init__()
            self._timeout = clamp_timeout(timeout)
            _pool_stats["per_host"] = int(os.getenv("HTTP_MAX_PER_HOST", str(_pool_stats["per_host"])))
            _pool_stats["workers"] = int(
                os.getenv("HTTP_POOL_WORKERS", os.getenv("HTTP_MAX_WORKERS", str(_pool_stats["workers"])))
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
            kwargs["timeout"] = clamp_timeout(timeout)
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


def _with_timeout(kwargs: dict) -> dict:
    """Clamp provided timeout while allowing session defaults."""
    if "timeout" in kwargs and kwargs["timeout"] is not None:
        kwargs["timeout"] = clamp_timeout(kwargs["timeout"])
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
    kwargs = _with_timeout(kwargs)
    retries, backoff, max_backoff, jitter = _retry_config()
    excs = (RequestException, RequestsRequestException, JSONDecodeError, TimeoutError, OSError, ValueError)
    attempt = {"n": 0}

    def _do_request() -> requests.Response:
        try:
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
    if isinstance(timeout, tuple):
        to = (
            clamp_timeout(timeout[0]),
            clamp_timeout(timeout[1]),
        )
    else:
        t = clamp_timeout(timeout)
        to = (t, t)

    sess = _get_session()

    def _fetch() -> requests.Response:
        return sess.request(method, url, headers=headers, params=params, timeout=to)

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


def post(url: str, **kwargs) -> requests.Response:
    return request("POST", url, **kwargs)


def put(url: str, **kwargs) -> requests.Response:
    return request("PUT", url, **kwargs)


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
    "request_json",
    "get",
    "post",
    "put",
    "pool_stats",
    "map_get",
]
