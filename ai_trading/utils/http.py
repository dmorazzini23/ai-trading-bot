from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import (
    RequestException as RequestsRequestException,  # AI-AGENT-REF: catch raw requests errors
)
from urllib3.util.retry import Retry

from ai_trading.exc import (
    TRANSIENT_HTTP_EXC,
    JSONDecodeError,
    RequestException,
)
from ai_trading.logging import get_logger  # AI-AGENT-REF: centralized logging
from ai_trading.utils.retry import retry_call  # AI-AGENT-REF: retry helper
from ai_trading.utils.timing import HTTP_TIMEOUT, clamp_timeout  # AI-AGENT-REF: timeout clamp

_log = get_logger(__name__)

_session = None
_session_lock = threading.Lock()
_pool_stats = {
    "workers": int(os.getenv("HTTP_POOL_WORKERS", "8")),
    "per_host": int(os.getenv("HTTP_MAX_PER_HOST", "6")),
    "pool_maxsize": 32,
    "requests": 0,
    "responses": 0,
    "errors": 0,
}

# AI-AGENT-REF: Stage 2.1 build session with pooling metadata


def _build_session() -> requests.Session:
    _pool_stats["per_host"] = int(os.getenv("HTTP_MAX_PER_HOST", str(_pool_stats["per_host"])))
    _pool_stats["workers"] = int(os.getenv("HTTP_POOL_WORKERS", str(_pool_stats["workers"])))
    s = requests.Session()
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
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = _build_session()
    return _session


def _with_timeout(kwargs: dict) -> dict:
    """Ensure a clamped timeout is always provided."""
    # AI-AGENT-REF: unify timeout handling
    kwargs["timeout"] = clamp_timeout(kwargs.get("timeout"), default_non_test=HTTP_TIMEOUT)
    return kwargs


def _retry_config() -> tuple[int, float, float, float]:
    """Load retry knobs from settings if available."""  # AI-AGENT-REF: Stage 2.2
    retries, backoff, max_backoff, jitter = 3, 0.1, 2.0, 0.1
    try:  # Lazy import to avoid heavy config at import time
        from ai_trading.config import get_settings  # type: ignore

        s = get_settings()
        retries = int(getattr(s, "RETRY_MAX_ATTEMPTS", retries))
        backoff = float(getattr(s, "RETRY_BASE_DELAY", backoff))
        max_backoff = float(getattr(s, "RETRY_MAX_DELAY", max_backoff))
        jitter = float(getattr(s, "RETRY_JITTER", jitter))
    except (
        AttributeError,
        TypeError,
        ValueError,
        ImportError,
    ):  # pragma: no cover - settings optional
        pass
    return retries, backoff, max_backoff, jitter


def request(method: str, url: str, **kwargs) -> requests.Response:
    sess = _get_session()
    kwargs = _with_timeout(kwargs)
    retries, backoff, max_backoff, jitter = _retry_config()
    excs = (
        RequestException,
        RequestsRequestException,  # AI-AGENT-REF: support direct requests exception
        JSONDecodeError,
        TimeoutError,
        OSError,
    )

    attempt = {"n": 0}

    def _do_request() -> requests.Response:
        try:
            return sess.request(method, url, **kwargs)
        except excs as e:  # AI-AGENT-REF: log retry attempt
            attempt["n"] += 1
            log_fn = _log.warning if attempt["n"] == 1 else _log.debug
            log_fn(
                "HTTP_RETRY",
                extra={"attempt": attempt["n"], "attempts": retries, "error": str(e)},
            )
            raise

    _pool_stats["requests"] += 1
    try:
        resp = retry_call(
            _do_request,
            exceptions=excs,
            retries=retries,
            backoff=backoff,
            max_backoff=max_backoff,
            jitter=jitter,
        )
        _pool_stats["responses"] += 1
        return resp
    except excs as e:  # AI-AGENT-REF: Stage 2.2 final log
        _pool_stats["errors"] += 1
        _log.error("HTTP_GIVEUP", extra={"attempts": retries, "error": str(e)})
        raise


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
    """Concurrent GET for a list of URLs."""  # AI-AGENT-REF: Stage 2.1
    if not urls:
        return []
    workers = _pool_stats["workers"]
    SAFE_EXC = TRANSIENT_HTTP_EXC + (ValueError, TypeError, JSONDecodeError)
    results: list[tuple[tuple[str, int, bytes] | None, Exception | None]] = [(None, None)] * len(
        urls
    )
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_idx = {ex.submit(_fetch_one, url, timeout): i for i, url in enumerate(urls)}
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                results[i] = (fut.result(), None)
            except SAFE_EXC as e:  # AI-AGENT-REF: Stage 2.1 narrowed catch
                results[i] = (None, e)
    return results
