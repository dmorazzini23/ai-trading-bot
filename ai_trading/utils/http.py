from __future__ import annotations

import json  # AI-AGENT-REF: JSON decode error handling
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ai_trading.logging import get_logger  # AI-AGENT-REF: centralized logging
from ai_trading.utils.retry import retry_call  # AI-AGENT-REF: retry helper
from ai_trading.utils.timing import HTTP_TIMEOUT, clamp_timeout  # AI-AGENT-REF: timeout clamp

_log = get_logger(__name__)

_session = None
_session_lock = threading.Lock()
_pool_stats = {
    "max_workers": int(os.getenv("HTTP_POOL_WORKERS", "8")),
    "requests": 0,
    "responses": 0,
    "errors": 0,
}


def _build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=retries)
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
    """Load retry knobs from settings if available."""
    # AI-AGENT-REF: optional configuration
    attempts, base, max_delay, jitter = 3, 0.1, 2.0, 0.1
    try:  # Lazy import to avoid heavy config at import time
        from ai_trading.config import get_settings  # type: ignore

        s = get_settings()
        attempts = int(getattr(s, "RETRY_MAX_ATTEMPTS", attempts))
        base = float(getattr(s, "RETRY_BASE_DELAY", base))
        max_delay = float(getattr(s, "RETRY_MAX_DELAY", max_delay))
        jitter = float(getattr(s, "RETRY_JITTER", jitter))
    except Exception:  # pragma: no cover - settings optional
        pass
    return attempts, base, max_delay, jitter


def request(method: str, url: str, **kwargs) -> requests.Response:
    sess = _get_session()
    kwargs = _with_timeout(kwargs)
    attempts, base, max_delay, jitter = _retry_config()
    excs = (
        requests.exceptions.RequestException,
        json.JSONDecodeError,
        TimeoutError,
        OSError,
    )

    attempt = {"n": 0}

    def _do_request() -> requests.Response:
        try:
            return sess.request(method, url, **kwargs)
        except excs as e:  # AI-AGENT-REF: log retry attempt
            attempt["n"] += 1
            _log.warning(
                "HTTP_RETRY",
                extra={"attempt": attempt["n"], "attempts": attempts, "error": str(e)},
            )
            raise

    _pool_stats["requests"] += 1
    try:
        resp = retry_call(
            _do_request,
            exceptions=excs,
            attempts=attempts,
            base_delay=base,
            max_delay=max_delay,
            jitter=jitter,
        )
        _pool_stats["responses"] += 1
        return resp
    except Exception:
        _pool_stats["errors"] += 1
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
    try:
        r = get(url, timeout=timeout)
        return (url, r.status_code, r.content)
    except Exception:
        return (url, 599, b"")


def map_get(urls: list[str], *, timeout: float | None = None) -> list[tuple[str, int, bytes]]:
    """Concurrent GET for a list of URLs. Returns list of (url, status_code, body)."""
    if not urls:
        return []
    max_workers = _pool_stats["max_workers"]
    out: list[tuple[str, int, bytes]] = [("", 0, b"")] * len(urls)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {ex.submit(_fetch_one, url, timeout): i for i, url in enumerate(urls)}
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                out[i] = fut.result()
            except Exception:
                out[i] = (urls[i], 599, b"")
    return out
