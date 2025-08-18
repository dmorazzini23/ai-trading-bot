from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ai_trading.utils.timing import HTTP_TIMEOUT

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
    if "timeout" not in kwargs or kwargs["timeout"] is None:
        kwargs["timeout"] = HTTP_TIMEOUT
    return kwargs


def request(method: str, url: str, **kwargs) -> requests.Response:
    sess = _get_session()
    kwargs = _with_timeout(kwargs)
    _pool_stats["requests"] += 1
    try:
        resp = sess.request(method, url, **kwargs)
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
