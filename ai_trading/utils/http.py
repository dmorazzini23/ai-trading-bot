"""HTTP utilities with default timeout, retry, and pooled concurrency."""

# AI-AGENT-REF: ensure timeouts

import os
import threading
import time
from ai_trading.utils.timing import HTTP_TIMEOUT
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


_DEFAULT_TIMEOUT = HTTP_TIMEOUT


def _ensure_timeout(kwargs: dict) -> dict:
    """Populate ``timeout`` in kwargs when missing."""  # AI-AGENT-REF: timeout guard
    if "timeout" not in kwargs or kwargs["timeout"] is None:
        kwargs["timeout"] = _DEFAULT_TIMEOUT
    return kwargs


def request(method: str, url: str, **kwargs) -> requests.Response:
    """Perform an HTTP request with a guaranteed timeout."""  # AI-AGENT-REF: timeout enforced
    kwargs = _ensure_timeout(kwargs)
    return requests.request(method, url, **kwargs)


def get(url: str, **kwargs) -> requests.Response:
    return request("GET", url, **kwargs)


def post(url: str, **kwargs) -> requests.Response:
    return request("POST", url, **kwargs)


def put(url: str, **kwargs) -> requests.Response:
    return request("PUT", url, **kwargs)


def delete(url: str, **kwargs) -> requests.Response:
    return request("DELETE", url, **kwargs)


def head(url: str, **kwargs) -> requests.Response:
    return request("HEAD", url, **kwargs)


def options(url: str, **kwargs) -> requests.Response:
    return request("OPTIONS", url, **kwargs)


# Lazy singletons
__HTTP_EXECUTOR: ThreadPoolExecutor | None = None
__HTTP_LOCK = threading.Lock()
__SESSIONS: dict[str, "requests.Session"] = {}
__HOST_SEMAPHORES: dict[str, threading.Semaphore] = {}
__HOST_RPS: dict[str, float] = {}
__HOST_RPS_WINDOWS: dict[str, list[float]] = {}


def _cfg_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:  # noqa: BLE001
        return default


def _cfg_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:  # noqa: BLE001
        return default


def _get_executor() -> ThreadPoolExecutor:
    global __HTTP_EXECUTOR
    if __HTTP_EXECUTOR is None:
        with __HTTP_LOCK:
            if __HTTP_EXECUTOR is None:
                workers = _cfg_int("HTTP_MAX_WORKERS", 8)
                __HTTP_EXECUTOR = ThreadPoolExecutor(
                    max_workers=max(2, workers), thread_name_prefix="httpw"
                )
    return __HTTP_EXECUTOR


def _session_for_host(host: str):
    s = __SESSIONS.get(host)
    if s is None:
        with __HTTP_LOCK:
            s = __SESSIONS.get(host)
            if s is None:
                s = requests.Session()
                retries = Retry(
                    total=_cfg_int("HTTP_RETRY_TOTAL", 3),
                    backoff_factor=0.5,
                    status_forcelist=(429, 500, 502, 503, 504),
                )
                pool_max = _cfg_int(
                    "HTTP_POOL_MAXSIZE",
                    max(_cfg_int("HTTP_MAX_WORKERS", 8), 10),
                )
                adapter = HTTPAdapter(
                    max_retries=retries,
                    pool_connections=pool_max,
                    pool_maxsize=pool_max,
                )
                s.mount("http://", adapter)
                s.mount("https://", adapter)
                __SESSIONS[host] = s
                key = f"HTTP_RPS_LIMIT_{host.replace('.', '_').replace('-', '_')}"
                if os.getenv(key):
                    __HOST_RPS[host] = _cfg_float(key, 0.0)
                    __HOST_RPS_WINDOWS[host] = []
    return s


def _host_semaphore(host: str) -> threading.Semaphore:
    sem = __HOST_SEMAPHORES.get(host)
    if sem is None:
        with __HTTP_LOCK:
            sem = __HOST_SEMAPHORES.get(host)
            if sem is None:
                sem = threading.Semaphore(_cfg_int("HTTP_MAX_PER_HOST", 6))
                __HOST_SEMAPHORES[host] = sem
    return sem


def _maybe_rate_limit(host: str):
    rps = __HOST_RPS.get(host, 0.0)
    if rps <= 0:
        return
    win = __HOST_RPS_WINDOWS[host]
    now = time.monotonic()
    while win and now - win[0] > 1.0:
        win.pop(0)
    if len(win) >= rps:
        sleep_for = 1.0 - (now - win[0])
        if sleep_for > 0:
            time.sleep(sleep_for)
    win.append(time.monotonic())


class HTTPSession:
    """HTTP session with default timeout and retry logic."""

    def __init__(self, timeout: int = 10, retries: int = 3):
        self.session = requests.Session()
        self.timeout = timeout

        # Configure retry strategy
        retry_strategy = Retry(
            total=retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"],
        )

        # Mount adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get(self, url: str, **kwargs) -> requests.Response:
        """GET request with default timeout."""  # AI-AGENT-REF: enforce timeout
        kwargs = _ensure_timeout(kwargs)
        return self.session.get(url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        """POST request with default timeout."""  # AI-AGENT-REF: enforce timeout
        kwargs = _ensure_timeout(kwargs)
        return self.session.post(url, **kwargs)

    def put(self, url: str, **kwargs) -> requests.Response:
        """PUT request with default timeout."""  # AI-AGENT-REF: enforce timeout
        kwargs = _ensure_timeout(kwargs)
        return self.session.put(url, **kwargs)

    def delete(self, url: str, **kwargs) -> requests.Response:
        """DELETE request with default timeout."""  # AI-AGENT-REF: enforce timeout
        kwargs = _ensure_timeout(kwargs)
        return self.session.delete(url, **kwargs)

    def head(self, url: str, **kwargs) -> requests.Response:
        """HEAD request with default timeout."""  # AI-AGENT-REF: enforce timeout
        kwargs = _ensure_timeout(kwargs)
        return self.session.head(url, **kwargs)

    def options(self, url: str, **kwargs) -> requests.Response:
        """OPTIONS request with default timeout."""  # AI-AGENT-REF: enforce timeout
        kwargs = _ensure_timeout(kwargs)
        return self.session.options(url, **kwargs)


# Bounded concurrency helpers


def map_get(
    urls: list[str],
    timeout: float | None = None,
    headers: dict | None = None,
) -> list[tuple[str, int, bytes]]:
    """Fetch multiple URLs concurrently while preserving order."""
    # Fast validation: every URL must include a scheme
    for u in urls:
        p = urlparse(u)
        if p.scheme not in {"http", "https"}:
            raise ValueError(f"Invalid URL (missing scheme): {u}")
    results: list[tuple[str, int, bytes]] = [("", 0, b"")] * len(urls)
    futs = []
    execu = _get_executor()
    tout = timeout or _cfg_float("HTTP_TIMEOUT_S", 10.0)

    def _task(idx: int, url: str):
        parsed = urlparse(url)
        host = parsed.netloc
        sem = _host_semaphore(host)
        sess = _session_for_host(host)
        _maybe_rate_limit(host)
        with sem:
            resp = sess.get(url, timeout=tout, headers=headers)
            return (idx, url, resp.status_code, resp.content)

    for i, u in enumerate(urls):
        futs.append(execu.submit(_task, i, u))

    for f in as_completed(futs):
        idx, url, code, content = f.result()
        results[idx] = (url, code, content)
    return results


def map_post(
    urls: list[str],
    data: list | None = None,
    timeout: float | None = None,
    headers: dict | None = None,
) -> list[tuple[str, int, bytes]]:
    """POST to multiple URLs concurrently while preserving order."""
    if data is None:
        data = [None] * len(urls)
    results: list[tuple[str, int, bytes]] = [("", 0, b"")] * len(urls)
    futs = []
    execu = _get_executor()
    tout = timeout or _cfg_float("HTTP_TIMEOUT_S", 10.0)

    def _task(idx: int, url: str, payload):
        parsed = urlparse(url)
        host = parsed.netloc
        sem = _host_semaphore(host)
        sess = _session_for_host(host)
        _maybe_rate_limit(host)
        with sem:
            resp = sess.post(url, data=payload, timeout=tout, headers=headers)
            return (idx, url, resp.status_code, resp.content)

    for i, (u, p) in enumerate(zip(urls, data, strict=False)):
        futs.append(execu.submit(_task, i, u, p))

    for f in as_completed(futs):
        idx, url, code, content = f.result()
        results[idx] = (url, code, content)
    return results


def pool_stats() -> dict:
    execu = __HTTP_EXECUTOR
    in_flight = execu._work_queue.qsize() if execu else 0
    sems = {h: getattr(s, "_value", 0) for h, s in __HOST_SEMAPHORES.items()}
    return {
        "workers": _cfg_int("HTTP_MAX_WORKERS", 8),
        "per_host": _cfg_int("HTTP_MAX_PER_HOST", 6),
        "pool_maxsize": _cfg_int(
            "HTTP_POOL_MAXSIZE", max(_cfg_int("HTTP_MAX_WORKERS", 8), 10)
        ),
        "hosts": list(__SESSIONS.keys()),
        "in_flight": in_flight,
        "host_semaphores": sems,
    }


# AI-AGENT-REF: HTTP safety module with default timeouts and retry logic
