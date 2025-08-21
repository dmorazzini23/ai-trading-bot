from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# AI-AGENT-REF: provide retrying session with default timeouts


class TimeoutSession(requests.Session):
    """Requests Session that injects a default timeout if none is provided."""  # AI-AGENT-REF

    def __init__(self, default_timeout: tuple[float, float] = (5.0, 10.0)) -> None:
        super().__init__()
        self._default_timeout = default_timeout

    def request(self, method, url, **kwargs):  # type: ignore[override]
        if "timeout" not in kwargs or kwargs["timeout"] is None:
            kwargs["timeout"] = self._default_timeout
        return super().request(method, url, **kwargs)


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
    """Create a session with urllib3 Retry and default timeout."""  # AI-AGENT-REF

    s = TimeoutSession(default_timeout=(connect_timeout, read_timeout))
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=("GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry, pool_connections=pool_maxsize, pool_maxsize=pool_maxsize
    )
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def set_global_session(s: TimeoutSession) -> None:
    """Register global session singleton."""  # AI-AGENT-REF

    global _GLOBAL_SESSION
    _GLOBAL_SESSION = s


def get_global_session() -> TimeoutSession:
    """Return the global session, building a default if missing."""  # AI-AGENT-REF

    if _GLOBAL_SESSION is None:
        set_global_session(build_retrying_session())
    return _GLOBAL_SESSION  # type: ignore[return-value]

