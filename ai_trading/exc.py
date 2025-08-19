# ruff: noqa
from __future__ import annotations

from json import JSONDecodeError

# Optional requests import with safe fallbacks
try:  # pragma: no cover
    import requests  # type: ignore
    RequestException = requests.exceptions.RequestException  # type: ignore[attr-defined]
    try:  # pragma: no cover
        from requests.exceptions import HTTPError  # type: ignore
    except ImportError:  # pragma: no cover
        HTTPError = Exception
except ImportError:  # pragma: no cover
    class RequestException(Exception): ...
    HTTPError = Exception  # minimal fallback

# A common family for “expected” programming/data/HTTP parse errors
COMMON_EXC = (
    TypeError,
    ValueError,
    KeyError,
    JSONDecodeError,
    RequestException,
    TimeoutError,
    ImportError,
)

# Transient network/IO-ish errors appropriate for retry backoff
TRANSIENT_HTTP_EXC = (
    RequestException,
    HTTPError,
    TimeoutError,
    OSError,          # DNS / socket hiccups
    ConnectionError,  # builtin
)
