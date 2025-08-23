from __future__ import annotations
from json import JSONDecodeError
try:
    import requests
    RequestException = requests.exceptions.RequestException
    try:
        from requests.exceptions import HTTPError
    except ImportError:
        HTTPError = Exception
except ImportError:

    class RequestException(Exception):
        pass
    HTTPError = Exception
COMMON_EXC = (TypeError, ValueError, KeyError, JSONDecodeError, RequestException, TimeoutError, ImportError)
TRANSIENT_HTTP_EXC = (RequestException, HTTPError, TimeoutError, OSError, ConnectionError)

class DataFeedUnavailable(RuntimeError):
    """Raised when required market data is unavailable."""

class InvalidBarsError(ValueError):
    """Raised when bar data is invalid or missing."""