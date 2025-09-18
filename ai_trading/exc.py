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

    class HTTPError(Exception):
        """Fallback HTTPError with optional response attribute."""

        def __init__(self, *args, response=None, **kwargs):  # noqa: D401 - simple passthrough
            super().__init__(*args)
            self.response = response
COMMON_EXC = (TypeError, ValueError, KeyError, JSONDecodeError, RequestException, TimeoutError, ImportError)
TRANSIENT_HTTP_EXC = (RequestException, HTTPError, TimeoutError, OSError, ConnectionError)

class DataFeedUnavailable(RuntimeError):
    """Raised when required market data is unavailable."""

class InvalidBarsError(ValueError):
    """Raised when bar data is invalid or missing."""