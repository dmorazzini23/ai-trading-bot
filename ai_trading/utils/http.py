"""
HTTP utilities with default timeout and retry logic.

Provides a centralized HTTP session with sensible defaults for timeouts
and retry behavior to replace raw requests calls.
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"]
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """GET request with default timeout."""
        kwargs.setdefault('timeout', self.timeout)
        return self.session.get(url, **kwargs)
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """POST request with default timeout."""
        kwargs.setdefault('timeout', self.timeout)
        return self.session.post(url, **kwargs)
    
    def put(self, url: str, **kwargs) -> requests.Response:
        """PUT request with default timeout."""
        kwargs.setdefault('timeout', self.timeout)
        return self.session.put(url, **kwargs)
    
    def delete(self, url: str, **kwargs) -> requests.Response:
        """DELETE request with default timeout."""
        kwargs.setdefault('timeout', self.timeout)
        return self.session.delete(url, **kwargs)
    
    def head(self, url: str, **kwargs) -> requests.Response:
        """HEAD request with default timeout."""
        kwargs.setdefault('timeout', self.timeout)
        return self.session.head(url, **kwargs)
    
    def options(self, url: str, **kwargs) -> requests.Response:
        """OPTIONS request with default timeout."""
        kwargs.setdefault('timeout', self.timeout)
        return self.session.options(url, **kwargs)


# Default session instance
_default_session = HTTPSession()

# Convenience functions that use the default session
get = _default_session.get
post = _default_session.post
put = _default_session.put
delete = _default_session.delete
head = _default_session.head
options = _default_session.options

# AI-AGENT-REF: HTTP safety module with default timeouts and retry logic