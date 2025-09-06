import json


class DummyResp:
    """Lightweight stand-in for ``requests.Response`` used in tests."""

    def __init__(self, data=None, *, status_code=200, headers=None, text=None):
        self._data = data or {}
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "application/json"}
        if text is None:
            text = json.dumps(self._data)
        self.text = text
        self.content = text.encode()

    def json(self):
        """Return the preloaded JSON payload or an empty dict."""
        return self._data
