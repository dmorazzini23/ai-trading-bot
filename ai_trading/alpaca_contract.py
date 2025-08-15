from types import SimpleNamespace


class MockClient:
    """Test helper used by unit tests to simulate Alpaca client."""

    def __init__(self, *args, **kwargs):
        self.last_payload = None

    def submit_order(self, **kwargs):
        self.last_payload = SimpleNamespace(**kwargs)
        return SimpleNamespace(id="1", status="mock", **kwargs)


__all__ = ["MockClient"]
