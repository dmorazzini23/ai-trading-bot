import pytest
from datetime import UTC, datetime
from types import SimpleNamespace

from ai_trading.data.bars import BarsFetchFailed
from ai_trading.data.fetch.core import fetch


class DummySession:
    def __init__(self):
        self.called = False

    def get(self, url, **kwargs):
        self.called = True
        return SimpleNamespace(url=url, kwargs=kwargs)


def test_fetch_requires_session():
    with pytest.raises(ValueError):
        fetch("https://example.com")


def test_fetch_uses_session():
    sess = DummySession()
    resp = fetch("https://example.com", session=sess)
    assert sess.called
    assert resp.url == "https://example.com"


def test_fetch_propagates_bars_sentinel():
    sentinel = BarsFetchFailed(symbol="SPY", feed="iex", since=datetime(2024, 1, 2, tzinfo=UTC))

    class SentinelSession(DummySession):
        def get(self, url, **kwargs):
            super().get(url, **kwargs)
            return sentinel

    sess = SentinelSession()
    resp = fetch("https://example.com", session=sess)
    assert resp is sentinel
