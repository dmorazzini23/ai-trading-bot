from __future__ import annotations

import pytest

from ai_trading.data import fetch as data_fetch


@pytest.mark.parametrize(
    ("feed", "sip_configured", "expected"),
    [
        ("iex", False, False),
        ("iex", True, True),
        ("sip", False, True),
        ("yahoo", False, True),
        (None, False, True),
    ],
)
def test_should_disable_alpaca_on_empty(feed, sip_configured, expected, monkeypatch):
    monkeypatch.setattr(data_fetch, "_sip_configured", lambda: sip_configured, raising=False)
    assert data_fetch._should_disable_alpaca_on_empty(feed) is expected
