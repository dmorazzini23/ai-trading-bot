"""Tests for short_close utility."""

from types import SimpleNamespace

from ai_trading.portfolio.short_close import short_close


def test_short_close_calls_submit():
    """short_close should call broker.submit for each short position."""

    class DummyAPI:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, str]] = []

        def list_positions(self):
            return [
                SimpleNamespace(symbol="TSLA", qty=-5),
                SimpleNamespace(symbol="LONG", qty=10),
            ]

        def submit(self, symbol: str, qty: int, side: str) -> None:
            self.calls.append((symbol, qty, side))

    api = DummyAPI()
    count = short_close(api, api.submit)
    assert count == 1
    assert api.calls == [("TSLA", 5, "buy")]
