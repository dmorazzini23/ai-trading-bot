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


def test_short_close_prefers_get_all_positions():
    """alpaca-py clients expose get_all_positions instead of list_positions."""

    class DummyAPI:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, str]] = []
            self.list_positions_called = False

        def get_all_positions(self):
            return [SimpleNamespace(symbol="NVDA", qty="-2")]

        def list_positions(self):
            self.list_positions_called = True
            return []

        def submit(self, symbol: str, qty: int, side: str) -> None:
            self.calls.append((symbol, qty, side))

    api = DummyAPI()

    count = short_close(api, api.submit)

    assert count == 1
    assert api.calls == [("NVDA", 2, "buy")]
    assert api.list_positions_called is False


def test_short_close_falls_back_to_list_positions():
    """Legacy clients remain supported when get_all_positions is unavailable."""

    class DummyAPI:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, str]] = []

        def list_positions(self):
            return [SimpleNamespace(symbol="TSLA", qty="-3")]

        def submit(self, symbol: str, qty: int, side: str) -> None:
            self.calls.append((symbol, qty, side))

    api = DummyAPI()

    count = short_close(api, api.submit)

    assert count == 1
    assert api.calls == [("TSLA", 3, "buy")]
