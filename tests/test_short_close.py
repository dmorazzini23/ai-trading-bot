import types

from ai_trading.portfolio.short_close import short_close


def test_short_close_calls_submit():
    class DummyAPI:
        def list_positions(self):
            return [
                types.SimpleNamespace(symbol="TSLA", qty=-5),
                types.SimpleNamespace(symbol="LONG", qty=10),
            ]

    calls: list[tuple[str, int, str]] = []

    def fake_submit(symbol: str, qty: int, side: str) -> None:
        calls.append((symbol, qty, side))

    count = short_close(DummyAPI(), fake_submit)
    assert count == 1
    assert calls == [("TSLA", 5, "buy")]
