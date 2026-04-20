import datetime
import types

import pytest

from ai_trading.core.startup_runtime import (
    _run_trade_updates_stream,
    initial_rebalance_runtime,
)


pd = pytest.importorskip("pandas")


class _DummyFetcher:
    def get_daily_df(self, ctx, symbol):
        return pd.DataFrame({"close": [100.0]})


class _DummyAPI:
    def __init__(self) -> None:
        self.positions: dict[str, int] = {}

    def get_account(self):
        return types.SimpleNamespace(cash=1000.0, equity=1000.0, buying_power=1000.0)

    def list_positions(self):
        return [types.SimpleNamespace(symbol=s, qty=q) for s, q in self.positions.items()]


def test_initial_rebalance_runtime_initializes_missing_tracking_attrs(monkeypatch):
    from ai_trading.core import bot_engine

    ctx = types.SimpleNamespace(api=_DummyAPI(), data_fetcher=_DummyFetcher())

    class FakeDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime(2025, 7, 26, 0, 16, tzinfo=datetime.UTC)

    monkeypatch.setattr(bot_engine, "datetime", FakeDateTime)
    monkeypatch.setattr(
        bot_engine,
        "submit_order",
        lambda ctx_, symbol, qty, side: ctx_.api.positions.setdefault(symbol, qty) or object(),
    )

    initial_rebalance_runtime(ctx, ["AAPL"])

    assert isinstance(ctx.rebalance_ids, dict)
    assert isinstance(ctx.rebalance_attempts, dict)
    assert isinstance(ctx.rebalance_buys, dict)
    assert ctx.api.positions["AAPL"] == 10


def test_run_trade_updates_stream_logs_failure() -> None:
    warnings: list[tuple[str, dict[str, object] | None]] = []
    be = types.SimpleNamespace(
        ALPACA_API_KEY="key",
        ALPACA_SECRET_KEY="secret",
        trading_client=object(),
        state=object(),
        logger=types.SimpleNamespace(
            warning=lambda event, extra=None: warnings.append((event, extra))
        ),
    )
    ctx = types.SimpleNamespace(stream_event=object())

    async def _raise(*args, **kwargs):
        raise RuntimeError("stream boom")

    _run_trade_updates_stream(be, ctx, _raise)

    assert warnings == [
        (
            "TRADE_UPDATES_STREAM_FAILED",
            {"error_type": "RuntimeError", "detail": "stream boom"},
        )
    ]
