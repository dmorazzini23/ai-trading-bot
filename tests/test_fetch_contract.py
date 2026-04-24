import os
import sys
import types
import asyncio
from typing import Any, cast

import pytest

pd = pytest.importorskip("pandas")
os.environ.setdefault("ALPACA_API_KEY", "dummy")
os.environ.setdefault("ALPACA_SECRET_KEY", "dummy")

from ai_trading.data import fetch as data_fetcher


class DummyClient:
    pass


def test_get_bars_never_none(monkeypatch):
    now = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": [now],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [100],
        }
    )
    monkeypatch.setattr(
        data_fetcher,
        "_alpaca_get_bars",
        lambda client, symbol, start, end, timeframe="1Day": df,
    )
    result = data_fetcher.get_bars(
        "AAPL", "1Day", now - pd.Timedelta(days=1), now, feed=cast(Any, DummyClient())
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert str(result["timestamp"].dt.tz) == "UTC"


def test_get_bars_requires_settings(monkeypatch):
    now = pd.Timestamp("2024-01-01", tz="UTC")
    monkeypatch.setattr(data_fetcher, "_current_settings", lambda: None)
    with pytest.raises(RuntimeError, match="Configuration is unavailable"):
        data_fetcher.get_bars(
            "AAPL", "1Day", now - pd.Timedelta(days=1), now
        )


@pytest.mark.asyncio
async def test_run_with_concurrency_returns_worker_failures() -> None:
    async def _ok() -> str:
        return "ok"

    async def _fail() -> str:
        raise RuntimeError("fetch failed")

    results, succeeded, failed = await data_fetcher.run_with_concurrency(
        2,
        [_ok(), _fail()],
    )

    assert results[0] == "ok"
    assert isinstance(results[1], RuntimeError)
    assert succeeded == 1
    assert failed == 1


@pytest.mark.asyncio
async def test_run_with_concurrency_propagates_cancellation() -> None:
    cleanup = asyncio.Event()

    async def _cancel() -> str:
        raise asyncio.CancelledError()

    async def _slow() -> str:
        try:
            await asyncio.sleep(30)
        finally:
            cleanup.set()
        return "slow"

    with pytest.raises(asyncio.CancelledError):
        await data_fetcher.run_with_concurrency(2, [_cancel(), _slow()])

    assert cleanup.is_set()
