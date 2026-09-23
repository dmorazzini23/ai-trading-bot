from ai_trading.core import bot_engine  # replace old bot import
from typing import Any, cast
from types import SimpleNamespace

import pandas as pd
import pytest


class _DummyTradingClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


def test_runner_starts(monkeypatch: pytest.MonkeyPatch):
    ctx = bot_engine.ctx
    monkeypatch.setattr(cast(Any, ctx), "api", _DummyTradingClient(), raising=False)
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-02", periods=3, tz="UTC"),
        "open": [1.0] * 3, "high": [1.0] * 3, "low": [1.0] * 3,
        "close": [1.0] * 3, "volume": [1] * 3,
    })
    monkeypatch.setattr(
        cast(Any, ctx),
        "data_fetcher",
        SimpleNamespace(get_daily_df=lambda _ctx, _symbol: frame),
        raising=False,
    )
    symbols = ["AAPL", "MSFT"]
    summary = bot_engine.pre_trade_health_check(ctx, symbols, min_rows=2)
    assert summary["checked"] == 2
    assert summary["failures"] == []
