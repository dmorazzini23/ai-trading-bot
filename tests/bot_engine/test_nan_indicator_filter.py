import numpy as np
import pytest
from types import SimpleNamespace
from typing import Any, cast

import ai_trading.core.bot_engine as bot_engine
from ai_trading.core.bot_engine import SignalManager, BotState

pd = pytest.importorskip("pandas")


def _base_df(length: int) -> Any:
    data = {
        "close": np.arange(length, dtype=float),
        "open": np.arange(length, dtype=float),
        "high": np.arange(length, dtype=float) + 1,
        "low": np.arange(length, dtype=float) - 1,
        "volume": np.ones(length),
        "macd": np.zeros(length),
        "vwap": np.ones(length),
        "macds": np.zeros(length),
        "atr": np.ones(length),
    }
    return pd.DataFrame(data)


def test_prepare_indicators_does_not_backfill_warmup_rows(monkeypatch):
    length = 40
    df = _base_df(length)

    rsi_values = [np.nan] * 14 + [45.0 + idx for idx in range(26)]
    monkeypatch.setattr(bot_engine.ta, "rsi", lambda *_args, **_kwargs: pd.Series(rsi_values))

    prepared = bot_engine.prepare_indicators(df)

    warmup_rsi = prepared.loc[prepared.index < 14, "rsi"]
    assert warmup_rsi.empty or warmup_rsi.isna().all()
    assert prepared["rsi"].iloc[-1] == 70.0
