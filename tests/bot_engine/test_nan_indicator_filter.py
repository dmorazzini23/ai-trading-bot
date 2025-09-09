import numpy as np
import pytest
from types import SimpleNamespace

import ai_trading.core.bot_engine as bot_engine
from ai_trading.core.bot_engine import SignalManager, BotState

pd = pytest.importorskip("pandas")


def _patch_signals(monkeypatch):
    monkeypatch.setattr(SignalManager, "signal_momentum", lambda self, df, model=None: None)
    monkeypatch.setattr(SignalManager, "signal_mean_reversion", lambda self, df, model=None: None)
    monkeypatch.setattr(
        SignalManager,
        "signal_ml",
        lambda self, df, model=None, symbol=None: None,
    )
    monkeypatch.setattr(
        SignalManager,
        "signal_sentiment",
        lambda self, ctx, ticker, df, model=None: None,
    )
    monkeypatch.setattr(
        SignalManager,
        "signal_regime",
        lambda self, ctx, state, df, model=None: None,
    )
    monkeypatch.setattr(SignalManager, "signal_obv", lambda self, df, model=None: None)
    monkeypatch.setattr(SignalManager, "signal_vsa", lambda self, df, model=None: None)
    monkeypatch.setattr(SignalManager, "load_signal_weights", lambda self: {})
    monkeypatch.setattr(bot_engine, "load_global_signal_performance", lambda: {})
    monkeypatch.setattr(bot_engine, "signals_evaluated", None, raising=False)


def _base_df(length: int) -> pd.DataFrame:
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


def test_nan_indicator_rows_ignored(monkeypatch):
    _patch_signals(monkeypatch)
    sm = SignalManager()
    state = BotState()
    ctx = SimpleNamespace(signal_manager=sm)
    length = 210
    df = _base_df(length)
    df["rsi"] = [30.0] * (length - 1) + [np.nan]
    df["rsi_14"] = df["rsi"]
    df["ichimoku_conv"] = 1.0
    df["ichimoku_base"] = 1.0
    df["stochrsi"] = [0.1] * (length - 1) + [np.nan]
    signal, _, label = sm.evaluate(ctx, state, df, "TEST", None)
    assert signal == 1
    assert label == "stochrsi"


def test_all_nan_indicators_return_no_data(monkeypatch):
    _patch_signals(monkeypatch)
    sm = SignalManager()
    state = BotState()
    ctx = SimpleNamespace(signal_manager=sm)
    length = 210
    df = _base_df(length)
    df["rsi"] = np.nan
    df["rsi_14"] = np.nan
    df["ichimoku_conv"] = np.nan
    df["ichimoku_base"] = np.nan
    df["stochrsi"] = np.nan
    signal, conf, label = sm.evaluate(ctx, state, df, "TEST", None)
    assert (signal, conf, label) == (0.0, 0.0, "no_data")

