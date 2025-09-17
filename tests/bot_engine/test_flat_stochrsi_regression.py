"""Regression tests for indicator preparation with flat price series."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from ai_trading.core import bot_engine


def _constant_minute_frame(rows: int = 300) -> pd.DataFrame:
    """Return a minute-level OHLCV frame with flat prices."""

    idx = pd.date_range("2024-01-01", periods=rows, freq="min")
    price = np.full(rows, 125.0)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": np.full(rows, 1_000),
        },
        index=idx,
    )


def test_prepare_indicators_constant_series_nan_rsi(monkeypatch: pytest.MonkeyPatch) -> None:
    """All-NaN RSI from pandas-ta should trigger the neutral fallback."""

    monkeypatch.setenv("PYTEST_RUNNING", "1")
    df = _constant_minute_frame()

    monkeypatch.setattr(
        bot_engine.ta,
        "rsi",
        lambda *_args, **_kwargs: pd.Series(np.nan, index=df.index),
    )

    engineered = bot_engine.prepare_indicators(df)

    required = {"rsi", "rsi_14", "ichimoku_conv", "ichimoku_base", "stochrsi"}
    assert not engineered.empty
    assert required.issubset(engineered.columns)
    assert np.isclose(engineered["rsi"], 50.0).all()
    assert np.isfinite(engineered["stochrsi"]).all()


def test_prepare_indicators_constant_series_zero_variance_rsi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constant RSI output should fall back to the manual neutral implementation."""

    monkeypatch.setenv("PYTEST_RUNNING", "1")
    df = _constant_minute_frame()

    monkeypatch.setattr(
        bot_engine.ta,
        "rsi",
        lambda *_args, **_kwargs: pd.Series(0.0, index=df.index),
    )

    engineered = bot_engine.prepare_indicators(df)

    assert not engineered.empty
    assert np.isclose(engineered["rsi"], 50.0).all()
    assert np.isfinite(engineered["stochrsi"]).all()


def test_prepare_indicators_constant_series_manual_rsi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manual RSI fallback should stabilise flat price windows at 50."""

    monkeypatch.setenv("PYTEST_RUNNING", "1")
    df = _constant_minute_frame()

    # Force the manual RSI implementation path by raising from pandas-ta helper.
    monkeypatch.setattr(
        bot_engine.ta,
        "rsi",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AttributeError("rsi")),
    )

    engineered = bot_engine.prepare_indicators(df)

    assert not engineered.empty
    assert np.isclose(engineered["rsi"], 50.0).all()
    assert np.isfinite(engineered["stochrsi"]).all()


def test_fetch_feature_data_constant_series(monkeypatch: pytest.MonkeyPatch) -> None:
    """_fetch_feature_data should not fall back to raw data for flat prices."""

    monkeypatch.setenv("PYTEST_RUNNING", "1")
    df = _constant_minute_frame()

    def _assign(df: pd.DataFrame, **cols: float) -> pd.DataFrame:
        out = df.copy()
        for name, value in cols.items():
            out[name] = value
        return out

    monkeypatch.setattr(bot_engine, "compute_macd", lambda frame: _assign(frame, macd=0.0))
    monkeypatch.setattr(bot_engine, "compute_atr", lambda frame: _assign(frame, atr=0.1))
    monkeypatch.setattr(bot_engine, "compute_vwap", lambda frame: _assign(frame, vwap=125.0))
    monkeypatch.setattr(
        bot_engine,
        "compute_sma",
        lambda frame: _assign(frame, sma_50=125.0, sma_200=125.0),
    )
    monkeypatch.setattr(bot_engine, "compute_macds", lambda frame: _assign(frame, macds=0.0))
    monkeypatch.setattr(bot_engine, "ensure_columns", lambda frame, cols, symbol: frame)

    ctx = SimpleNamespace(
        halt_manager=None,
        data_fetcher=SimpleNamespace(get_daily_df=lambda _ctx, _symbol: None),
    )

    raw_df, feat_df, skip = bot_engine._fetch_feature_data(ctx, object(), "FLAT", price_df=df)

    assert raw_df is not None
    assert skip is None
    assert feat_df is not None and not feat_df.empty

    required = {"macd", "atr", "vwap", "macds", "rsi", "stochrsi"}
    assert required.issubset(feat_df.columns)
    assert not feat_df["stochrsi"].isna().all()

