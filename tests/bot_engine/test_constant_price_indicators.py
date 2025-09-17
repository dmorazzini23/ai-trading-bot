import types

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.core import bot_engine


def _constant_price_frame(rows: int = 120):
    index = pd.date_range(
        "2024-01-01", periods=rows, freq="T", tz="UTC"
    )
    prices = pd.Series([100.0] * rows, index=index)
    data = {
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": pd.Series([1_000.0] * rows, index=index),
    }
    return pd.DataFrame(data, index=index)


def test_prepare_indicators_constant_series():
    frame = _constant_price_frame()

    engineered = bot_engine.prepare_indicators(frame)

    assert not engineered.empty
    for column in ("rsi", "stochrsi", "ichimoku_conv", "ichimoku_base"):
        assert column in engineered.columns
        assert engineered[column].notna().any()
    assert engineered["stochrsi"].between(0, 1, inclusive="both").all()


def test_fetch_feature_data_retains_indicators_for_constant_series(monkeypatch):
    frame = _constant_price_frame()
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", lambda symbol: frame.copy())

    def _with_column(df: pd.DataFrame, column: str, value: float) -> pd.DataFrame:
        result = df.copy()
        result[column] = value
        return result

    monkeypatch.setattr(
        bot_engine,
        "compute_macd",
        lambda df: _with_column(df, "macd", 0.0),
    )
    monkeypatch.setattr(
        bot_engine,
        "compute_atr",
        lambda df: _with_column(df, "atr", 0.0),
    )
    monkeypatch.setattr(
        bot_engine,
        "compute_vwap",
        lambda df: _with_column(df, "vwap", 100.0),
    )
    monkeypatch.setattr(
        bot_engine,
        "compute_sma",
        lambda df: _with_column(_with_column(df, "sma_50", 100.0), "sma_200", 100.0),
    )
    monkeypatch.setattr(
        bot_engine,
        "compute_macds",
        lambda df: _with_column(df, "macds", 0.0),
    )
    monkeypatch.setattr(
        bot_engine,
        "ensure_columns",
        lambda df, *_args, **_kwargs: df,
    )

    ctx = types.SimpleNamespace(
        data_fetcher=types.SimpleNamespace(get_daily_df=lambda _ctx, _symbol: frame)
    )

    raw_df, feat_df, skip_flag = bot_engine._fetch_feature_data(ctx, None, "CONST")

    assert skip_flag is None
    assert feat_df is not None
    assert not feat_df.empty
    for column in ("rsi", "stochrsi", "ichimoku_conv", "ichimoku_base"):
        assert column in feat_df.columns
    assert feat_df["stochrsi"].notna().all()
