import types
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.core import bot_engine


def _sample_df():
    idx = pd.date_range("2024-01-01", periods=250, freq="min")
    data = {
        "open": pd.Series(range(1, 251), index=idx, dtype=float),
        "high": pd.Series(range(2, 252), index=idx, dtype=float),
        "low": pd.Series(range(0, 250), index=idx, dtype=float),
        "close": pd.Series(range(1, 251), index=idx, dtype=float),
        "volume": pd.Series([1000] * 250, index=idx, dtype=float),
    }
    return pd.DataFrame(data, index=idx)


def test_sma_features_available(monkeypatch):
    df = _sample_df()
    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", lambda symbol: df)
    monkeypatch.setattr(bot_engine, "prepare_indicators", lambda frame: frame)
    ctx = types.SimpleNamespace(data_fetcher=types.SimpleNamespace(get_daily_df=lambda ctx, s: df))
    raw, feat, skip = bot_engine._fetch_feature_data(ctx, None, "AAPL")
    assert skip is None
    assert "sma_50" in feat.columns
    assert "sma_200" in feat.columns

    class Model:
        feature_names_in_ = ["macd", "atr", "vwap", "macds", "sma_50", "sma_200"]

    feature_names = bot_engine._model_feature_names(Model())
    missing = [f for f in feature_names if f not in feat.columns]
    assert missing == []
