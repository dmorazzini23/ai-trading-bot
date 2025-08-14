
import pandas as pd


def test_ichimoku_indicator_returns_dataframe(monkeypatch):
    from ai_trading.core import bot_engine as bot

    if not hasattr(bot.ta, "ichimoku"):
        setattr(bot.ta, "ichimoku", lambda *a, **k: (pd.DataFrame(), {}))

    # Patch pandas_ta.ichimoku to return a tuple (DataFrame, params)
    ich_df = pd.DataFrame({"ITS_9": [1.0], "IKS_26": [1.0]})
    monkeypatch.setattr(bot.ta, "ichimoku", lambda *a, **k: (ich_df, {"foo": 1}))
    df = pd.DataFrame({"high": [1, 2], "low": [0, 1], "close": [1, 2]})
    result = bot.ichimoku_indicator(df, "TEST", None)
    assert isinstance(result, tuple)
    out_df = result[0]
    assert isinstance(out_df, pd.DataFrame)
    assert "ITS_9" in out_df.columns
    assert "IKS_26" in out_df.columns


def test_compute_ichimoku_returns_df_pair(monkeypatch):
    from ai_trading.core import bot_engine as bot
    if not hasattr(bot.ta, "ichimoku"):
        setattr(bot.ta, "ichimoku", lambda *a, **k: (pd.DataFrame(), {}))
    ich_df = pd.DataFrame({"ITS_9": [1.0]})
    signal_df = pd.DataFrame({"ITSs_9": [1.0]})
    monkeypatch.setattr(bot.ta, "ichimoku", lambda *a, **k: (ich_df, signal_df))
    df1, df2 = bot.compute_ichimoku(pd.Series([1, 2]), pd.Series([1, 2]), pd.Series([1, 2]))
    assert isinstance(df1, pd.DataFrame)
    assert isinstance(df2, pd.DataFrame)
    assert "ITS_9" in df1.columns
    assert "ITSs_9" in df2.columns


def test_vwap_calculation():
    from ai_trading.indicators import calculate_vwap
    high = pd.Series([10, 11, 12])
    low = pd.Series([5, 6, 7])
    close = pd.Series([7, 8, 9])
    volume = pd.Series([1000, 1100, 1200])
    vwap = calculate_vwap(high, low, close, volume)
    assert vwap.iloc[-1] > 0

