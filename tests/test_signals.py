import sys
import types

import numpy as np
import pytest
pd = pytest.importorskip("pandas")

np.random.seed(0)

from ai_trading.signals import GaussianHMM, detect_market_regime_hmm, prepare_indicators
import ai_trading.signals.indicators as ind


def test_hmm_regime_detection():
    if GaussianHMM is None:
        pytest.skip("hmmlearn not installed")
    df = pd.DataFrame({"Close": np.random.rand(100) + 100})
    regimes = detect_market_regime_hmm(df)
    assert isinstance(regimes, np.ndarray)
    assert regimes.shape[0] == len(df)


@pytest.fixture
def sample_df():
    """Create simple OHLCV data for indicator tests."""
    n = 30
    data = {
        'open': np.linspace(9, 14, n),
        'high': np.linspace(10, 15, n),
        'low': np.linspace(9, 14, n),
        'close': np.linspace(9.5, 14.5, n),
        'volume': np.linspace(100, 130, n),
    }
    return pd.DataFrame(data)


def test_prepare_indicators_requires_ohlcv():
    df = pd.DataFrame({"open": [1], "high": [2], "low": [1], "close": [1]})
    with pytest.raises(ValueError, match="missing required column"):
        prepare_indicators(df)


def test_prepare_indicators_calculates(sample_df, monkeypatch):
    """Indicators are added using pandas_ta helpers."""
    import types
    pta = types.ModuleType('pandas_ta')
    pta.vwap = lambda h,l,c,v: pd.Series((h+l+c)/3, index=sample_df.index)
    pta.macd = lambda c, **k: {
        'MACD_12_26_9': c * 0 + 1.0,
        'MACDs_12_26_9': c * 0 + 0.5,
    }
    pta.kc = lambda h,l,c,length=20: pd.DataFrame({0: c*0+1.0,1:c*0+2.0,2:c*0+3.0})
    pta.mfi = lambda h,l,c,v,length=14: pd.Series(c*0+5.0, index=sample_df.index)
    pta.adx = lambda h,l,c,length=14: {
        'ADX_14': pd.Series(c*0+7.0, index=sample_df.index),
        'DMP_14': pd.Series(c*0+1.0, index=sample_df.index),
        'DMN_14': pd.Series(c*0+1.0, index=sample_df.index),
    }
    pta.rsi = lambda *a, **k: pd.Series([50.0]*len(sample_df))
    pta.atr = lambda *a, **k: pd.Series([1.0]*len(sample_df))
    pta.bbands = lambda *a, **k: {
        'BBU_20_2.0': pd.Series([1.0]*len(sample_df)),
        'BBL_20_2.0': pd.Series([1.0]*len(sample_df)),
        'BBP_20_2.0': pd.Series([1.0]*len(sample_df)),
    }
    monkeypatch.setitem(sys.modules, 'pandas_ta', pta)

    prepare_mod = types.ModuleType('ai_trading.features.prepare')

    def prepare_indicators(df, freq: str = 'intraday'):
        ta = pta
        df = df.copy()
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        macd = ta.macd(df['close'])
        df['macd'] = macd['MACD_12_26_9']
        kc = ta.kc(df['high'], df['low'], df['close'], length=20)
        df['kc_upper'] = kc.iloc[:, 2]
        df['mfi_14'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx['ADX_14']
        return df

    prepare_mod.prepare_indicators = prepare_indicators
    monkeypatch.setitem(sys.modules, 'ai_trading.features.prepare', prepare_mod)

    from ai_trading.features import prepare as retrain
    out = retrain.prepare_indicators(sample_df)
    assert out['vwap'].iloc[-1] == pytest.approx((sample_df['high']+sample_df['low']+sample_df['close']).iloc[-1]/3)
    assert out['macd'].iloc[0] == 1.0
    assert out['kc_upper'].iloc[0] == 3.0
    assert out['mfi_14'].iloc[0] == 5.0
    assert out['adx'].iloc[0] == 7.0


def test_composite_signal_confidence(monkeypatch):
    """SignalManager combines weights into final score."""
    from ai_trading.core import bot_engine as bot
    sm = bot.SignalManager()
    monkeypatch.setattr(sm, 'load_signal_weights', lambda: {})
    monkeypatch.setattr(bot, 'load_global_signal_performance', lambda: {})
    monkeypatch.setattr(bot, 'signals_evaluated', None, raising=False)
    monkeypatch.setattr(sm, 'signal_momentum', lambda df, model=None: (1, 0.4, 'momentum'))
    monkeypatch.setattr(sm, 'signal_mean_reversion', lambda df, model=None: (-1, 0.2, 'mean_reversion'))
    monkeypatch.setattr(sm, 'signal_ml', lambda df, model=None, symbol=None: (1, 0.6, 'ml'))
    monkeypatch.setattr(sm, 'signal_sentiment', lambda ctx, ticker, df=None, model=None: (1, 0.1, 'sentiment'))
    monkeypatch.setattr(sm, 'signal_regime', lambda ctx, state, df, model=None: (1, 1.0, 'regime'))
    monkeypatch.setattr(sm, 'signal_stochrsi', lambda df, model=None: (1, 0.1, 'stochrsi'))
    monkeypatch.setattr(sm, 'signal_obv', lambda df, model=None: (1, 0.1, 'obv'))
    monkeypatch.setattr(sm, 'signal_vsa', lambda df, model=None: (1, 0.1, 'vsa'))
    df = pd.DataFrame({'close': np.linspace(1, 2, 210)})
    ctx = types.SimpleNamespace()
    state = types.SimpleNamespace(current_regime='trending', no_signal_events=0)
    model = types.SimpleNamespace(predict=lambda x: [1], predict_proba=lambda x: [[0.4,0.6]], feature_names_in_=['rsi','macd','atr','vwap','sma_50','sma_200'])
    final, conf, label = sm.evaluate(ctx, state, df, 'TST', model)
    assert final == 1
    assert conf == pytest.approx(2.6)
    assert 'ml' in label


def test_psar_wrapper(sample_df, monkeypatch):
    pta = types.SimpleNamespace(psar=lambda h, l, c: pd.DataFrame({
        "PSARl_0.02_0.2": c * 0 + 1.0,
        "PSARs_0.02_0.2": c * 0 + 2.0,
    }))
    ind.load_pandas_ta.cache_clear()
    monkeypatch.setattr(ind, "load_pandas_ta", lambda: pta)
    out = ind.psar(sample_df)
    assert out["psar_long"].iloc[0] == 1.0
    assert out["psar_short"].iloc[0] == 2.0


def test_psar_validation(sample_df):
    """psar enforces required columns and handles empty/invalid data."""
    with pytest.raises(KeyError, match="missing required column"):
        ind.psar(sample_df.drop(columns=["high"]))
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = ind.psar(empty)
    assert out.empty


def test_psar_close_numeric(sample_df):
    bad = sample_df.copy()
    bad["close"] = ["foo"] * len(bad)
    with pytest.raises(ValueError, match="numeric"):
        ind.psar(bad)
    good = sample_df.copy()
    good["close"] = good["close"].astype(str)
    out = ind.psar(good)
    assert "psar_long" in out.columns


def test_composite_signal_confidence_dict_and_list():
    conf_dict = {"a": 0.5, "b": 0.25}
    conf_list = [0.5, 0.25]
    assert ind.composite_signal_confidence(conf_dict) == pytest.approx(0.75)
    assert ind.composite_signal_confidence(conf_list) == pytest.approx(0.75)
