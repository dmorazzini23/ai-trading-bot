from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ai_trading import indicators as ind


def test_indicator_validation_paths_and_fallbacks() -> None:
    high = pd.Series(np.linspace(10.0, 20.0, 60))
    low = high - 1.0
    close = high - 0.5

    cloud, signal = ind.ichimoku_fallback(high, low, close)
    assert {"ITS_9", "IKS_26", "ISA_26", "ISB_52"} <= set(cloud.columns)
    assert "ICS_26" not in cloud.columns
    assert not signal.empty
    empty_cloud, empty_signal = ind.ichimoku_fallback(pd.Series([], dtype=float), low, close)
    assert empty_cloud.empty
    assert empty_signal.empty

    assert ind.ema(tuple(close), 5).notna().all()
    assert ind.ema(tuple(), 5).empty
    assert ind.sma(tuple(close), 5).notna().any()
    assert ind.sma(tuple(close), 0).empty
    assert ind.rsi(tuple(close), 14).shape[0] == len(close)
    assert ind.rsi(tuple([1.0, 2.0]), 14).empty
    assert ind.atr(high, low, close, 14).notna().any()
    assert ind.atr(high, low.iloc[:-1], close, 14).empty


def test_dataframe_indicator_mutators_and_signal_helpers() -> None:
    frame = pd.DataFrame(
        {
            "open": np.linspace(9.0, 19.0, 60),
            "high": np.linspace(10.0, 20.0, 60),
            "low": np.linspace(8.0, 18.0, 60),
            "close": np.linspace(9.5, 19.5, 60),
            "volume": np.linspace(100.0, 200.0, 60),
        },
        index=pd.date_range("2026-01-01", periods=60),
    )

    with_ema = ind.compute_ema(frame.copy(), [3])
    with_sma = ind.compute_sma(frame.copy(), [3])
    with_bollinger = ind.compute_bollinger(frame.copy(), window=5)
    with_atr = ind.compute_atr(frame.copy(), [5])

    assert "EMA_3" in with_ema
    assert "SMA_3" in with_sma
    assert {"MB", "STD", "UB", "LB", "BollingerWidth"} <= set(with_bollinger.columns)
    assert {"TR_5", "ATR_5"} <= set(with_atr.columns)

    macd, signal = ind.calculate_macd(frame["close"])
    assert len(macd) == len(frame)
    assert len(signal) == len(frame)
    assert ind.calculate_vwap(frame["high"], frame["low"], frame["close"], frame["volume"]).notna().all()
    assert ind.get_rsi_signal(frame[["close"]]).shape[0] == len(frame)
    assert ind.get_vwap_bias(frame["close"], frame["high"], frame["low"], frame["volume"]).notna().all()

    ind._INDICATOR_CACHE.clear()  # noqa: SLF001
    first = ind.cached_atr_trailing_stop("AAPL", frame, period=5)
    second = ind.cached_atr_trailing_stop("AAPL", frame, period=5)
    assert first is second
    assert ind.cached_atr_trailing_stop("AAPL", pd.DataFrame()).empty


def test_array_indicators_and_error_paths() -> None:
    prices = np.array([10.0, 11.0, 12.0])
    volumes = np.array([100.0, 200.0, 300.0])
    assert ind.vwap(prices, volumes) == pytest.approx((10 * 100 + 11 * 200 + 12 * 300) / 600)
    assert ind.vwap(np.array([]), volumes) == 0.0
    assert ind.vwap(prices, np.array([0.0, 0.0, 0.0])) == 0.0

    channel = ind.donchian_channel(np.array([1.0, 3.0, 2.0]), np.array([0.5, 1.0, 1.5]), period=2)
    assert channel == {"upper": 3.0, "lower": 1.0}
    assert ind.donchian_channel(np.array([1.0]), np.array([1.0]), period=2) == {"upper": 0.0, "lower": 0.0}

    assert ind.obv(np.array([1.0, 2.0, 1.5, 1.5]), np.array([10.0, 20.0, 30.0, 40.0])).tolist() == [0.0, 20.0, -10.0, -10.0]
    assert ind.obv(np.array([1.0]), np.array([10.0])).tolist() == [0.0]
    with pytest.raises(ValueError, match="same-length"):
        ind.obv(np.array([1.0, 2.0]), np.array([1.0]))

    stoch = ind.stochastic_rsi(np.array([1.0, 2.0, 3.0, 2.0]), period=3)
    assert stoch.shape == (4,)
    hurst = ind.hurst_exponent(pd.Series(np.linspace(1.0, 2.0, 200)))
    assert isinstance(hurst, float)
