from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.features import prepare as prepare_mod


def _market_frame(rows: int = 64, *, unsorted: bool = False) -> Any:
    index = pd.date_range("2026-01-05", periods=rows, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100.0, 132.0, rows), index=index)
    frame = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(10_000.0, 12_000.0, rows),
        },
        index=index,
    )
    if unsorted:
        return frame.iloc[[2, 0, 1, *range(3, rows)]]
    return frame


def _series_like(values: Any, value: float | None = None) -> Any:
    if value is None:
        return pd.Series(np.asarray(values, dtype=float), index=values.index)
    return pd.Series(value, index=values.index, dtype=float)


class _CoreTA:
    @staticmethod
    def vwap(high: Any, low: Any, close: Any, volume: Any) -> Any:
        return (high + low + close) / 3.0

    @staticmethod
    def rsi(close: Any, length: int = 14) -> Any:
        return _series_like(close, 51.0)

    @staticmethod
    def atr(high: Any, low: Any, close: Any, length: int = 14) -> Any:
        return (high - low).abs() + 0.25


class _FullTA(_CoreTA):
    @staticmethod
    def kc(high: Any, low: Any, close: Any, length: int = 20) -> Any:
        return pd.DataFrame({"lower": close - 2.0, "mid": close, "upper": close + 2.0})

    @staticmethod
    def macd(close: Any, fast: int = 12, slow: int = 26, signal: int = 9) -> Any:
        return pd.DataFrame(
            {"MACD_12_26_9": close * 0.01, "MACDs_12_26_9": close * 0.02},
            index=close.index,
        )

    @staticmethod
    def bbands(close: Any, length: int = 20) -> Any:
        return pd.DataFrame(
            {
                "BBU_20_2.0": close + 4.0,
                "BBL_20_2.0": close - 4.0,
                "BBP_20_2.0": _series_like(close, 0.6),
            },
            index=close.index,
        )

    @staticmethod
    def adx(high: Any, low: Any, close: Any, length: int = 14) -> Any:
        return pd.DataFrame(
            {
                "ADX_14": _series_like(close, 22.0),
                "DMP_14": _series_like(close, 27.0),
                "DMN_14": _series_like(close, 12.0),
            },
            index=close.index,
        )

    @staticmethod
    def cci(high: Any, low: Any, close: Any, length: int = 20) -> Any:
        return _series_like(close, -80.0)

    @staticmethod
    def mfi(high: Any, low: Any, close: Any, volume: Any, length: int = 14) -> Any:
        return _series_like(close, 44.0)

    @staticmethod
    def tema(close: Any, length: int = 10) -> Any:
        return close + 0.1

    @staticmethod
    def willr(high: Any, low: Any, close: Any, length: int = 14) -> Any:
        return _series_like(close, -25.0)

    @staticmethod
    def psar(high: Any, low: Any, close: Any) -> Any:
        return pd.DataFrame({"PSARl_0.02_0.2": low, "PSARs_0.02_0.2": high}, index=close.index)

    @staticmethod
    def ichimoku(high: Any, low: Any, close: Any) -> Any:
        conv = pd.DataFrame({"conv": (high + low) / 2.0}, index=close.index)
        base = pd.DataFrame({"base": (high + low + close) / 3.0}, index=close.index)
        return conv, base

    @staticmethod
    def stochrsi(close: Any) -> Any:
        return pd.DataFrame({"STOCHRSIk_14_14_3_3": _series_like(close, 0.7)}, index=close.index)

    @staticmethod
    def sma(close: Any, length: int) -> Any:
        return close.rolling(length, min_periods=1).mean()


class _ExplodingTA(_CoreTA):
    @staticmethod
    def kc(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("kc boom")

    @staticmethod
    def macd(*_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("macd boom")

    @staticmethod
    def bbands(*_args: Any, **_kwargs: Any) -> Any:
        raise AttributeError("bbands boom")

    @staticmethod
    def adx(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("adx boom")

    @staticmethod
    def cci(*_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("cci boom")

    @staticmethod
    def mfi(*_args: Any, **_kwargs: Any) -> Any:
        raise AttributeError("mfi boom")

    @staticmethod
    def tema(*_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("tema boom")

    @staticmethod
    def willr(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("willr boom")

    @staticmethod
    def psar(*_args: Any, **_kwargs: Any) -> Any:
        raise AttributeError("psar boom")

    @staticmethod
    def ichimoku(*_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("ichimoku boom")

    @staticmethod
    def stochrsi(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("stochrsi boom")

    @staticmethod
    def sma(*_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("sma boom")


def _patch_ta(monkeypatch: pytest.MonkeyPatch, ta: type[Any]) -> None:
    monkeypatch.setattr(prepare_mod.importlib, "import_module", lambda name: ta)


def test_prepare_indicators_fills_missing_ohlv_and_intraday_schema(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_ta(monkeypatch, _CoreTA)
    frame = pd.DataFrame(
        {"C": [10.0, 10.5, 11.0, 11.5]},
        index=pd.date_range("2026-01-06 09:30", periods=4, freq="min", tz="UTC"),
    )

    result = prepare_mod.prepare_indicators(frame, freq="minute")

    assert isinstance(result.index, pd.DatetimeIndex)
    assert "timestamp" in result.columns
    assert result[["open", "high", "low", "close"]].eq(result["close"], axis=0).all().all()
    assert result["volume"].eq(1.0).all()
    assert {"vwap", "rsi", "atr", "mfi_14", "macd", "stochrsi", "lag_close_1"} <= set(
        result.columns
    )
    assert result["mfi_14"].isna().all()
    assert "MFI indicator unavailable" in caplog.text
    assert "MACD indicator unavailable" in caplog.text


def test_prepare_indicators_rejects_missing_close_and_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ta(monkeypatch, _CoreTA)

    with pytest.raises(KeyError, match="Column 'close'"):
        prepare_mod.prepare_indicators(
            pd.DataFrame(
                {"high": [2.0], "low": [1.0], "volume": [100.0]},
                index=pd.date_range("2026-01-06", periods=1, tz="UTC"),
            )
        )

    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    with pytest.raises(ValueError, match="Invalid date values"):
        prepare_mod.prepare_indicators(empty)


def test_prepare_indicators_populates_full_daily_schema_and_sorted_dow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ta(monkeypatch, _FullTA)

    result = prepare_mod.prepare_indicators(_market_frame(64, unsorted=True), freq="daily")

    assert result.index.is_monotonic_increasing
    assert result["dow"].tolist() == [float(ts.dayofweek) for ts in result.index]
    expected = {
        "kc_lower",
        "kc_mid",
        "kc_upper",
        "macd",
        "macds",
        "bb_upper",
        "bb_lower",
        "bb_percent",
        "adx",
        "dmp",
        "dmn",
        "cci",
        "mfi_14",
        "tema",
        "willr",
        "psar_long",
        "psar_short",
        "ichimoku_conv",
        "ichimoku_base",
        "stochrsi",
        "sma_50",
        "sma_200",
    }
    assert expected <= set(result.columns)
    assert result[list(expected)].notna().all().all()


def test_prepare_indicators_logs_indicator_failures_and_keeps_base_features(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_ta(monkeypatch, _ExplodingTA)

    result = prepare_mod.prepare_indicators(_market_frame(), freq="daily")

    assert result["vwap"].notna().all()
    assert result["rsi"].notna().all()
    assert result["atr"].notna().all()
    assert result[["macd", "adx", "mfi_14", "tema", "sma_50"]].isna().all().all()
    for message in (
        "KC indicator failed",
        "MACD calculation failed",
        "Bollinger Bands failed",
        "ADX calculation failed",
        "CCI calculation failed",
        "MFI calculation failed",
        "TEMA calculation failed",
        "Williams %R calculation failed",
        "PSAR calculation failed",
        "Ichimoku calculation failed",
        "StochRSI calculation failed",
        "SMA calculation failed",
    ):
        assert message in caplog.text


def test_prepare_indicators_logs_multi_timeframe_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_ta(monkeypatch, _FullTA)

    def raise_pct_change(self: Any, *_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("pct_change unavailable")

    monkeypatch.setattr(pd.Series, "pct_change", raise_pct_change)

    result = prepare_mod.prepare_indicators(_market_frame(), freq="minute")

    assert "Multi-timeframe features failed" in caplog.text
    assert result["ret_5m"].isna().all()
    assert result["lag_close_1"].isna().all()


def test_prepare_indicators_does_not_backfill_lagged_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ta(monkeypatch, _FullTA)

    result = prepare_mod.prepare_indicators(_market_frame(), freq="minute")

    assert pd.isna(result["lag_close_1"].iloc[0])
    assert pd.isna(result["ret_5m"].iloc[0])


def test_prepare_indicators_intraday_ffill_stays_within_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SessionGapTA(_FullTA):
        @staticmethod
        def macd(close: Any, fast: int = 12, slow: int = 26, signal: int = 9) -> Any:
            macd = close * 0.01
            signal_values = close * 0.02
            macd.iloc[3] = np.nan
            signal_values.iloc[3] = np.nan
            return pd.DataFrame(
                {"MACD_12_26_9": macd, "MACDs_12_26_9": signal_values},
                index=close.index,
            )

    _patch_ta(monkeypatch, SessionGapTA)
    index = pd.DatetimeIndex(
        [
            "2026-01-05T14:30:00Z",
            "2026-01-05T14:31:00Z",
            "2026-01-05T14:32:00Z",
            "2026-01-06T14:30:00Z",
            "2026-01-06T14:31:00Z",
            "2026-01-06T14:32:00Z",
        ],
        tz="UTC",
    )
    close = pd.Series(np.linspace(100.0, 105.0, len(index)), index=index)
    frame = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )

    result = prepare_mod.prepare_indicators(frame, freq="minute")

    assert isinstance(result.index, pd.DatetimeIndex)
    assert "timestamp" in result.columns
    assert result["timestamp"].tolist() == list(result.index)
    assert pd.notna(result["macd"].iloc[2])
    assert pd.isna(result["macd"].iloc[3])
