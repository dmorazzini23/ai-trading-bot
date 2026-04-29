from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from ai_trading.features import prepare as prepare_mod


def _frame(rows: int = 40) -> Any:
    index = pd.date_range("2026-01-01", periods=rows, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100.0, 120.0, rows), index=index)
    return pd.DataFrame(
        {
            "Open": close - 0.25,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.linspace(10_000.0, 20_000.0, rows),
        },
        index=index,
    )


def _series(values: Any, offset: float = 0.0) -> Any:
    return pd.Series(np.asarray(values, dtype=float) + offset, index=values.index)


class _RequiredOnlyTA:
    @staticmethod
    def vwap(high: Any, low: Any, close: Any, volume: Any) -> Any:
        return (high + low + close) / 3.0

    @staticmethod
    def rsi(close: Any, length: int = 14) -> Any:
        return _series(close, 1.0)

    @staticmethod
    def atr(high: Any, low: Any, close: Any, length: int = 14) -> Any:
        return (high - low).abs() + 0.5

    @staticmethod
    def mfi(high: Any, low: Any, close: Any, volume: Any, length: int = 14) -> Any:
        return _series(close, 2.0)


class _FullTA(_RequiredOnlyTA):
    @staticmethod
    def kc(high: Any, low: Any, close: Any, length: int = 20) -> Any:
        return pd.DataFrame({"lower": close - 2.0, "mid": close, "upper": close + 2.0})

    @staticmethod
    def macd(close: Any, fast: int = 12, slow: int = 26, signal: int = 9) -> Any:
        return pd.DataFrame({"MACD_12_26_9": close * 0.01, "MACDs_12_26_9": close * 0.005})

    @staticmethod
    def bbands(close: Any, length: int = 20) -> Any:
        return pd.DataFrame(
            {
                "BBU_20_2.0": close + 3.0,
                "BBL_20_2.0": close - 3.0,
                "BBP_20_2.0": pd.Series(0.5, index=close.index),
            }
        )

    @staticmethod
    def adx(high: Any, low: Any, close: Any, length: int = 14) -> Any:
        return pd.DataFrame(
            {
                "ADX_14": pd.Series(20.0, index=close.index),
                "DMP_14": pd.Series(25.0, index=close.index),
                "DMN_14": pd.Series(15.0, index=close.index),
            }
        )

    @staticmethod
    def cci(high: Any, low: Any, close: Any, length: int = 20) -> Any:
        return _series(close, -100.0)

    @staticmethod
    def tema(close: Any, length: int = 10) -> Any:
        return _series(close, 0.25)

    @staticmethod
    def willr(high: Any, low: Any, close: Any, length: int = 14) -> Any:
        return pd.Series(-20.0, index=close.index)

    @staticmethod
    def psar(high: Any, low: Any, close: Any) -> Any:
        return pd.DataFrame({"PSARl_0.02_0.2": low, "PSARs_0.02_0.2": high})

    @staticmethod
    def ichimoku(high: Any, low: Any, close: Any) -> Any:
        conv = pd.DataFrame({"conv": (high + low) / 2.0})
        base = pd.DataFrame({"base": (high + low + close) / 3.0})
        return conv, base

    @staticmethod
    def stochrsi(close: Any) -> Any:
        return pd.DataFrame({"STOCHRSIk_14_14_3_3": pd.Series(0.75, index=close.index)})

    @staticmethod
    def sma(close: Any, length: int) -> Any:
        return close.rolling(length, min_periods=1).mean()


class _ExplodingOptionalTA(_RequiredOnlyTA):
    @staticmethod
    def kc(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("kc failed")

    @staticmethod
    def macd(*_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("macd failed")

    @staticmethod
    def bbands(*_args: Any, **_kwargs: Any) -> Any:
        raise AttributeError("bb failed")

    @staticmethod
    def adx(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("adx failed")

    @staticmethod
    def cci(*_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("cci failed")

    @staticmethod
    def tema(*_args: Any, **_kwargs: Any) -> Any:
        raise AttributeError("tema failed")

    @staticmethod
    def willr(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("willr failed")

    @staticmethod
    def psar(*_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("psar failed")

    @staticmethod
    def ichimoku(*_args: Any, **_kwargs: Any) -> Any:
        raise AttributeError("ichimoku failed")

    @staticmethod
    def stochrsi(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("stochrsi failed")

    @staticmethod
    def sma(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("sma failed")


def test_prepare_indicators_normalizes_columns_and_handles_missing_optional_ta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prepare_mod.importlib,
        "import_module",
        lambda name: _RequiredOnlyTA if name == "pandas_ta" else pytest.fail(name),
    )

    result = prepare_mod.prepare_indicators(_frame(), freq="daily")

    assert {"open", "high", "low", "close", "volume"} <= set(result.columns)
    assert result["vwap"].notna().all()
    assert result["mfi_14"].notna().all()
    assert result["kc_lower"].isna().all()
    assert result["sma_50"].isna().all()


def test_prepare_indicators_full_ta_intraday_preserves_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame().rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
    monkeypatch.setattr(prepare_mod.importlib, "import_module", lambda _name: _FullTA)

    result = prepare_mod.prepare_indicators(frame, freq="minute")

    assert isinstance(result.index, pd.DatetimeIndex)
    assert "timestamp" in result.columns
    assert result["timestamp"].tolist() == list(result.index)
    assert result["kc_upper"].notna().all()
    assert result["macd"].notna().all()
    assert result["bb_percent"].notna().all()
    assert result["ichimoku_conv"].notna().all()
    assert result["stochrsi"].notna().all()


def test_prepare_indicators_optional_failures_do_not_drop_base_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prepare_mod.importlib, "import_module", lambda _name: _ExplodingOptionalTA)

    result = prepare_mod.prepare_indicators(_frame(), freq="daily")

    assert result["vwap"].notna().all()
    assert result["atr"].notna().all()
    assert result["macd"].isna().all()
    assert result["adx"].isna().all()
    assert result["tema"].isna().all()


def test_prepare_indicators_missing_close_and_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare_mod.importlib, "import_module", lambda _name: _RequiredOnlyTA)
    with pytest.raises(KeyError, match="Column 'close'"):
        prepare_mod.prepare_indicators(
            pd.DataFrame(
                {"high": [2.0], "low": [1.0], "volume": [1.0]},
                index=pd.date_range("2026-01-01", periods=1),
            )
        )

    def raise_import_error(_name: str) -> Any:
        raise ImportError("missing")

    monkeypatch.setattr(prepare_mod.importlib, "import_module", raise_import_error)
    with pytest.raises(ImportError, match="pandas_ta is required"):
        prepare_mod.prepare_indicators(_frame())
