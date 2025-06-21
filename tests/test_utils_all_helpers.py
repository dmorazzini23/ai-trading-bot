import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest  # noqa: F401

from utils import (get_close_column, get_datetime_column, get_high_column,
                   get_indicator_column, get_low_column, get_open_column,
                   get_order_column, get_return_column, get_symbol_column,
                   get_volume_column)


def make_df(cols, dtype="float64", values=None):
    """Return a small DataFrame with columns tailored for testing."""

    if values is None:
        values = [1, 2, 3]
    dct = {}
    for c in cols:
        if "date" in c.lower() or "time" in c.lower():
            dct[c] = pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC")
        elif "symbol" in c.lower() or "ticker" in c.lower():
            dct[c] = [f"SYM{i}" for i in range(len(values))]
        elif "ret" in c.lower():
            dct[c] = np.random.randn(len(values))
        elif "volume" in c.lower() or c == "v":
            dct[c] = [100, 200, 300]
        elif "open" in c.lower() or c == "o":
            dct[c] = [10, 11, 12]
        elif "high" in c.lower() or c == "h":
            dct[c] = [20, 21, 22]
        elif "low" in c.lower() or c == "l":
            dct[c] = [5, 4, 6]
        elif "close" in c.lower() or c == "c":
            dct[c] = [15, 16, 17]
        else:
            dct[c] = values
    return pd.DataFrame(dct)


def test_ohlcv_variants():
    """Validate helper functions that identify OHLCV columns."""
    for fn, names in [
        (get_open_column, ["Open", "open", "o"]),
        (get_high_column, ["High", "high", "h"]),
        (get_low_column, ["Low", "low", "l"]),
        (
            get_close_column,
            [
                "Close",
                "close",
                "c",
                "adj_close",
                "Adj Close",
                "adjclose",
                "adjusted_close",
            ],
        ),
        (get_volume_column, ["Volume", "volume", "v"]),
    ]:
        for name in names:
            assert fn(make_df([name])) == name
        try:
            res = fn(make_df(["other"]))
        except Exception:
            res = None
        assert res is None


def test_get_datetime_column_variants():
    """Ensure datetime column detection handles edge cases."""
    for name in ["Datetime", "datetime", "timestamp", "date"]:
        df = make_df([name])
        df[name] = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        assert get_datetime_column(df) == name
    # Not datetime dtype
    df = make_df(["Datetime"])
    df["Datetime"] = [1, 2, 3]
    assert get_datetime_column(df) is None
    # Not monotonic
    df = make_df(["datetime"])
    df["datetime"] = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-03"], utc=True)
    assert get_datetime_column(df) is None
    # Not timezone aware
    df = make_df(["datetime"])
    df["datetime"] = pd.date_range("2024-01-01", periods=3, freq="D")
    assert get_datetime_column(df) is None


def test_get_symbol_column_variants():
    """Check detection of symbol column names and uniqueness."""
    for name in ["symbol", "ticker", "SYMBOL"]:
        df = make_df([name])
        assert get_symbol_column(df) == name
    # Not unique
    df = make_df(["symbol"])
    df["symbol"] = ["SYM", "SYM", "SYM"]
    assert get_symbol_column(df) is None


def test_get_return_column_variants():
    """Verify return column detection with valid and null data."""
    for name in ["Return", "ret", "returns"]:
        df = make_df([name])
        assert get_return_column(df) == name
    # All null
    df = make_df(["Return"])
    df["Return"] = [np.nan, np.nan, np.nan]
    assert get_return_column(df) is None


def test_get_indicator_column():
    """Test selection of technical indicator columns."""
    df = make_df(["SMA", "ema", "RSI"])
    assert get_indicator_column(df, ["SMA", "EMA"]) == "SMA"
    assert get_indicator_column(df, ["EMA", "ema"]) == "ema"
    assert get_indicator_column(df, ["MACD", "ADX"]) is None


def test_get_order_column():
    """Confirm order identifier columns are correctly resolved."""
    df = make_df(["OrderID", "TradeID"])
    assert get_order_column(df, "OrderID") == "OrderID"
    assert get_order_column(df, "TradeID") == "TradeID"
    df = make_df(["orderid"])
    assert get_order_column(df, "OrderID") == "orderid"
    df = make_df(["tradeid"])
    assert get_order_column(df, "TradeID") == "tradeid"
    # All null
    df = make_df(["OrderID"])
    df["OrderID"] = [None, None, None]
    assert get_order_column(df, "OrderID") is None
