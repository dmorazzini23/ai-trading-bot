"""Provider schema normalization preserves values and rejects unusable bars."""
import pandas as pd
import pytest

from ai_trading.data import fetch


@pytest.mark.parametrize("layout", ["canonical", "abbreviated", "yahoo", "multi_field", "multi_symbol"])
def test_provider_schemas_preserve_observed_prices(layout):
    names = ["open", "high", "low", "close", "volume"]
    if layout == "abbreviated":
        names = ["o", "h", "l", "c", "v"]
    elif layout == "yahoo":
        names = ["regular_market_open", "regular_market_day_high", "regular_market_day_low", "regular_market_price", "regular_market_volume"]
    frame = pd.DataFrame([[100, 103, 99, 102, 25]], columns=names,
                         index=pd.DatetimeIndex(["2026-08-03T14:30:00Z"]))
    if layout.startswith("multi"):
        frame.columns = pd.MultiIndex.from_tuples([(n,"AAPL") if layout == "multi_field" else ("AAPL",n) for n in names])
    result = fetch._flatten_and_normalize_ohlcv(frame, symbol="AAPL", timeframe="1Min")
    assert result is not None
    assert result[["open", "high", "low", "close", "volume"]].iloc[0].tolist() == [100,103,99,102,25]


@pytest.mark.parametrize("close_name", ["latest_price", "market_price", "official_price", "ending_price", "final_value", "last_value"])
def test_alternate_close_fields_preserve_price(close_name):
    frame = pd.DataFrame({"open":[100],"high":[103],"low":[99],close_name:[102],"volume":[25]},
                         index=pd.DatetimeIndex(["2026-08-03T14:30:00Z"]))
    result = fetch._flatten_and_normalize_ohlcv(frame, symbol="AAPL", timeframe="1Min")
    assert result is not None and result.close.iloc[0] == 102


def test_unusable_prices_do_not_become_tradable_bars():
    frame = pd.DataFrame({key:["bad"] for key in ["open","high","low","close","volume"]},
                         index=pd.DatetimeIndex(["2026-08-03T14:30:00Z"]))
    with pytest.raises(fetch.DataFetchError, match="close_column_all_nan"):
        fetch._flatten_and_normalize_ohlcv(frame, symbol="AAPL", timeframe="1Min")
