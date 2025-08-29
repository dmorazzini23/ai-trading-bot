import pytest

pd = pytest.importorskip("pandas")

from ai_trading.market.symbol_map import to_alpaca_symbol


class DummyAPIError(Exception):
    pass


def _fake_get_stock_bars(symbol: str):
    if symbol == "BRK-B":
        raise DummyAPIError("invalid symbol")
    return pd.DataFrame({"open": [1.0], "close": [1.1]})


def test_brkb_conversion_avoids_api_error():
    with pytest.raises(DummyAPIError):
        _fake_get_stock_bars("BRK-B")
    symbol = to_alpaca_symbol("BRK-B")
    df = _fake_get_stock_bars(symbol)
    assert not df.empty
