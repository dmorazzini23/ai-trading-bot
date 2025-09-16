import pytest
pd = pytest.importorskip("pandas")
from ai_trading.features import compute_macd, compute_macds, ensure_columns


def test_macd_pipeline_produces_macds():
    df = pd.DataFrame(
        {
            "close": [i for i in range(1, 400)],
            "open": [1] * 399,
            "high": [2] * 399,
            "low": [0] * 399,
            "volume": [100] * 399,
        }
    )
    df = compute_macd(df)
    df = compute_macds(df)
    df = ensure_columns(df)
    assert "macds" in df.columns
    assert df["macds"].notna().any()


def test_compute_macd_ignores_string_nan(caplog):
    df = pd.DataFrame({"close": ["nan", "nan", "nan"]})

    with caplog.at_level("ERROR"):
        result = compute_macd(df.copy())

    assert not any("Error calculating EMA" in message for message in caplog.messages)
    assert list(result.columns) == ["close"]


def test_position_none_safe(monkeypatch):
    class Dummy:
        def get_position(self, symbol):
            return None

    class Ctx:
        pass

    ctx = Ctx()
    ctx.api = Dummy()
    from ai_trading.core import bot_engine as be

    assert be._current_position_qty(ctx, "SPY") == 0
