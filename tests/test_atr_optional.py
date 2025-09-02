import pytest
pd = pytest.importorskip("pandas")

from ai_trading.features.indicators import compute_atr, ensure_columns


def test_compute_atr_skips_when_columns_missing():
    df = pd.DataFrame({"close": [1, 2, 3, 4]})
    out = compute_atr(df.copy())
    assert "atr" not in out.columns
    out = ensure_columns(out, ["macd"], symbol="TEST")
    assert "atr" not in out.columns


def test_compute_atr_when_available():
    df = pd.DataFrame(
        {
            "high": [2, 3, 4, 5],
            "low": [1, 1, 2, 2],
            "close": [1.5, 2.5, 3.5, 4.5],
        }
    )
    out = compute_atr(df.copy())
    assert "atr" in out.columns
    assert out["atr"].notna().any()
