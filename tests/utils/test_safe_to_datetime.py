import pytest

pd = pytest.importorskip("pandas")

from ai_trading.utils.base import safe_to_datetime


def test_safe_to_datetime_parses_epoch_seconds():
    result = safe_to_datetime([1_700_000_000, 1_700_000_060])
    expected = pd.to_datetime([1_700_000_000, 1_700_000_060], unit="s", utc=True)
    pd.testing.assert_index_equal(result, expected)


def test_safe_to_datetime_parses_epoch_milliseconds():
    result = safe_to_datetime([1_700_000_000_000, 1_700_000_060_000])
    expected = pd.to_datetime([1_700_000_000_000, 1_700_000_060_000], unit="ms", utc=True)
    pd.testing.assert_index_equal(result, expected)


def test_safe_to_datetime_coerces_placeholders_to_nat():
    result = safe_to_datetime([
        "2024-01-01 09:30:00",
        "0",
        "",
        None,
    ])
    assert result.tz is not None
    assert result[0] == pd.Timestamp("2024-01-01 09:30:00", tz="UTC")
    assert pd.isna(result[1])
    assert pd.isna(result[2])
    assert pd.isna(result[3])
