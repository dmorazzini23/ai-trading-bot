from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from ai_trading.core.data_contract import normalize_bars, validate_bars


def _sample_df(start: datetime, periods: int = 3):
    idx = pd.date_range(start=start, periods=periods, freq="1min", tz=UTC)
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
        },
        index=idx,
    )


def test_validate_bars_ok():
    df = _sample_df(datetime.now(UTC) - timedelta(minutes=2))
    result = validate_bars(df, "1Min", freshness_seconds=300, rth_only=False)
    assert result.ok


def test_validate_bars_duplicate():
    df = _sample_df(datetime.now(UTC) - timedelta(minutes=2))
    df.index = [df.index[0], df.index[0], df.index[2]]
    result = validate_bars(df, "1Min", freshness_seconds=300, rth_only=False)
    assert result.ok is False
    assert result.reason == "DUPLICATE_BARS"


def test_validate_bars_stale():
    df = _sample_df(datetime.now(UTC) - timedelta(days=5))
    result = validate_bars(df, "1Day", freshness_seconds=60, rth_only=False)
    assert result.ok is False
    assert result.reason == "STALE_BAR"


def test_validate_five_minute_freshness_uses_bar_close_timestamp():
    now = datetime(2026, 7, 17, 14, 55, 6, tzinfo=UTC)
    df = _sample_df(datetime(2026, 7, 17, 14, 35, tzinfo=UTC))
    df.index = pd.date_range(
        start=datetime(2026, 7, 17, 14, 35, tzinfo=UTC),
        periods=3,
        freq="5min",
    )

    fresh = validate_bars(
        df,
        "5Min",
        freshness_seconds=600,
        rth_only=False,
        now=now,
    )
    stale = validate_bars(
        df,
        "5Min",
        freshness_seconds=600,
        rth_only=False,
        now=datetime(2026, 7, 17, 15, 0, 1, tzinfo=UTC),
    )

    assert fresh.ok is True
    assert stale.ok is False
    assert stale.reason == "STALE_BAR"
    assert stale.detail == {
        "age_seconds": 601.0,
        "age_reference": "bar_close",
    }


def test_validate_bars_rejects_future_bar_beyond_skew():
    df = _sample_df(datetime.now(UTC) + timedelta(minutes=1))
    result = validate_bars(df, "1Min", freshness_seconds=300, rth_only=False)

    assert result.ok is False
    assert result.reason == "FUTURE_BAR"
    assert result.detail["future_skew_seconds"] > 5.0


def test_normalize_bars_lowercase():
    df = _sample_df(datetime.now(UTC) - timedelta(minutes=2))
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    normalized = normalize_bars(df, "1Min", tz=ZoneInfo("UTC"), rth_only=False)
    assert set(["open", "high", "low", "close", "volume"]).issubset(normalized.columns)


def test_normalize_bars_rejects_naive_timestamps():
    df = _sample_df(datetime(2026, 4, 27, 14, 30, tzinfo=UTC))
    df.index = df.index.tz_localize(None)

    try:
        normalize_bars(df, "1Min", tz=ZoneInfo("UTC"), rth_only=False)
    except ValueError as exc:
        assert "naive timestamps" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected naive timestamp rejection")


def test_validate_bars_rejects_naive_timestamps():
    df = _sample_df(datetime.now(UTC) - timedelta(minutes=2))
    df.index = df.index.tz_localize(None)

    result = validate_bars(df, "1Min", freshness_seconds=300, rth_only=False)

    assert result.ok is False
    assert result.reason == "NAIVE_INDEX"


def test_rth_filter_excludes_exact_close_boundary():
    df = _sample_df(datetime(2026, 4, 27, 19, 58, tzinfo=UTC), periods=3)

    normalized = normalize_bars(df, "1Min", rth_only=True)

    assert list(normalized.index) == list(df.index[:2])
