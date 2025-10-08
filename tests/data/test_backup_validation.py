from datetime import UTC, datetime, timedelta

import pandas as pd

from ai_trading.data import fetch


def test_fallback_frame_is_usable_rejects_nan_close():
    now = datetime.now(UTC)
    start = now - timedelta(minutes=2)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start - timedelta(minutes=1), periods=2, freq="1min", tz="UTC"),
            "close": [float("nan"), float("nan")],
        }
    )

    assert fetch._fallback_frame_is_usable(frame, start, now) is False


def test_fallback_frame_is_usable_accepts_recent_bars():
    now = datetime.now(UTC)
    start = now - timedelta(minutes=2)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=3, freq="1min", tz="UTC"),
            "close": [101.0, 101.5, 102.0],
        }
    )

    assert fetch._fallback_frame_is_usable(frame, start, now) is True

