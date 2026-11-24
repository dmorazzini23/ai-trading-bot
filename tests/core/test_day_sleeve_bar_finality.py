from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from ai_trading.core import bot_engine


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-07-10 14:00", periods=3, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000.0, 1100.0, 1200.0],
        },
        index=index,
    )


def test_finalized_day_bars_exclude_forming_five_minute_bar() -> None:
    frame = _frame()

    finalized = bot_engine._finalized_day_sleeve_bars(
        frame,
        now=datetime(2026, 7, 10, 14, 12, tzinfo=UTC),
        grace_seconds=2.0,
    )

    assert list(finalized.index) == list(frame.index[:2])
    assert frame.index[-1] not in finalized.index


def test_day_sleeve_opening_warmup_precedes_first_finalized_bar() -> None:
    prior_session = _frame()

    assert bot_engine._day_sleeve_waiting_for_first_finalized_bar(
        prior_session,
        now=datetime(2026, 8, 3, 13, 34, tzinfo=UTC),
        grace_seconds=2.0,
    ) is True
    assert bot_engine._day_sleeve_waiting_for_first_finalized_bar(
        prior_session,
        now=datetime(2026, 8, 3, 13, 35, 3, tzinfo=UTC),
        grace_seconds=2.0,
    ) is False


def test_day_sleeve_fetch_window_covers_sma_200_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 10, 14, 12, tzinfo=UTC)
    monkeypatch.setenv("AI_TRADING_DAY_SLEEVE_ML_LOOKBACK_DAYS", "10")

    day_start = bot_engine._netting_sleeve_fetch_start(
        sleeve_name="day",
        timeframe="5Min",
        now=now,
    )
    other_start = bot_engine._netting_sleeve_fetch_start(
        sleeve_name="swing",
        timeframe="15Min",
        now=now,
    )

    assert (now - day_start).days == 10
    assert (now - day_start).total_seconds() >= 7 * 24 * 60 * 60
    assert (now - other_start).total_seconds() == 1800 * 50


def test_day_sleeve_fetch_window_cannot_be_configured_below_seven_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 10, 14, 12, tzinfo=UTC)
    monkeypatch.setenv("AI_TRADING_DAY_SLEEVE_ML_LOOKBACK_DAYS", "1")

    start = bot_engine._netting_sleeve_fetch_start(
        sleeve_name="day",
        timeframe="5Min",
        now=now,
    )

    assert (now - start).days == 7


@pytest.mark.parametrize(
    ("sleeve_name", "timeframe", "expected"),
    [
        ("day", "5Min", True),
        ("day", "5m", True),
        ("day", "1Min", False),
        ("swing", "5Min", False),
    ],
)
def test_day_sleeve_model_path_is_narrowly_scoped(
    sleeve_name: str,
    timeframe: str,
    expected: bool,
) -> None:
    assert bot_engine._is_day_sleeve_ml_timeframe(sleeve_name, timeframe) is expected
