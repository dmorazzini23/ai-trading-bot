from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Sequence
from zoneinfo import ZoneInfo

from ai_trading.utils.lazy_imports import load_pandas

pd = load_pandas()
logger = logging.getLogger(__name__)


def _ensure_data_fresh(
    df: pd.DataFrame | None,
    max_age_seconds: int,
    *,
    symbol: str | None = None,
    now: datetime | None = None,
    tz: str | ZoneInfo | None = None,
) -> None:
    """Raise ``RuntimeError`` if ``df`` is stale or empty.

    Parameters
    ----------
    df:
        DataFrame containing market data. Must have a DatetimeIndex or a
        ``timestamp`` column.
    max_age_seconds:
        Maximum allowed age in seconds for the most recent data point.
    symbol:
        Optional symbol used in log messages and error context.
    now:
        Current time. Defaults to ``datetime.now`` in ``tz``.
    tz:
        Timezone for ``now``. Accepts ``ZoneInfo`` or a string identifier.
    """
    if df is None or getattr(df, "empty", True):
        raise RuntimeError("no_data")

    # Determine "now" with timezone handling
    if tz is None:
        tzinfo = UTC
    elif isinstance(tz, str):
        tzinfo = ZoneInfo(tz)
    else:
        tzinfo = tz
    raw_now = now or datetime.now(tzinfo)
    if raw_now.tzinfo is None:
        logger.debug("Naive 'now' timestamp provided; assuming %s", tzinfo)
        raw_now = raw_now.replace(tzinfo=tzinfo)
    else:
        logger.debug("Aware 'now' timestamp provided with tz %s", raw_now.tzinfo)
    now = pd.to_datetime(raw_now, utc=True)

    # Extract timestamp from index or column
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        last_ts_raw = df.index[-1]
    elif "timestamp" in df.columns and len(df) > 0:
        last_ts_raw = df["timestamp"].iloc[-1]
    else:
        raise RuntimeError("no_timestamp")

    tzinfo_raw = getattr(last_ts_raw, "tzinfo", None)
    last_ts = pd.to_datetime(last_ts_raw, utc=True)
    if tzinfo_raw is None:
        logger.debug("Naive data timestamp converted to UTC: %s", last_ts.isoformat())
    else:
        logger.debug(
            "Aware data timestamp (%s) converted to UTC: %s",
            tzinfo_raw,
            last_ts.isoformat(),
        )

    age_secs = int((now - last_ts).total_seconds())
    if age_secs > max_age_seconds:
        raise RuntimeError(f"age={age_secs}s")

    logger.debug("Data freshness OK [UTC now=%s]", now.isoformat())


def ensure_data_fresh(
    fetcher: object,
    symbols: Sequence[str],
    max_age_seconds: int,
    *,
    now: datetime | None = None,
    tz: str | ZoneInfo | None = None,
) -> None:
    """Validate freshness for each symbol using ``fetcher.get_minute_df``.

    Raises ``RuntimeError`` summarizing any stale symbols.
    """
    if tz is None:
        tzinfo = UTC
    elif isinstance(tz, str):
        tzinfo = ZoneInfo(tz)
    else:
        tzinfo = tz
    raw_now = now or datetime.now(tzinfo)
    if raw_now.tzinfo is None:
        logger.debug("Naive 'now' timestamp provided; assuming %s", tzinfo)
        raw_now = raw_now.replace(tzinfo=tzinfo)
    else:
        logger.debug("Aware 'now' timestamp provided with tz %s", raw_now.tzinfo)
    now = pd.to_datetime(raw_now, utc=True)
    start = now - pd.to_timedelta(1, unit="min")
    stale: list[str] = []

    for sym in symbols:
        try:
            df = fetcher.get_minute_df(sym, start, now)
        except Exception as e:  # noqa: BLE001 - propagate as runtime error
            stale.append(f"{sym}(error={e})")
            continue
        try:
            _ensure_data_fresh(
                df,
                max_age_seconds,
                symbol=sym,
                now=now,
                tz=tzinfo,
            )
        except RuntimeError as e:
            stale.append(f"{sym}({e})")

    if stale:
        details = ", ".join(stale)
        logger.warning(
            "Data staleness detected [UTC now=%s]: %s",
            now.isoformat(),
            details,
        )
        raise RuntimeError(f"Stale data for symbols: {details}")

    logger.debug("Data freshness OK [UTC now=%s]", now.isoformat())
