"""Fallback minute-bar retrieval for price snapshots.

Splits long ranges into smaller windows, fetches each slice using the configured backup provider, then concatenates and reindexes the result. Ensures a ``close`` column is present in the final DataFrame.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
from ai_trading.logging import get_logger


logger = get_logger(__name__)


def _slice_range(start: datetime, end: datetime, span: timedelta) -> Iterable[tuple[datetime, datetime]]:
    """Yield (start, end) tuples covering ``start``..``end`` in ``span`` chunks."""
    cur = start
    while cur < end:
        nxt = min(cur + span, end)
        yield cur, nxt
        cur = nxt


def fetch(symbol: str, start: datetime, end: datetime):
    """Fetch minute bars for ``symbol`` using the backup provider.

    Parameters
    ----------
    symbol:
        Ticker symbol to query.
    start, end:
        Range to request. If the span exceeds eight days the range is
        split to avoid provider limits.

    Returns
    -------
    pandas.DataFrame
        Concatenated minute bars containing at least a ``close`` column.
    """
    from ai_trading.data.fetch import _backup_get_bars, _ensure_pandas, ensure_datetime

    pd = _ensure_pandas()
    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)
    dfs = []
    for s, e in _slice_range(start_dt, end_dt, timedelta(days=8)):
        try:
            frame = _backup_get_bars(symbol, s, e, interval="1m")
        except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
            logger.warning(
                "PRICE_SNAPSHOT_BACKUP_SLICE_FAILED",
                extra={
                    "symbol": symbol,
                    "start": s.isoformat(),
                    "end": e.isoformat(),
                    "reason": type(exc).__name__,
                    "error": str(exc),
                },
            )
            continue
        if frame is not None:
            dfs.append(frame)
    if pd is None:
        return [] if not dfs else dfs[0]  # pragma: no cover - pandas missing
    dfs = [frame for frame in dfs if frame is not None]
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    df = df.reset_index(drop=True)
    if "close" not in df.columns:
        lower = {c.lower(): c for c in df.columns}
        if "close" in lower:
            df = df.rename(columns={lower["close"]: "close"})
    return df


__all__ = ["fetch"]
