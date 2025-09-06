from __future__ import annotations

import time
from typing import Callable, Any

from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.logging import get_logger
from . import EmptyBarsError, is_market_open

pd = load_pandas()
logger = get_logger(__name__)

_RETRY_COUNTS: dict[tuple[str, str], int] = {}


def _empty_df() -> Any:
    if pd is None:  # pragma: no cover - defensive
        return []
    return pd.DataFrame()


def fetch_with_retries(
    symbol: str,
    timeframe: str,
    fetch_fn: Callable[[], Any],
    retry_delays: list[float],
) -> Any:
    """Fetch data with retry handling for empty bar responses.

    Parameters
    ----------
    symbol: str
        Symbol being fetched, used for logging and retry tracking.
    timeframe: str
        Timeframe being fetched, used for logging and retry tracking.
    fetch_fn: Callable[[], Any]
        Callable returning the data or raising :class:`EmptyBarsError` when
        the response is empty.
    retry_delays: list[float]
        Sequence of delays between retries. The list is mutated in place as
        delays are consumed. When the list is empty the function will return
        an empty DataFrame without retrying.
    """
    key = (symbol, timeframe)
    attempts = 0
    while True:
        try:
            data = fetch_fn()
        except EmptyBarsError:
            attempts += 1
            _RETRY_COUNTS[key] = attempts
            if not is_market_open():
                logger.info(
                    "EMPTY_FETCH_MARKET_CLOSED",
                    extra={"symbol": symbol, "timeframe": timeframe, "attempts": attempts},
                )
                _RETRY_COUNTS.pop(key, None)
                return _empty_df()
            if not retry_delays:
                logger.warning(
                    "EMPTY_FETCH_NO_RETRIES",
                    extra={"symbol": symbol, "timeframe": timeframe, "attempts": attempts},
                )
                _RETRY_COUNTS.pop(key, None)
                return _empty_df()
            delay = retry_delays.pop(0)
            logger.debug(
                "EMPTY_FETCH_RETRY",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "attempts": attempts,
                    "remaining_retries": len(retry_delays),
                    "retry_delay": delay,
                },
            )
            time.sleep(delay)
        else:
            _RETRY_COUNTS.pop(key, None)
            return data


__all__ = ["fetch_with_retries", "_RETRY_COUNTS"]
