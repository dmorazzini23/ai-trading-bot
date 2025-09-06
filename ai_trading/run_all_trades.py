"""Simplified trading loop for tests.

This module provides a small façade around an injected API callable. It
ensures a trade log exists and adds a bit of behaviour used by tests:

* If the API callable raises an exception, a warning is logged and the
  symbol is skipped.
* When no symbols are provided the callable `sleep` is invoked exactly
  once and the function returns immediately.
* The trade log ``trades.csv`` is created on first use with a basic CSV
  header.
"""
from __future__ import annotations

from pathlib import Path
import csv
import logging
import time
from typing import Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

TRADE_LOG_FILE = Path("trades.csv")


def _ensure_trade_log(path: Path) -> None:
    """Create the trade log with a header if it does not yet exist."""
    if path.exists():  # pragma: no cover - early return
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "symbol",
            "entry_time",
            "entry_price",
            "exit_time",
            "exit_price",
            "qty",
            "side",
            "strategy",
            "classification",
            "signal_tags",
            "confidence",
            "reward",
        ])


def run_all_trades(
    symbols: Sequence[str],
    api: Callable[[str], object],
    *,
    trade_log: Path | str = TRADE_LOG_FILE,
    sleep: Callable[[float], object] = time.sleep,
) -> list[object]:
    """Execute trades for ``symbols`` using ``api``.

    Parameters
    ----------
    symbols:
        The collection of symbols to trade. If empty a warning is logged and
        ``sleep`` is invoked once before returning.
    api:
        Callable accepting a symbol and performing the trade. Any exception is
        converted into a warning log and the symbol is skipped.
    trade_log:
        Location of the trade log CSV. The file is created with a header on
        first use.
    sleep:
        Sleep function used when ``symbols`` is empty. Exposed for tests.
    """

    path = Path(trade_log)
    _ensure_trade_log(path)

    if not symbols:
        logger.warning("NO_SYMBOLS")
        sleep(1.0)
        return []

    results: list[object] = []
    for sym in symbols:
        try:
            res = api(sym)
        except Exception as exc:  # pragma: no cover - exercised in tests
            logger.warning("API_ERROR", exc_info=exc)
            continue
        results.append(res)
    return results


__all__ = ["run_all_trades"]
