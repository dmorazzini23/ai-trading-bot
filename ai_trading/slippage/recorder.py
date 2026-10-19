from __future__ import annotations

"""Helpers for recording slippage metrics to CSV."""

from datetime import datetime
import csv
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_trading.logging import get_logger
from ai_trading.paths import SLIPPAGE_LOG_PATH

logger = get_logger(__name__)

# The one CSV schema for the slippage log. Every writer appends rows in this
# order and every header writer uses this tuple, so readers can rely on it.
SLIPPAGE_LOG_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "symbol",
    "side",
    "quantity",
    "expected_price",
    "actual_price",
    "slippage_bps",
)


def side_normalized_slippage_bps(expected_price: float, actual_price: float, side: str) -> float:
    side_token = str(side or "").strip().lower().replace("-", "_").replace(" ", "_")
    sell_side_tokens = {"sell", "sell_short", "sellshort", "short", "short_sell"}
    sign = -1.0 if side_token in sell_side_tokens or side_token.startswith("sell") else 1.0
    return sign * ((float(actual_price) - float(expected_price)) / float(expected_price)) * 10000.0


def ensure_slippage_log_header(path: Path | str) -> None:
    """Create ``path`` with the :data:`SLIPPAGE_LOG_COLUMNS` header if it is missing."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with target.open("w", newline="") as f:
            csv.writer(f).writerow(SLIPPAGE_LOG_COLUMNS)


def log_slippage(
    symbol: str,
    expected: float,
    actual: float,
    log_path: Path | str = SLIPPAGE_LOG_PATH,
    *,
    side: str = "unknown",
    quantity: float = 1.0,
) -> None:
    """Append a slippage record to CSV, creating directories and file on first use."""
    path = Path(log_path)
    try:
        ensure_slippage_log_header(path)
    except OSError as e:  # pragma: no cover - filesystem errors
        logger.warning("SLIPPAGE_LOG_INIT_FAILED", extra={"cause": e.__class__.__name__, "detail": str(e)})
        return
    try:
        expected_price = float(expected)
        actual_price = float(actual)
        slippage_bps = side_normalized_slippage_bps(expected_price, actual_price, side)
    except (TypeError, ValueError, ZeroDivisionError) as e:
        logger.warning("SLIPPAGE_LOG_VALUE_INVALID", extra={"cause": e.__class__.__name__, "detail": str(e)})
        return
    ts = datetime.now(ZoneInfo("UTC")).isoformat()
    try:
        with path.open("a", newline="") as f:
            csv.writer(f).writerow([ts, symbol, side, quantity, expected_price, actual_price, slippage_bps])
    except OSError as e:  # pragma: no cover - filesystem errors
        logger.warning("SLIPPAGE_LOG_WRITE_FAILED", extra={"cause": e.__class__.__name__, "detail": str(e)})
