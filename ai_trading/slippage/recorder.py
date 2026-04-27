from __future__ import annotations

"""Helpers for recording slippage metrics to CSV."""

from datetime import datetime
import csv
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_trading.logging import get_logger
from ai_trading.paths import SLIPPAGE_LOG_PATH

logger = get_logger(__name__)


def _ensure_file(path: Path) -> None:
    """Ensure directory exists and file has CSV header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "symbol",
                    "side",
                    "quantity",
                    "expected_price",
                    "actual_price",
                    "slippage_bps",
                ]
            )


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
        _ensure_file(path)
    except OSError as e:  # pragma: no cover - filesystem errors
        logger.warning("SLIPPAGE_LOG_INIT_FAILED", extra={"cause": e.__class__.__name__, "detail": str(e)})
        return
    try:
        expected_price = float(expected)
        actual_price = float(actual)
        slippage_bps = ((actual_price - expected_price) / expected_price) * 10000.0
    except (TypeError, ValueError, ZeroDivisionError) as e:
        logger.warning("SLIPPAGE_LOG_VALUE_INVALID", extra={"cause": e.__class__.__name__, "detail": str(e)})
        return
    ts = datetime.now(ZoneInfo("UTC")).isoformat()
    try:
        with path.open("a", newline="") as f:
            csv.writer(f).writerow([ts, symbol, side, quantity, expected_price, actual_price, slippage_bps])
    except OSError as e:  # pragma: no cover - filesystem errors
        logger.warning("SLIPPAGE_LOG_WRITE_FAILED", extra={"cause": e.__class__.__name__, "detail": str(e)})
