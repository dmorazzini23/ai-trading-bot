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
            writer.writerow(["timestamp", "symbol", "expected", "actual", "slippage_cents"])


def log_slippage(symbol: str, expected: float, actual: float, log_path: Path | str = SLIPPAGE_LOG_PATH) -> None:
    """Append a slippage record to CSV, creating directories and file on first use."""
    path = Path(log_path)
    try:
        _ensure_file(path)
    except OSError as e:  # pragma: no cover - filesystem errors
        logger.warning("SLIPPAGE_LOG_INIT_FAILED", extra={"cause": e.__class__.__name__, "detail": str(e)})
        return
    slippage_cents = (actual - expected) * 100.0
    ts = datetime.now(ZoneInfo("UTC")).isoformat()
    try:
        with path.open("a", newline="") as f:
            csv.writer(f).writerow([ts, symbol, expected, actual, slippage_cents])
    except OSError as e:  # pragma: no cover - filesystem errors
        logger.warning("SLIPPAGE_LOG_WRITE_FAILED", extra={"cause": e.__class__.__name__, "detail": str(e)})

