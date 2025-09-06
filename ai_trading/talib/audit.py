"""Minimal trade audit for TA-Lib tests."""
from __future__ import annotations

import csv
import os
import uuid
from pathlib import Path

_LOG_DIR = Path("data")
_LOG_FILE = _LOG_DIR / "trades.csv"
_HEADERS = [
    "id",
    "timestamp",
    "symbol",
    "side",
    "qty",
    "price",
    "exposure",
    "mode",
    "result",
]


def log_trade(
    symbol: str,
    qty: int | float,
    side: str,
    fill_price: float,
    timestamp: str = "",
    extra_info: str = "",
    exposure: float | None = None,
) -> None:
    """Append a trade record to ``data/trades.csv`` ensuring directory exists.

    Creates the ``data`` directory on first use and writes a header if the file
    is new. File permissions are set to ``0o664`` when the file is created.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = _LOG_FILE.exists()
    with open(_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_HEADERS)
        if not file_exists:
            writer.writeheader()
            try:
                os.chmod(_LOG_FILE, 0o664)
            except OSError:
                pass
        writer.writerow(
            {
                "id": str(uuid.uuid4()),
                "timestamp": timestamp,
                "symbol": symbol,
                "side": side,
                "qty": str(qty),
                "price": str(fill_price),
                "exposure": "" if exposure is None else str(exposure),
                "mode": extra_info,
                "result": "",
            }
        )
