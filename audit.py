import csv
import os
import uuid
from datetime import datetime, timezone
import logging

TRADE_LOG_FILE = os.getenv("TRADE_LOG_FILE", "trades.csv")

logger = logging.getLogger(__name__)
_fields = ["id", "timestamp", "symbol", "side", "qty", "price", "mode", "result"]


def log_trade(symbol: str, side: str, qty: int, price: float, result: str, mode: str) -> None:
    """Append trade details to the trade log CSV."""
    row = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "mode": mode,
        "result": result,
    }
    try:
        write_header = not os.path.exists(TRADE_LOG_FILE)
        with open(TRADE_LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:  # pragma: no cover - I/O issues
        logger.error("Failed to log trade: %s", exc)

