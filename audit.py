import csv
import logging
import os
import uuid
from datetime import datetime, timezone

from validate_env import settings
import json
import config

TRADE_LOG_FILE = settings.TRADE_LOG_FILE

logger = logging.getLogger(__name__)
_fields = ["id", "timestamp", "symbol", "side", "qty", "price", "mode", "result"]


def log_trade(symbol, qty, side, fill_price, timestamp, extra_info=None):
    """Persist a trade event to ``TRADE_LOG_FILE`` and log a summary."""
    # AI-AGENT-REF: new signature for flexible logging
    logger.info(
        f"Trade Log | {symbol=} {qty=} {side=} {fill_price=} {timestamp=} {extra_info=}"
    )

    os.makedirs(os.path.dirname(TRADE_LOG_FILE) or ".", exist_ok=True)
    exists = os.path.exists(TRADE_LOG_FILE)
    try:
        with open(TRADE_LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_fields)
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": fill_price,
                    "mode": (extra_info or ""),
                    "result": "",
                }
            )
    except Exception as exc:  # pragma: no cover - I/O errors
        logger.error("Failed to record trade: %s", exc)


def log_json_audit(details: dict) -> None:
    """Write detailed trade audit record to JSON file."""
    # AI-AGENT-REF: compliance style audit logging
    os.makedirs(config.TRADE_AUDIT_DIR, exist_ok=True)
    order_id = details.get("client_order_id") or str(uuid.uuid4())
    fname = os.path.join(config.TRADE_AUDIT_DIR, f"{order_id}.json")
    try:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, default=str)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed JSON audit log %s: %s", fname, exc)

