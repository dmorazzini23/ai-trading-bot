import csv
import logging
import os
import uuid
from datetime import datetime, timezone

from validate_env import settings
import json
import config

TRADE_LOG_FILE = config.TRADE_LOG_FILE

logger = logging.getLogger(__name__)
_fields = [
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


def log_trade(symbol, qty, side, fill_price, timestamp, extra_info=None, exposure=None):
    """Persist a trade event to ``TRADE_LOG_FILE`` and log a summary."""
    # AI-AGENT-REF: record exposure and intent
    logger.info(
        "Trade Log | symbol=%s, qty=%s, side=%s, fill_price=%.2f, exposure=%s, timestamp=%s",
        symbol,
        qty,
        side,
        fill_price,
        f"{exposure:.4f}" if exposure is not None else "n/a",
        timestamp,
    )

    os.makedirs(os.path.dirname(TRADE_LOG_FILE) or ".", exist_ok=True)
    exists = os.path.exists(TRADE_LOG_FILE)
    try:
        with open(TRADE_LOG_FILE, "a", newline="") as f:
            if not exists:
                try:
                    os.chmod(TRADE_LOG_FILE, 0o664)
                except OSError:
                    pass
            writer = csv.DictWriter(
                f,
                fieldnames=_fields,
                quoting=csv.QUOTE_MINIMAL,
            )
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
                    "exposure": exposure if exposure is not None else "",
                    "mode": (extra_info or ""),
                    "result": "",
                }
            )
    except PermissionError as exc:  # pragma: no cover - permission errors
        logger.error(
            "ERROR [audit] permission denied writing %s: %s", TRADE_LOG_FILE, exc
        )
    except Exception as exc:  # pragma: no cover - other I/O errors
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
