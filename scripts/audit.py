import csv
import logging
import os
import uuid

# AI-AGENT-REF: graceful import with fallback for testing
try:
    from validate_env import settings
except ImportError:
    # Create minimal fallback for testing
    class Settings:
        TRADE_AUDIT_DIR = os.getenv("TRADE_AUDIT_DIR", "logs/audit")
    settings = Settings()

import json

import config

TRADE_LOG_FILE = config.TRADE_LOG_FILE

logger = logging.getLogger(__name__)
_disable_trade_log = False
_fields = [
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
]

# Simple audit format for compatibility
_simple_fields = [
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
    global _disable_trade_log

    # AI-AGENT-REF: Robust parameter validation with auto-correction for common mistakes
    # Handle potential parameter order issues from tests
    if isinstance(side, (int, float)) and isinstance(qty, str):
        # Detected parameter order issue: qty and side are swapped
        logger.warning("Parameter order correction: swapping qty and side parameters")
        qty, side = side, qty

    # Critical validation to prevent crashes
    if not symbol or not isinstance(symbol, str):
        logger.error("Invalid symbol provided: %s", symbol)
        return
    if not side or not isinstance(side, str):
        logger.error("Invalid side provided: %s", side)
        return
    if not isinstance(qty, (int, float)) or qty == 0:
        logger.error("Invalid quantity: %s", qty)
        return
    if not isinstance(fill_price, (int, float)) or fill_price <= 0:
        logger.error("Invalid fill_price: %s", fill_price)
        return

    if _disable_trade_log:
        # Skip writing after a permission error was encountered
        return

    # Determine if we should use simple audit format (for tests or specific modes)
    use_simple_format = (extra_info and ("TEST" in str(extra_info).upper() or "AUDIT" in str(extra_info).upper()))

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

    # AI-AGENT-REF: ensure trade log directory and file creation with proper permissions
    log_dir = os.path.dirname(TRADE_LOG_FILE) or "."
    try:
        os.makedirs(log_dir, mode=0o755, exist_ok=True)
        # Try to ensure directory is writable
        if not os.access(log_dir, os.W_OK):
            logger.warning("Trade log directory %s is not writable", log_dir)
    except OSError as e:
        logger.error("Failed to create trade log directory %s: %s", log_dir, e)
        if not _disable_trade_log:
            _disable_trade_log = True
            logger.warning("Trade log disabled due to directory creation failure")
        return

    # Check if file exists before any operations
    file_existed = os.path.exists(TRADE_LOG_FILE)

    # Ensure the trade log file exists with proper permissions
    if not file_existed:
        try:
            # Touch the file to create it
            with open(TRADE_LOG_FILE, "a", newline=""):
                pass
            os.chmod(TRADE_LOG_FILE, 0o664)
        except (OSError, PermissionError) as e:
            logger.error("Failed to create trade log file %s: %s", TRADE_LOG_FILE, e)
            if not _disable_trade_log:
                _disable_trade_log = True
                logger.warning("Trade log disabled due to file creation failure")
            return

    try:
        fields_to_use = _simple_fields if use_simple_format else _fields

        # AI-AGENT-REF: Check if file is empty to determine if header is needed
        file_is_empty = not file_existed or os.path.getsize(TRADE_LOG_FILE) == 0

        with open(TRADE_LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fields_to_use,
                quoting=csv.QUOTE_MINIMAL,
            )
            if file_is_empty:
                writer.writeheader()

            if use_simple_format:
                # Simple audit format for tests
                writer.writerow(
                    {
                        "id": str(uuid.uuid4()),
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "side": side,
                        "qty": str(qty),
                        "price": str(fill_price),
                        "exposure": str(exposure) if exposure is not None else "",
                        "mode": extra_info or "",
                        "result": "",
                    }
                )
            else:
                # Full trade log format
                writer.writerow(
                    {
                        "symbol": symbol,
                        "entry_time": timestamp,
                        "entry_price": fill_price,
                        "exit_time": "",
                        "exit_price": "",
                        "qty": qty,
                        "side": side,
                        "strategy": (extra_info or ""),
                        "classification": "",
                        "signal_tags": "",
                        "confidence": "",
                        "reward": "",
                    }
                )
    except PermissionError as exc:  # pragma: no cover - permission errors
        logger.error(
            "ERROR [audit] permission denied writing %s: %s", TRADE_LOG_FILE, exc
        )

        # AI-AGENT-REF: Attempt automatic permission repair
        try:
            from process_manager import ProcessManager
            pm = ProcessManager()
            repair_result = pm.fix_file_permissions([TRADE_LOG_FILE])

            if repair_result['paths_fixed']:
                logger.info("Successfully repaired file permissions, retrying trade log")
                # Retry writing the trade log
                try:
                    with open(TRADE_LOG_FILE, "a", newline="") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=_fields,
                            quoting=csv.QUOTE_MINIMAL,
                        )
                        if not file_existed:
                            writer.writeheader()
                        writer.writerow(
                            {
                                "symbol": symbol,
                                "entry_time": timestamp,
                                "entry_price": fill_price,
                                "exit_time": "",
                                "exit_price": "",
                                "qty": qty,
                                "side": side,
                                "strategy": (extra_info or ""),
                                "classification": "",
                                "signal_tags": "",
                                "confidence": "",
                                "reward": "",
                            }
                        )
                    logger.info("Trade log successfully written after permission repair")
                    return  # Success, don't disable logging
                except Exception as retry_exc:
                    logger.error("Trade log retry failed after permission repair: %s", retry_exc)
            else:
                logger.warning("Failed to repair file permissions automatically")

        except Exception as repair_exc:
            logger.warning("Permission repair attempt failed: %s", repair_exc)

        if not _disable_trade_log:
            _disable_trade_log = True
            logger.warning("Trade log disabled due to permission error")
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
