import logging
import json
import traceback
from datetime import datetime, timezone
import warnings

# AI-AGENT-REF: This module is deprecated to prevent duplicate logging
# Use ai_trading.logging module instead
warnings.warn(
    "logging_config.py is deprecated and causes duplicate logging. "
    "Use ai_trading.logging.setup_logging() instead.",
    DeprecationWarning,
    stacklevel=2
)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        try:
            log_record = {
                "ts": self.formatTime(record),
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
                "thread": record.thread,
                "process": record.process,
            }

            # Add exception info if present
            if record.exc_info:
                log_record["exception"] = traceback.format_exception(*record.exc_info)

            if hasattr(record, "bot_phase"):
                log_record["bot_phase"] = record.bot_phase

            return json.dumps(log_record, default=str)
        except Exception:
            # Fallback to simple format if JSON fails
            return f"{datetime.now(timezone.utc).isoformat()} {record.levelname} {record.name} {record.getMessage()}"


def setup_logging():
    """
    DEPRECATED: This function is now a no-op to prevent duplicate logging.
    Use ai_trading.logging.setup_logging() instead.
    """
    # AI-AGENT-REF: No-op to prevent duplicate logging configuration
    # The ai_trading.logging module should be used instead
    import logging
    logger = logging.getLogger(__name__)
    
    # Only log deprecation warning once per session
    if not hasattr(setup_logging, '_warned'):
        logger.warning(
            "logging_config.setup_logging() is deprecated and disabled. "
            "Use ai_trading.logging.setup_logging() to avoid duplicate logging."
        )
        setup_logging._warned = True
    
    # Check if any logging is already configured
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return  # Already configured by another system
    
    # If no logging configured at all, set up minimal fallback
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
