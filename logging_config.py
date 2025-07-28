import logging
import json
import traceback
from datetime import datetime, timezone


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
        except Exception as e:
            # Fallback to simple format if JSON fails
            return f"{datetime.now(timezone.utc).isoformat()} {record.levelname} {record.name} {record.getMessage()}"


def setup_logging():
    if logging.getLogger().handlers:
        return  # Already configured
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
