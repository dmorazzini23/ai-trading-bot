import logging
import os
from logger_rotator import get_rotating_handler

LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE_NAME = os.getenv("LOG_FILE_NAME", "ai_trading_bot.log")
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", 10_000_000))  # 10MB default
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 5))

def setup_logging():
    """Configure root logger with rotating file handler and console output."""

    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Rotating file handler
    file_handler = get_rotating_handler(LOG_PATH, max_bytes=LOG_MAX_BYTES, backup_count=LOG_BACKUP_COUNT)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (optional, recommended for dev/debug)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info("Logging initialized. Writing logs to %s", LOG_PATH)