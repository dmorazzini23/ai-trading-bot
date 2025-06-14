import logging
import os
from logging.handlers import RotatingFileHandler

LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "bot.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3)
        fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
