import logging
from logging.handlers import RotatingFileHandler
import os
import sys


def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')

    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    log_file = os.path.join(os.path.dirname(__file__), "logs", "bot.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    return setup_logger(name, log_file)


def log_uncaught_exceptions(ex_cls, ex, tb):
    logger = logging.getLogger()
    logger.critical("Uncaught exception", exc_info=(ex_cls, ex, tb))


sys.excepthook = log_uncaught_exceptions
