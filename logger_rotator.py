from logging.handlers import RotatingFileHandler


def get_rotating_handler(path: str, max_bytes: int = 5_000_000, backup_count: int = 3):
    """Return a configured RotatingFileHandler."""
    return RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
