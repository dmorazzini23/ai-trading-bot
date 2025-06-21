import logging

import logger


def test_setup_logging_with_file(monkeypatch, tmp_path):
    """File handler is added when log_file is provided."""
    logger._configured = False
    fake = logging.NullHandler()

    def fake_makedirs(path, exist_ok=False):
        pass

    monkeypatch.setattr(logger.os, "makedirs", fake_makedirs)
    monkeypatch.setattr(logger, "RotatingFileHandler", lambda *a, **k: fake)

    log_file = tmp_path / "x" / "app.log"
    logger.setup_logging(log_file=str(log_file))
    root = logging.getLogger()
    assert fake in root.handlers
