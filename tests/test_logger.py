import importlib
import logging

import pytest
from conftest import reload_module

import logger


def test_get_rotating_handler_fallback(monkeypatch, tmp_path, caplog):
    caplog.set_level("ERROR")

    def raise_os(*_a, **_k):
        raise OSError("fail")

    monkeypatch.setattr(logger, "RotatingFileHandler", raise_os)
    try:
        handler = logger.get_rotating_handler(str(tmp_path / "x.log"))
    except OSError:
        pytest.fail("OSError should be handled inside get_rotating_handler")

    assert isinstance(handler, logging.StreamHandler)
    assert "Cannot open log file" in caplog.text


def test_setup_logging_idempotent(monkeypatch, tmp_path):
    mod = reload_module(logger)
    created = []

    def fake_get_rotating(path, **_):
        created.append(path)
        return logging.StreamHandler()

    monkeypatch.setattr(mod, "get_rotating_handler", fake_get_rotating)
    lg = mod.setup_logging(debug=True, log_file=str(tmp_path / "f.log"))
    assert lg.level in (logging.DEBUG, logging.INFO)
    assert created, f"No rotating handler paths created. Captured: {created}"
    created.clear()
    lg2 = mod.setup_logging(debug=False)
    assert lg2 is lg
    assert created == []


def test_get_logger():
    mod = reload_module(logger)
    root = mod.setup_logging(debug=True)
    lg = mod.get_logger("test")
    assert lg is mod._loggers["test"]
    assert len(lg.handlers) == len(root.handlers)
