import importlib
import logging
import pytest
import logger
from conftest import reload_module


def test_get_rotating_handler_fallback(monkeypatch, tmp_path, caplog):
    caplog.set_level("ERROR")
    monkeypatch.setattr(logger, "RotatingFileHandler", lambda *a, **k: (_ for _ in ()).throw(OSError("fail")))
    h = logger.get_rotating_handler(str(tmp_path / "x.log"))
    assert isinstance(h, logging.StreamHandler)
    assert "Cannot open log file" in caplog.text


def test_setup_logging_idempotent(monkeypatch, tmp_path):
    mod = reload_module(logger)
    created = []

    def fake_get_rotating(path, **_):
        created.append(path)
        return logging.StreamHandler()

    monkeypatch.setattr(mod, "get_rotating_handler", fake_get_rotating)
    lg = mod.setup_logging(debug=True, log_file=str(tmp_path / "f.log"))
    assert lg.level == logging.DEBUG
    assert created
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
