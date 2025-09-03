import importlib
import logging
import sys
import types

import joblib
import pytest


def reload_bot_engine():
    return importlib.reload(__import__('ai_trading.core.bot_engine', fromlist=['dummy']))


def test_load_model_from_path(monkeypatch, tmp_path):
    be = reload_bot_engine()
    mpath = tmp_path / "m.pkl"
    joblib.dump({"ok": True}, mpath)
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", str(mpath))
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    mdl = be._load_required_model()
    assert isinstance(mdl, dict) and mdl["ok"] is True


def test_load_model_logs_activation(monkeypatch, tmp_path):
    be = reload_bot_engine()
    mpath = tmp_path / "m.pkl"
    joblib.dump({"ok": True}, mpath)
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", str(mpath))
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)

    class ListHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
            """Store log records for assertions."""
            self.records.append(record)

    handler = ListHandler()
    logger = logging.getLogger("ai_trading.core.bot_engine")
    logger.addHandler(handler)
    try:
        be._load_required_model()
    finally:
        logger.removeHandler(handler)

    assert any(r.levelno == logging.INFO and r.getMessage() == "MODEL_LOADED" for r in handler.records)


def test_load_model_from_module(monkeypatch, tmp_path):
    be = reload_bot_engine()
    mod = types.ModuleType("fake_model_mod")
    class Dummy:
        pass
    mod.get_model = lambda: Dummy()
    sys.modules["fake_model_mod"] = mod
    monkeypatch.delenv("AI_TRADING_MODEL_PATH", raising=False)
    monkeypatch.setenv("AI_TRADING_MODEL_MODULE", "fake_model_mod")
    mdl = be._load_required_model()
    assert isinstance(mdl, Dummy)


def test_model_missing_raises(monkeypatch):
    be = reload_bot_engine()
    monkeypatch.delenv("AI_TRADING_MODEL_PATH", raising=False)
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    with pytest.raises(RuntimeError, match="Model required but not configured"):
        be._load_required_model()


def test_model_loaded_once(monkeypatch):
    be = reload_bot_engine()
    calls = {"count": 0}
    def factory():
        calls["count"] += 1
        return object()
    mod = types.ModuleType("fake_model_once")
    mod.get_model = factory
    sys.modules["fake_model_once"] = mod
    monkeypatch.delenv("AI_TRADING_MODEL_PATH", raising=False)
    monkeypatch.setenv("AI_TRADING_MODEL_MODULE", "fake_model_once")
    mdl1 = be._load_required_model()
    mdl2 = be._load_required_model()
    assert mdl1 is mdl2
    assert calls["count"] == 1


def test_missing_model_file_fallback(monkeypatch, tmp_path, caplog):
    missing = tmp_path / "no_model.pkl"
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", str(missing))
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    caplog.set_level("WARNING")
    be = reload_bot_engine()
    assert be.USE_ML is False
    assert any(r.levelname == "WARNING" and "ML_" in r.msg for r in caplog.records)


def test_default_model_missing_no_warning(monkeypatch, caplog):
    monkeypatch.delenv("AI_TRADING_MODEL_PATH", raising=False)
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    caplog.set_level("WARNING")
    be = reload_bot_engine()
    assert be.USE_ML is False
    assert "ML_MODEL_MISSING" not in caplog.text

