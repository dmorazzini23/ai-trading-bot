import os
os.environ.setdefault("PYTEST_RUNNING", "1")
import importlib
import pytest

from ai_trading.core import bot_engine as be
import ai_trading.config.settings as settings


def test_model_missing_env_fails(monkeypatch):
    monkeypatch.delenv("AI_TRADER_MODEL_PATH", raising=False)
    monkeypatch.delenv("AI_TRADER_MODEL_MODULE", raising=False)
    importlib.reload(settings)
    importlib.reload(be)
    with pytest.raises(RuntimeError):
        be._load_required_model()


def test_model_bad_path_fails(monkeypatch):
    monkeypatch.setenv("AI_TRADER_MODEL_PATH", "/tmp/does_not_exist.pkl")
    monkeypatch.delenv("AI_TRADER_MODEL_MODULE", raising=False)
    importlib.reload(settings)
    importlib.reload(be)
    with pytest.raises(FileNotFoundError):
        be._load_required_model()


def test_model_module_get_model_ok(monkeypatch, tmp_path):
    mod_dir = tmp_path / "temppkg"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text(
        "def get_model():\n    class M: pass\n    return M()\n",
        encoding="utf-8",
    )
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        monkeypatch.setenv("AI_TRADER_MODEL_MODULE", "temppkg")
        monkeypatch.delenv("AI_TRADER_MODEL_PATH", raising=False)
        importlib.reload(settings)
        importlib.reload(be)
        m = be._load_required_model()
        assert m is not None
    finally:
        sys.path.remove(str(tmp_path))
