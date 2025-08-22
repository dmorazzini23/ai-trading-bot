import types

import joblib

from ai_trading.core.bot_engine import _load_required_model


def test_load_model_from_path(monkeypatch, tmp_path):
    mpath = tmp_path / "m.pkl"
    joblib.dump({"ok": True}, mpath)
    monkeypatch.setenv("AI_TRADER_MODEL_PATH", str(mpath))
    monkeypatch.delenv("AI_TRADER_MODEL_MODULE", raising=False)
    mdl = _load_required_model()
    assert isinstance(mdl, dict) and mdl["ok"] is True


def test_load_model_from_module(monkeypatch, tmp_path):
    mod = types.ModuleType("fake_model_mod")
    class Dummy: pass
    mod.get_model = lambda: Dummy()
    import sys
    sys.modules["fake_model_mod"] = mod
    monkeypatch.delenv("AI_TRADER_MODEL_PATH", raising=False)
    monkeypatch.setenv("AI_TRADER_MODEL_MODULE", "fake_model_mod")
    mdl = _load_required_model()
    assert isinstance(mdl, Dummy)


def test_model_missing_raises(monkeypatch):
    monkeypatch.delenv("AI_TRADER_MODEL_PATH", raising=False)
    monkeypatch.delenv("AI_TRADER_MODEL_MODULE", raising=False)
    try:
        _ = _load_required_model()
    except RuntimeError as e:
        assert "Model required but not configured" in str(e)
    else:
        assert False, "Expected RuntimeError"
