import importlib
import logging
import sys
import time
import types
from pathlib import Path

import pytest

joblib = pytest.importorskip("joblib")


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
    setattr(mod, "get_model", lambda: Dummy())
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
    setattr(mod, "get_model", factory)
    sys.modules["fake_model_once"] = mod
    monkeypatch.delenv("AI_TRADING_MODEL_PATH", raising=False)
    monkeypatch.setenv("AI_TRADING_MODEL_MODULE", "fake_model_once")
    mdl1 = be._load_required_model()
    mdl2 = be._load_required_model()
    assert mdl1 is mdl2
    assert calls["count"] == 1


def test_model_reloads_when_artifact_changes(monkeypatch, tmp_path):
    be = reload_bot_engine()
    mpath = tmp_path / "m.pkl"
    joblib.dump({"version": 1}, mpath)
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", str(mpath))
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)

    first = be._load_required_model()
    assert first["version"] == 1

    time.sleep(0.02)
    joblib.dump({"version": 2}, mpath)
    second = be._load_required_model()
    assert second["version"] == 2


def test_missing_model_file_creates_placeholder(monkeypatch, tmp_path, caplog):
    missing = tmp_path / "no_model.pkl"
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", str(missing))
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    monkeypatch.delenv("AI_TRADING_WARN_IF_MODEL_MISSING", raising=False)
    caplog.set_level("WARNING")
    be = reload_bot_engine()
    monkeypatch.setattr(be, "_resolve_registry_production_model_path", lambda *_args, **_kwargs: None)
    assert "ML_MODEL_MISSING" not in caplog.text
    mdl = be._load_required_model()
    assert getattr(mdl, "is_disabled_model", False) is True
    assert not missing.exists()
    assert any(
        r.levelname == "WARNING" and r.getMessage() == "MODEL_RUNTIME_DISABLED"
        for r in caplog.records
    )


def test_placeholder_model_file_disables_runtime_ml(monkeypatch, tmp_path, caplog):
    mpath = tmp_path / "placeholder.pkl"
    joblib.dump({"placeholder": True}, mpath)
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", str(mpath))
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    caplog.set_level("WARNING")
    be = reload_bot_engine()
    monkeypatch.setattr(be, "_resolve_registry_production_model_path", lambda *_args, **_kwargs: None)

    mdl = be._load_required_model()

    assert getattr(mdl, "is_disabled_model", False) is True
    assert any(
        r.levelname == "WARNING" and r.getMessage() == "MODEL_RUNTIME_DISABLED"
        for r in caplog.records
    )


def test_missing_model_file_uses_registry_production_fallback(monkeypatch, tmp_path):
    configured = tmp_path / "missing.pkl"
    fallback = tmp_path / "approved.pkl"
    joblib.dump({"approved": True}, fallback)
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", str(configured))
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    be = reload_bot_engine()
    monkeypatch.setattr(
        be,
        "_resolve_registry_production_model_path",
        lambda *_args, **_kwargs: (
            str(fallback),
            {"model_id": "prod-123", "strategy": "ml_edge", "source": "registry_production"},
        ),
    )

    mdl = be._load_required_model()

    assert mdl == {"approved": True}


def test_missing_model_file_uses_runtime_promotion_registry_fallback(monkeypatch, tmp_path):
    from ai_trading.model_registry import ModelRegistry

    configured = tmp_path / "missing.pkl"
    runtime_model_path = tmp_path / "runtime" / "ml_latest.joblib"
    runtime_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"approved": "runtime"}, runtime_model_path)

    registry = ModelRegistry(tmp_path / "registry")
    model_id = registry.register_model({"stale": True}, "ml_edge", "dict")
    registry.update_governance_status(model_id, "production")
    Path(registry.model_index[model_id]["artifact_path"]).unlink()
    registry.record_runtime_promotion(
        model_id,
        model_path=runtime_model_path,
        manifest_path=runtime_model_path.with_suffix(".manifest.json"),
    )

    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path / "registry"))
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", str(configured))
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    be = reload_bot_engine()

    mdl = be._load_required_model()

    assert mdl == {"approved": "runtime"}


def test_default_model_missing_no_warning(monkeypatch, caplog):
    monkeypatch.delenv("AI_TRADING_MODEL_PATH", raising=False)
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    monkeypatch.delenv("AI_TRADING_WARN_IF_MODEL_MISSING", raising=False)
    caplog.set_level("WARNING")
    be = reload_bot_engine()
    assert be.USE_ML is False
    assert "ML_MODEL_MISSING" not in caplog.text


def test_missing_model_warns_when_flag_set(monkeypatch, tmp_path, caplog):
    missing = tmp_path / "no_model.pkl"
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", str(missing))
    monkeypatch.setenv("AI_TRADING_WARN_IF_MODEL_MISSING", "1")
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    caplog.set_level("WARNING")
    be = reload_bot_engine()
    be._refresh_model_loading_flags(log_missing=True)
    assert "ML_MODEL_MISSING" in caplog.text


def test_heuristic_fallback_marked_placeholder(monkeypatch, tmp_path):
    external = tmp_path / "ext"
    internal = tmp_path / "int"
    external.mkdir()
    internal.mkdir()

    monkeypatch.setenv("AI_TRADING_MODELS_DIR", str(external))

    import ai_trading.paths as paths
    import ai_trading.model_loader as model_loader

    importlib.reload(paths)
    ml = importlib.reload(model_loader)
    monkeypatch.setattr(ml, "INTERNAL_MODELS_DIR", internal)
    ml.ML_MODELS.clear()

    model = ml.load_model("MISSING_PLACEHOLDER")

    assert getattr(model, "is_placeholder_model", False) is True
    assert tuple(getattr(model, "classes_", ())) == (0, 1)


def test_registry_model_runtime_failure_does_not_fall_back(monkeypatch, tmp_path):
    fallback_path = tmp_path / "SPY.pkl"
    fallback_path.write_bytes(b"fallback")
    registry = types.ModuleType("ai_trading.model_registry")
    registry.get_active_model_meta = lambda _symbol: {  # type: ignore[attr-defined]
        "path": str(tmp_path / "registry-model.pkl"),
        "manifest_path": str(tmp_path / "registry-model.pkl.manifest.json"),
    }
    monkeypatch.setitem(sys.modules, "ai_trading.model_registry", registry)

    import ai_trading.model_loader as model_loader

    model_loader.ML_MODELS.clear()
    monkeypatch.setattr(model_loader, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(model_loader, "INTERNAL_MODELS_DIR", tmp_path / "internal")
    monkeypatch.setattr(
        model_loader,
        "load_verified_joblib_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("CHECKSUM_MISMATCH")),
    )

    with pytest.raises(RuntimeError, match="registry model"):
        model_loader.load_model("SPY")


def test_train_and_save_model_rejects_real_bars_without_labels(monkeypatch, tmp_path):
    pytest.importorskip("sklearn")
    pd = pytest.importorskip("pandas")
    import ai_trading.data.fetch as data_fetch
    import ai_trading.model_loader as model_loader

    monkeypatch.setattr(
        data_fetch,
        "get_daily_df",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "close": [100.0 + idx for idx in range(21)],
                "volume": [1_000.0 + idx for idx in range(21)],
            }
        ),
    )

    with pytest.raises(RuntimeError, match="No labeled training rows"):
        model_loader.train_and_save_model("TINY", tmp_path)

    assert not (tmp_path / "TINY.pkl").exists()
