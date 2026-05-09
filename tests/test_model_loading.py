import importlib
import json
import logging
import sys
import time
import types
from datetime import UTC, datetime
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


def test_governance_required_rejects_unregistered_configured_model(monkeypatch, tmp_path):
    be = reload_bot_engine()
    mpath = tmp_path / "m.pkl"
    joblib.dump({"ok": True}, mpath)
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", str(mpath))
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    monkeypatch.setenv("AI_TRADING_REQUIRE_MODEL_REGISTRY_APPROVAL", "1")
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path / "registry"))

    with pytest.raises(RuntimeError, match="registry_entry_missing"):
        be._load_required_model()


def test_governance_required_accepts_fresh_production_registry_model(monkeypatch, tmp_path):
    from ai_trading.model_registry import ModelRegistry

    be = reload_bot_engine()
    mpath = tmp_path / "m.pkl"
    joblib.dump({"ok": True}, mpath)
    registry = ModelRegistry(tmp_path / "registry")
    model_id = registry.register_model(
        {"artifact_path": str(mpath)},
        "ml_edge",
        "dict",
        metadata={"model_path": str(mpath)},
    )
    registry.update_governance_status(model_id, "production")
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", str(mpath))
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    monkeypatch.setenv("AI_TRADING_REQUIRE_MODEL_REGISTRY_APPROVAL", "1")
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path / "registry"))

    mdl = be._load_required_model()

    assert mdl == {"ok": True}


def test_governance_required_rejects_unapproved_module(monkeypatch):
    be = reload_bot_engine()
    mod = types.ModuleType("fake_model_unapproved")
    setattr(mod, "get_model", lambda: object())
    sys.modules["fake_model_unapproved"] = mod
    monkeypatch.delenv("AI_TRADING_MODEL_PATH", raising=False)
    monkeypatch.setenv("AI_TRADING_MODEL_MODULE", "fake_model_unapproved")
    monkeypatch.setenv("AI_TRADING_REQUIRE_MODEL_REGISTRY_APPROVAL", "1")
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE_APPROVED", raising=False)

    with pytest.raises(RuntimeError, match="module_requires_explicit_approval"):
        be._load_required_model()


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


def test_registry_fallback_passes_manifest_to_verified_loader(monkeypatch, tmp_path):
    configured = tmp_path / "missing.pkl"
    fallback = tmp_path / "approved.pkl"
    manifest = tmp_path / "approved.manifest.json"
    fallback.write_bytes(b"model")
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", str(configured))
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    be = reload_bot_engine()
    calls = []
    monkeypatch.setattr(
        be,
        "_resolve_registry_production_model_path",
        lambda *_args, **_kwargs: (
            str(fallback),
            {
                "model_id": "prod-123",
                "strategy": "ml_edge",
                "source": "runtime_promotion",
                "manifest_path": str(manifest),
            },
        ),
    )

    def fake_load(path, *, manifest_path=None):
        calls.append((str(path), str(manifest_path)))
        return {"approved": True}

    monkeypatch.setattr(be, "load_verified_joblib_artifact", fake_load)

    mdl = be._load_required_model()

    assert mdl == {"approved": True}
    assert calls == [(str(fallback), str(manifest))]


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


def test_missing_placeholder_symbol_requires_active_registry(monkeypatch, tmp_path):
    external = tmp_path / "ext"
    external.mkdir()

    monkeypatch.setenv("AI_TRADING_MODELS_DIR", str(external))

    import ai_trading.paths as paths
    import ai_trading.model_loader as model_loader

    importlib.reload(paths)
    ml = importlib.reload(model_loader)
    ml.ML_MODELS.clear()

    with pytest.raises(RuntimeError, match="Active registry model required"):
        ml.load_model("MISSING_PLACEHOLDER")


def test_registry_model_runtime_failure_does_not_fall_back(monkeypatch, tmp_path):
    fallback_path = tmp_path / "SPY.pkl"
    fallback_path.write_bytes(b"fallback")
    registry = types.ModuleType("ai_trading.model_registry")
    registry.get_active_model_meta = lambda _symbol: {  # type: ignore[attr-defined]
        "path": str(tmp_path / "registry-model.pkl"),
        "manifest_path": str(tmp_path / "registry-model.pkl.manifest.json"),
        "registered_at": datetime.now(UTC).isoformat(),
    }
    monkeypatch.setitem(sys.modules, "ai_trading.model_registry", registry)

    import ai_trading.model_loader as model_loader

    model_loader.ML_MODELS.clear()
    monkeypatch.setattr(model_loader, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(
        model_loader,
        "load_verified_joblib_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("CHECKSUM_MISMATCH")),
    )

    with pytest.raises(RuntimeError, match="registry model"):
        model_loader.load_model("SPY")


def test_runtime_model_loader_rejects_ungoverned_local_file(monkeypatch, tmp_path):
    import ai_trading.model_loader as model_loader

    fallback_path = tmp_path / "SPY.pkl"
    fallback_path.write_bytes(b"fallback")
    model_loader.ML_MODELS.clear()
    monkeypatch.setattr(model_loader, "MODELS_DIR", tmp_path)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.delenv("AI_TRADING_ALLOW_UNGOVERNED_MODEL_FILE_LOADING", raising=False)
    monkeypatch.delenv("AI_TRADING_OFFLINE_RESEARCH", raising=False)

    with pytest.raises(RuntimeError, match="Active registry model required"):
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


def test_train_and_save_model_writes_live_feature_contract(monkeypatch, tmp_path):
    pytest.importorskip("sklearn")
    pd = pytest.importorskip("pandas")
    import ai_trading.data.fetch as data_fetch
    import ai_trading.model_loader as model_loader

    rows = 430
    frame = pd.DataFrame(
        {
            "open": [100.0 + idx * 0.1 for idx in range(rows)],
            "high": [101.0 + idx * 0.1 for idx in range(rows)],
            "low": [99.0 + idx * 0.1 for idx in range(rows)],
            "close": [100.5 + idx * 0.1 for idx in range(rows)],
            "volume": [1_000.0 + idx for idx in range(rows)],
        }
    )
    monkeypatch.setattr(data_fetch, "get_daily_df", lambda *_args, **_kwargs: frame)

    model = model_loader.train_and_save_model("CONTRACT", tmp_path)

    assert list(model.feature_names_in_) == ["rsi", "macd", "atr", "vwap", "sma_50", "sma_200"]
    assert getattr(model, "required_bar_timeframe_") == "1Day"
    manifest_payload = json.loads(
        (tmp_path / "CONTRACT.pkl.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_payload["metadata"]["feature_columns"] == list(model.feature_names_in_)
    assert manifest_payload["metadata"]["required_bar_timeframe"] == "1Day"
