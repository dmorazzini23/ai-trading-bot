import importlib
import sys
import types
from datetime import UTC, datetime

import pytest

joblib = pytest.importorskip("joblib")


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




