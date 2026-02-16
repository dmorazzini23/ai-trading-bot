from __future__ import annotations

import importlib
import json

import pytest

joblib = pytest.importorskip("joblib")


def _reload_bot_engine():
    return importlib.reload(__import__("ai_trading.core.bot_engine", fromlist=["dummy"]))


def test_model_verification_fails_closed_in_live(monkeypatch, tmp_path) -> None:
    be = _reload_bot_engine()
    model_path = tmp_path / "model.pkl"
    manifest_path = tmp_path / "model.manifest.json"
    joblib.dump({"ok": True}, model_path)
    manifest_path.write_text(
        json.dumps(
            {
                "model_version": "v1",
                "checksum_sha256": "deadbeef",
                "created_ts": "2026-01-01T00:00:00+00:00",
                "training_data_range": {"start": "2025-01-01", "end": "2025-01-31"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(be, "_MODEL_CACHE", None, raising=False)
    monkeypatch.setattr(be, "_is_testing_env", lambda: False)
    monkeypatch.setattr(be, "is_runtime_contract_testing_mode", lambda: False)
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("AI_TRADING_MODEL_VERIFY_CHECKSUM", "1")
    monkeypatch.setenv("AI_TRADING_MODEL_PATH", str(model_path))
    monkeypatch.setenv("AI_TRADING_MODEL_MANIFEST_PATH", str(manifest_path))
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)

    with pytest.raises(RuntimeError, match="MODEL_VERIFICATION_FAILED"):
        be._load_required_model()


def test_auto_train_policy_requires_paper_and_flag() -> None:
    be = _reload_bot_engine()
    assert be._auto_train_allowed("paper", allow_paper_train=True) is True
    assert be._auto_train_allowed("paper", allow_paper_train=False) is False
    assert be._auto_train_allowed("live", allow_paper_train=True) is False
    assert be._auto_train_allowed("sim", allow_paper_train=True) is False
