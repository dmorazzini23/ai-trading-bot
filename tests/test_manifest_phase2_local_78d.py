from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ai_trading.registry import manifest
from ai_trading.runtime import run_manifest


def _valid_manifest() -> dict[str, object]:
    return {
        "strategy": " edge ",
        "symbols": ["AAPL"],
        "rows": "10",
        "lookback_days": 20,
        "horizon_days": 1,
        "embargo_days": 1,
        "feature_columns": ["close"],
        "feature_hash": "A" * 64,
        "default_threshold": "0.5",
        "thresholds_by_regime": {"normal": "0.5"},
        "cost_floor_bps": 1,
        "cost_model_version": "v1",
        "data_sources": {
            "daily_source": "yahoo",
            "minute_source": "alpaca",
            "data_provenance": "unit",
            "alpaca_data_feed": "iex",
        },
        "dataset_fingerprint": "b" * 64,
        "sensitivity_sweep": {"enabled": "yes", "gate": "off"},
    }


def test_manifest_validation_error_branches() -> None:
    normalized = manifest.validate_manifest_metadata(_valid_manifest())
    assert normalized["strategy"] == "edge"
    assert normalized["feature_hash"] == "a" * 64
    assert normalized["sensitivity_sweep"]["enabled"] is True
    assert normalized["sensitivity_sweep"]["gate"] is False

    bad_cases = [
        ("manifest_metadata", "not-a-mapping"),
        ("rows", {**_valid_manifest(), "rows": 0}),
        ("sensitivity_sweep.enabled", {**_valid_manifest(), "sensitivity_sweep": {"enabled": "maybe", "gate": True}}),
        ("default_threshold", {**_valid_manifest(), "default_threshold": object()}),
        ("feature_columns", {**_valid_manifest(), "feature_columns": []}),
    ]
    for match, payload in bad_cases:
        with pytest.raises(manifest.ManifestValidationError, match=match):
            manifest.validate_manifest_metadata(payload)  # type: ignore[arg-type]


def test_run_manifest_redaction_hash_flags_and_path_fallbacks(tmp_path, monkeypatch) -> None:
    cfg = SimpleNamespace(
        execution_mode="LIVE",
        account_id="ACCT123456789",
        run_manifest_path="runtime/run_manifest.json",
        snapshot_sanitized=lambda: {"enabled": True, "disabled": False, "api_key": "already-safe"},
    )
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path / "data:ignored"))
    monkeypatch.delenv("STATE_DIRECTORY", raising=False)
    monkeypatch.setattr(run_manifest, "_git_commit_hash", lambda: "abc123")

    built = run_manifest.build_run_manifest(
        cfg,
        runtime_contract={"stubs_enabled": True},
        effective_policy_hash="policy-hash",
        effective_policy={"max_position": 0.2},
    )
    assert built["mode"] == "live"
    assert built["account_id"] == "456789"
    assert built["enabled_feature_flags"] == ["enabled"]
    assert built["git_commit_hash"] == "abc123"
    assert built["effective_policy"]["max_position"] == 0.2

    path = run_manifest.write_run_manifest(cfg)
    assert path == (tmp_path / "data" / "runtime" / "run_manifest.json").resolve()
    assert json.loads(path.read_text())["account_id"] == "456789"


def test_run_manifest_default_and_dict_redaction_fallbacks(monkeypatch) -> None:
    class _Cfg:
        execution_mode = ""
        alpaca_api_key = "PKSECRET123"

        def to_dict(self):
            return {"token": "secret", "plain": "value", "feature": True}

    monkeypatch.delenv("AI_TRADING_RUN_MANIFEST_PATH", raising=False)
    monkeypatch.setattr(run_manifest, "_git_commit_hash", lambda: None)
    built = run_manifest.build_run_manifest(_Cfg())
    assert built["mode"] == "sim"
    assert built["account_id"] == "RET123"
    assert built["enabled_feature_flags"] == ["feature"]

    cfg_payload = run_manifest._redacted_cfg(_Cfg())
    assert cfg_payload["token"] == "***"
    assert cfg_payload["plain"] == "value"
    assert run_manifest._default_manifest_path() == "runtime/run_manifest.json"
