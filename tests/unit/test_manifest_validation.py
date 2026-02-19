from __future__ import annotations

import pytest

from ai_trading.registry.manifest import ManifestValidationError, validate_manifest_metadata


def _valid_payload() -> dict[str, object]:
    return {
        "strategy": "after_hours_ml_edge",
        "symbols": ["AAPL", "MSFT"],
        "rows": 1000,
        "lookback_days": 720,
        "horizon_days": 1,
        "embargo_days": 1,
        "feature_columns": ["rsi", "macd"],
        "feature_hash": "a" * 64,
        "default_threshold": 0.52,
        "thresholds_by_regime": {"uptrend": 0.4, "downtrend": 0.45},
        "cost_floor_bps": 8.5,
        "cost_model_version": "tca_floor_v1",
        "data_sources": {
            "daily_source": "yahoo",
            "minute_source": "alpaca",
            "data_provenance": "iex",
            "alpaca_data_feed": "iex",
        },
        "dataset_fingerprint": "b" * 64,
        "sensitivity_sweep": {"enabled": True, "gate": False, "summary": {"valid_scenarios": 5}},
    }


def test_validate_manifest_metadata_accepts_valid_payload() -> None:
    payload = _valid_payload()
    validated = validate_manifest_metadata(payload)
    assert validated["strategy"] == "after_hours_ml_edge"
    assert validated["rows"] == 1000
    assert validated["sensitivity_sweep"]["enabled"] is True


def test_validate_manifest_metadata_rejects_invalid_hash() -> None:
    payload = _valid_payload()
    payload["feature_hash"] = "not-a-hash"
    with pytest.raises(ManifestValidationError, match="feature_hash"):
        validate_manifest_metadata(payload)


def test_validate_manifest_metadata_rejects_missing_symbols() -> None:
    payload = _valid_payload()
    payload["symbols"] = []
    with pytest.raises(ManifestValidationError, match="symbols"):
        validate_manifest_metadata(payload)
