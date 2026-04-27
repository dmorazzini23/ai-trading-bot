from __future__ import annotations

import json

import pytest

from ai_trading.core import hyperparams_schema as hp


def test_hyperparams_load_save_validate_and_version_paths(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    valid_path = tmp_path / "hyperparams.json"
    valid_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "buy_threshold": 0.7,
                "rebalance_frequency": "weekly",
            }
        ),
        encoding="utf-8",
    )

    loaded = hp.load_hyperparams(str(valid_path))
    assert loaded.buy_threshold == 0.7
    assert loaded.rebalance_frequency == "weekly"

    report = hp.validate_hyperparams_file(str(valid_path))
    assert report["file_exists"] is True
    assert report["valid_json"] is True
    assert report["valid_schema"] is True
    assert report["version_compatible"] is True

    mismatch_path = tmp_path / "future.json"
    mismatch_path.write_text(json.dumps({"schema_version": "2.0.0"}), encoding="utf-8")
    assert hp.load_hyperparams(str(mismatch_path)).schema_version == "2.0.0"
    assert hp.validate_hyperparams_file(str(mismatch_path))["version_compatible"] is False

    invalid_json = tmp_path / "bad.json"
    invalid_json.write_text("{bad", encoding="utf-8")
    assert hp.load_hyperparams(str(invalid_json)).schema_version == hp.HYPERPARAMS_SCHEMA_VERSION
    assert "Invalid JSON" in hp.validate_hyperparams_file(str(invalid_json))["errors"][0]

    invalid_schema = tmp_path / "schema_bad.json"
    invalid_schema.write_text(json.dumps({"rebalance_frequency": "yearly"}), encoding="utf-8")
    assert hp.load_hyperparams(str(invalid_schema)).rebalance_frequency == "daily"
    assert "Schema validation failed" in hp.validate_hyperparams_file(str(invalid_schema))["errors"][0]

    missing = tmp_path / "missing.json"
    assert hp.load_hyperparams(str(missing)).schema_version == hp.HYPERPARAMS_SCHEMA_VERSION
    assert hp.validate_hyperparams_file(str(missing))["warnings"] == [f"File not found: {missing}"]

    monkeypatch.setattr(hp, "has_default", lambda name: name == "hyperparams.json")
    monkeypatch.setattr(hp, "load_default_json", lambda _name: {"buy_threshold": 0.42})
    assert hp.load_hyperparams("hyperparams.json").buy_threshold == 0.42

    monkeypatch.setattr(hp, "load_default_json", lambda _name: "bad")
    assert hp.load_hyperparams("hyperparams.json").buy_threshold == 0.5

    save_path = tmp_path / "nested" / "saved.json"
    params = hp.get_default_hyperparams()
    assert hp.save_hyperparams(params, str(save_path)) is True
    saved = json.loads(save_path.read_text(encoding="utf-8"))
    assert saved["created_at"]
    assert saved["updated_at"]

    monkeypatch.setattr(hp.json, "dump", lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("bad dump")))
    assert hp.save_hyperparams(params, str(tmp_path / "fail.json")) is False


def test_hyperparams_schema_validators_and_compatibility() -> None:
    with pytest.raises(ValueError, match="rebalance_frequency"):
        hp.HyperparametersSchema(rebalance_frequency="yearly")
    with pytest.raises(ValueError, match="schema_version"):
        hp.HyperparametersSchema(schema_version="1")

    assert hp._is_compatible_version("1.0.0", "1.1.0") is True  # noqa: SLF001
    assert hp._is_compatible_version("1.2.0", "1.1.0") is False  # noqa: SLF001
    assert hp._is_compatible_version("bad", "1.1.0") is False  # noqa: SLF001
