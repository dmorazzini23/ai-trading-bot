from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_migrate_secret_writes_payload_via_private_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script("migrate_secrets_to_aws_sm_under_test", "scripts/migrate_secrets_to_aws_sm.py")
    captured: dict[str, Path] = {}

    def _fake_run_json(command: list[str]) -> dict[str, object]:
        secret_index = command.index("--secret-string") + 1
        secret_arg = command[secret_index]
        assert secret_arg.startswith("file://")
        secret_path = Path(secret_arg.removeprefix("file://"))
        captured["secret_path"] = secret_path
        assert stat.S_IMODE(secret_path.stat().st_mode) == 0o600
        assert secret_path.read_text(encoding="utf-8") == '{"TOKEN":"s3cr3t"}'
        assert "s3cr3t" not in command
        return {}

    monkeypatch.setattr(script, "_secret_exists", lambda *args, **kwargs: False)
    monkeypatch.setattr(script, "_run_json", _fake_run_json)

    script._write_secret("secret-id", {"TOKEN": "s3cr3t"}, region="", profile="")

    assert not captured["secret_path"].exists()


def test_migrate_secret_redacts_secret_string_from_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script("migrate_secrets_to_aws_sm_redact_under_test", "scripts/migrate_secrets_to_aws_sm.py")

    def _fake_run(*args, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(returncode=1, stderr="denied TOPSECRET", stdout="")

    monkeypatch.setattr(script.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        script._run_json(["aws", "secretsmanager", "create-secret", "--secret-string", "TOPSECRET"])

    message = str(exc_info.value)
    assert "TOPSECRET" not in message
    assert "<redacted>" in message


def test_repair_reconciliation_never_reads_parquet_as_pickle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = _load_script(
        "repair_open_position_reconciliation_under_test",
        "scripts/repair_open_position_reconciliation.py",
    )
    path = tmp_path / "trade_history.parquet"
    path.write_text("not parquet", encoding="utf-8")

    monkeypatch.setattr(script.pd, "read_parquet", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad parquet")))
    monkeypatch.setattr(script.pd, "read_pickle", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("pickle fallback used")))

    with pytest.raises(ValueError, match="bad parquet"):
        script._load_trade_history(path)


def test_repair_reconciliation_requires_explicit_pickle_allowance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = _load_script(
        "repair_open_position_reconciliation_pickle_under_test",
        "scripts/repair_open_position_reconciliation.py",
    )
    path = tmp_path / "trade_history.pkl"
    path.write_bytes(b"pickle")
    expected = pd.DataFrame({"symbol": ["AAPL"]})
    monkeypatch.setattr(script.pd, "read_pickle", lambda *_args, **_kwargs: expected)

    with pytest.raises(ValueError, match="allow-trusted-pickle-read"):
        script._load_trade_history(path)

    frame, loaded_fmt, write_fmt = script._load_trade_history(
        path,
        allow_trusted_pickle_read=True,
    )
    assert frame is expected
    assert loaded_fmt == "pickle"
    assert write_fmt == "pickle"


def test_scalability_restore_rejects_path_traversal(tmp_path: Path) -> None:
    script = _load_script("scalability_manager_under_test", "scripts/scalability_manager.py")
    primary_dir = tmp_path / "primary" / "data"
    backup_dir = tmp_path / "backups"
    primary_dir.mkdir(parents=True)
    sentinel = primary_dir / "keep.json"
    sentinel.write_text("keep", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "backup_metadata.json").write_text("{}", encoding="utf-8")

    manager = script.DataReplicationManager(
        primary_data_dir=str(primary_dir),
        backup_dir=str(backup_dir),
    )

    assert manager.restore_backup("../outside") is False
    assert sentinel.read_text(encoding="utf-8") == "keep"
