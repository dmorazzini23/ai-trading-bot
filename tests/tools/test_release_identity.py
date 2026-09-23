"""Release identity must fail closed on untested code, schema or model bytes."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_trading.tools import release_identity


def _sqlite_schema(path: Path, revision: str) -> str:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE alembic_version (version_num TEXT NOT NULL)")
        connection.execute("INSERT INTO alembic_version VALUES (?)", (revision,))
    return f"sqlite:///{path}"


def test_release_identity_matches_exact_code_config_schema_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()
    model_path = model_root / "candidate.bin"
    model_path.write_bytes(b"qualified-artifact-bytes")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    database_url = _sqlite_schema(tmp_path / "oms.db", "rev-current")
    commit = "a" * 40
    monkeypatch.setattr(release_identity, "_git_identity", lambda _root: (commit, False))
    monkeypatch.setattr(release_identity, "_schema_head", lambda _ini: "rev-current")
    monkeypatch.setattr(
        release_identity, "build_run_manifest",
        lambda _cfg: {
            "mode": "paper", "resolved_config_hash": "config-1",
            "launch_profile_hash": "profile-1",
        },
    )
    spec = {
        "schema_version": 1, "tested_commit_sha": commit,
        "resolved_config_hash": "config-1", "launch_profile_hash": "profile-1",
        "schema_revision": "rev-current",
        "model_artifact": {"path": "candidate.bin", "sha256": model_hash},
    }
    kwargs = {
        "cfg": SimpleNamespace(), "repo_root": tmp_path,
        "alembic_ini": tmp_path / "alembic.ini",
        "database_url": database_url, "models_root": model_root,
        "configured_model_path": str(model_path),
    }

    assert release_identity.verify_release_identity(spec, **kwargs)["status"] == "pass"

    with sqlite3.connect(tmp_path / "oms.db") as connection:
        connection.execute("UPDATE alembic_version SET version_num = 'rev-previous'")
    assert release_identity.verify_release_identity(
        spec, **kwargs, pre_migration=True
    )["status"] == "pass"
    assert release_identity.verify_release_identity(spec, **kwargs)["status"] == "blocked"
    with sqlite3.connect(tmp_path / "oms.db") as connection:
        connection.execute("UPDATE alembic_version SET version_num = 'rev-current'")

    model_path.write_bytes(b"different-artifact-bytes")
    changed_model = release_identity.verify_release_identity(spec, **kwargs)
    assert changed_model["status"] == "blocked"
    assert changed_model["checks"]["model_artifact"] is False

    model_path.write_bytes(b"qualified-artifact-bytes")
    monkeypatch.setattr(release_identity, "_git_identity", lambda _root: (commit, True))
    dirty = release_identity.verify_release_identity(spec, **kwargs)
    assert dirty["checks"]["clean_checkout"] is False

    monkeypatch.setattr(release_identity, "_git_identity", lambda _root: (commit, False))
    monkeypatch.setattr(release_identity, "_schema_head", lambda _ini: "rev-new")
    stale_schema = release_identity.verify_release_identity(spec, **kwargs)
    assert stale_schema["checks"]["schema_revision"] is False

    monkeypatch.setattr(release_identity, "_schema_head", lambda _ini: "rev-current")
    wrong_runtime_model = release_identity.verify_release_identity(
        spec, **{**kwargs, "configured_model_path": str(tmp_path / "other-model.bin")}
    )
    assert wrong_runtime_model["checks"]["model_path_matches_runtime"] is False

    unexpected_field = release_identity.verify_release_identity(
        {**spec, "accidental_secret": "do-not-persist"}, **kwargs
    )
    assert unexpected_field["checks"]["spec_fields"] is False
    assert "do-not-persist" not in json.dumps(unexpected_field)


def test_release_identity_rejects_missing_database_and_model_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_db = tmp_path / "missing.db"
    assert release_identity._applied_schema_revision(f"sqlite:///{missing_db}") is None
    assert not missing_db.exists()

    model_root = tmp_path / "models"
    model_root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("not a model", encoding="utf-8")
    assert release_identity._model_sha256(model_root, "../secret.txt") is None
    assert release_identity._model_sha256(model_root, str(outside)) is None

    monkeypatch.setattr(release_identity, "_git_identity", lambda _root: ("b" * 40, False))
    monkeypatch.setattr(release_identity, "_schema_head", lambda _ini: "rev")
    monkeypatch.setattr(
        release_identity, "build_run_manifest",
        lambda _cfg: {
            "mode": "live", "resolved_config_hash": "config",
            "launch_profile_hash": "profile",
        },
    )
    spec = {
        "schema_version": 1, "tested_commit_sha": "b" * 40,
        "resolved_config_hash": "config", "launch_profile_hash": "profile",
        "schema_revision": "rev", "model_artifact": None,
    }
    result = release_identity.verify_release_identity(
        spec, cfg=SimpleNamespace(), repo_root=tmp_path,
        alembic_ini=tmp_path / "alembic.ini",
        database_url=f"sqlite:///{missing_db}", models_root=model_root,
        configured_model_path=None,
    )
    assert result["status"] == "blocked"
    assert result["checks"]["model_artifact"] is False
    assert result["checks"]["schema_revision"] is False


def test_release_identity_cli_writes_non_secret_blocked_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    output = tmp_path / "report.json"
    monkeypatch.setattr(release_identity, "get_trading_config", lambda: SimpleNamespace())
    monkeypatch.setattr(release_identity, "_database_url_from_runtime", lambda: "sqlite:///missing.db")
    monkeypatch.setattr(
        release_identity, "verify_release_identity",
        lambda *_args, **_kwargs: {
            "status": "blocked", "checks": {"model_artifact": False},
        },
    )

    assert release_identity.main([
        "--spec", str(spec_path), "--models-root", str(tmp_path),
        "--output", str(output),
    ]) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "blocked"


def test_packaged_service_checks_identity_before_and_after_migration() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    unit = (repo_root / "packaging/systemd/ai-trading.service").read_text(
        encoding="utf-8"
    )
    sync = unit.index("ExecStartPre=/home/aiuser/ai-trading-bot/scripts/sync_env_runtime.sh")
    before = unit.index("release_identity --pre-migration")
    migrate = unit.index("-m alembic upgrade head")
    after = unit.index("release_identity --spec")
    start = unit.index("ExecStart=/bin/bash")
    assert sync < before < migrate < after < start
    assert unit.count("sync_env_runtime.sh") == 1


def test_isolated_migration_rollback_restores_preupgrade_oms_state(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    database = tmp_path / "oms.db"
    snapshot = tmp_path / "preupgrade.db"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(repo_root),
        "DATABASE_URL": f"sqlite:///{database}",
    }

    def migrate(target: str) -> None:
        subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", target],
            cwd=repo_root, env=env, check=True, capture_output=True, text=True,
            timeout=30,
        )

    migrate("20260414_0001")
    with sqlite3.connect(database) as connection:
        for event_id in (1, 2):
            connection.execute(
                """INSERT INTO oms_events
                (event_uuid, intent_id, event_type, event_ts, event_source,
                 idempotency_key, sequence_no, payload_json, created_at)
                VALUES (?, 'intent-1', 'submit', '2026-09-23T00:00:00Z',
                        'rehearsal', ?, 1, '{}', '2026-09-23T00:00:00Z')""",
                (f"event-{event_id}", f"key-{event_id}"),
            )
    with sqlite3.connect(database) as source, sqlite3.connect(snapshot) as backup:
        source.backup(backup)

    migrate("head")
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "20260506_0001",
        )
        assert connection.execute(
            "SELECT COUNT(DISTINCT sequence_no) FROM oms_events"
        ).fetchone() == (2,)

    with sqlite3.connect(snapshot) as source, sqlite3.connect(database) as target:
        source.backup(target)
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "20260414_0001",
        )
        assert connection.execute(
            "SELECT COUNT(DISTINCT sequence_no) FROM oms_events"
        ).fetchone() == (1,)
