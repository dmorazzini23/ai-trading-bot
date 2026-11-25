"""Release identity must fail closed on untested code, config or schema."""

from __future__ import annotations

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


def test_release_identity_matches_exact_code_config_and_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_url = _sqlite_schema(tmp_path / "oms.db", "rev-current")
    commit = "a" * 40
    monkeypatch.setattr(release_identity, "_git_identity", lambda _root: (commit, False))
    monkeypatch.setattr(release_identity, "_schema_head", lambda _ini: "rev-current")
    monkeypatch.setattr(release_identity, "_schema_path_contains", lambda _ini, _rev: True)
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
    }
    kwargs = {
        "cfg": SimpleNamespace(), "repo_root": tmp_path,
        "alembic_ini": tmp_path / "alembic.ini",
        "database_url": database_url,
    }

    passed = release_identity.verify_release_identity(spec, **kwargs)
    assert passed["status"] == "pass"
    assert set(passed["checks"]) == {
        "spec_version", "spec_fields", "tested_commit", "clean_checkout",
        "config_hash", "profile_hash", "schema_revision",
    }
    assert passed["observed"]["model_identity"] == "absent"

    with sqlite3.connect(tmp_path / "oms.db") as connection:
        connection.execute("UPDATE alembic_version SET version_num = 'rev-previous'")
    assert release_identity.verify_release_identity(
        spec, **kwargs, pre_migration=True
    )["status"] == "pass"
    assert release_identity.verify_release_identity(spec, **kwargs)["status"] == "blocked"
    with sqlite3.connect(tmp_path / "oms.db") as connection:
        connection.execute("UPDATE alembic_version SET version_num = 'rev-current'")

    monkeypatch.setattr(release_identity, "_git_identity", lambda _root: (commit, True))
    dirty = release_identity.verify_release_identity(spec, **kwargs)
    assert dirty["checks"]["clean_checkout"] is False

    monkeypatch.setattr(release_identity, "_git_identity", lambda _root: (commit, False))
    monkeypatch.setattr(release_identity, "_schema_head", lambda _ini: "rev-new")
    stale_schema = release_identity.verify_release_identity(spec, **kwargs)
    assert stale_schema["checks"]["schema_revision"] is False

    monkeypatch.setattr(release_identity, "_schema_head", lambda _ini: "rev-current")
    unexpected_field = release_identity.verify_release_identity(
        {**spec, "accidental_secret": "do-not-persist"}, **kwargs
    )
    assert unexpected_field["checks"]["spec_fields"] is False
    assert "do-not-persist" not in json.dumps(unexpected_field)


def test_release_identity_accepts_spec_written_with_legacy_model_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live spec still has ``model_artifact``; only the SHA changes at deploy."""

    database_url = _sqlite_schema(tmp_path / "oms.db", "rev-current")
    commit = "c" * 40
    monkeypatch.setattr(release_identity, "_git_identity", lambda _root: (commit, False))
    monkeypatch.setattr(release_identity, "_schema_head", lambda _ini: "rev-current")
    monkeypatch.setattr(
        release_identity, "build_run_manifest",
        lambda _cfg: {
            "mode": "paper", "resolved_config_hash": "config-1",
            "launch_profile_hash": "profile-1",
        },
    )
    legacy_spec = {
        "schema_version": 1, "tested_commit_sha": commit,
        "resolved_config_hash": "config-1", "launch_profile_hash": "profile-1",
        "schema_revision": "rev-current",
        # No file exists for this pin: the model is no longer checked.
        "model_artifact": {"path": "trained_model.pkl", "sha256": "f" * 64},
    }
    result = release_identity.verify_release_identity(
        legacy_spec, cfg=SimpleNamespace(), repo_root=tmp_path,
        alembic_ini=tmp_path / "alembic.ini", database_url=database_url,
    )
    assert result["status"] == "pass"
    assert "model_artifact" not in result["checks"]
    assert "model_path_matches_runtime" not in result["checks"]
    assert result["observed"]["model_identity"] == "legacy_field_ignored"
    assert "f" * 64 not in json.dumps(result)


def test_release_identity_rejects_missing_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_db = tmp_path / "missing.db"
    assert release_identity._applied_schema_revision(f"sqlite:///{missing_db}") is None
    assert not missing_db.exists()

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
        "schema_revision": "rev",
    }
    result = release_identity.verify_release_identity(
        spec, cfg=SimpleNamespace(), repo_root=tmp_path,
        alembic_ini=tmp_path / "alembic.ini",
        database_url=f"sqlite:///{missing_db}",
    )
    assert result["status"] == "blocked"
    assert result["checks"]["schema_revision"] is False


def test_pre_migration_requires_known_revision_in_checkout_history() -> None:
    alembic_ini = Path(__file__).resolve().parents[2] / "alembic.ini"
    assert release_identity._schema_path_contains(alembic_ini, "20260414_0001")
    assert release_identity._schema_path_contains(alembic_ini, "20260506_0001")
    assert not release_identity._schema_path_contains(alembic_ini, "unrelated_revision")


@pytest.mark.parametrize("models_root_args", [[], ["--models-root", "IGNORED"]])
def test_release_identity_cli_writes_non_secret_blocked_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, models_root_args: list[str]
) -> None:
    """``--models-root`` is optional and ignored, so old and new units both run."""

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    output = tmp_path / "report.json"
    monkeypatch.setattr(release_identity, "get_trading_config", lambda: SimpleNamespace())
    monkeypatch.setattr(release_identity, "_database_url_from_runtime", lambda: "sqlite:///missing.db")
    seen_kwargs: list[dict[str, object]] = []

    def _verify(*_args: object, **kwargs: object) -> dict[str, object]:
        seen_kwargs.append(kwargs)
        return {"status": "blocked", "checks": {"schema_revision": False}}

    monkeypatch.setattr(release_identity, "verify_release_identity", _verify)

    argv = ["--spec", str(spec_path), "--output", str(output)]
    argv += [str(tmp_path) if arg == "IGNORED" else arg for arg in models_root_args]
    assert release_identity.main(argv) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "blocked"
    assert "models_root" not in seen_kwargs[0]
    assert "configured_model_path" not in seen_kwargs[0]


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
    # The model pin is gone; the tool still accepts the flag for older units.
    assert "--models-root" not in unit


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
