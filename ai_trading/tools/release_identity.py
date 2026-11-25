"""Read-only predeployment identity check for code, config and schema.

The runtime serves no ML model, so the check no longer pins a model artifact.
A spec written before that change still carries a ``model_artifact`` entry; it
is accepted and ignored, so a deploy edits only the spec's commit SHA and, when
the resolved config changed, its config hash. The
``--models-root`` option is accepted and ignored for the same reason, while the
installed systemd unit may still pass it. Both can be removed once the live
spec and the installed unit no longer carry them.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.script.revision import RevisionError
from alembic.util.exc import CommandError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from ai_trading.config.management import get_env
from ai_trading.config.runtime import get_trading_config
from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.runtime.run_manifest import build_run_manifest

logger = get_logger(__name__)

_SPEC_FIELDS = frozenset({
    "schema_version", "tested_commit_sha", "resolved_config_hash",
    "launch_profile_hash", "schema_revision",
})
# Written by specs from before the model pin was removed; ignored if present.
_LEGACY_SPEC_FIELD = "model_artifact"


def _git_identity(repo_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo_root, check=True, capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit or None, bool(status.strip())


def _schema_head(alembic_ini: Path) -> str | None:
    try:
        config = Config(str(alembic_ini))
        config.set_main_option("script_location", str(alembic_ini.parent / "migrations"))
        heads = ScriptDirectory.from_config(config).get_heads()
    except (CommandError, OSError, RuntimeError, ValueError):
        return None
    return heads[0] if len(heads) == 1 else None


def _schema_path_contains(alembic_ini: Path, revision: str) -> bool:
    """Require the pre-migration revision to belong to this checkout's history."""

    try:
        config = Config(str(alembic_ini))
        config.set_main_option("script_location", str(alembic_ini.parent / "migrations"))
        scripts = ScriptDirectory.from_config(config)
        return any(
            script.revision == revision
            for script in scripts.iterate_revisions("head", "base")
        )
    except (CommandError, RevisionError, OSError, RuntimeError, ValueError):
        return False


def _applied_schema_revision(database_url: str) -> str | None:
    """Read the migration version without creating a missing SQLite database."""

    url = make_url(database_url)
    rows: Sequence[Any]
    if url.get_backend_name() == "sqlite":
        database = url.database
        if not database or not Path(database).is_file():
            return None
        with sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True) as connection:
            rows = connection.execute("SELECT version_num FROM alembic_version").fetchall()
    else:
        engine = create_engine(url)
        try:
            with engine.connect() as connection:
                rows = connection.execute(text("SELECT version_num FROM alembic_version")).fetchall()
        finally:
            engine.dispose()
    return str(rows[0][0]) if len(rows) == 1 else None


def verify_release_identity(
    spec: Mapping[str, Any],
    *,
    cfg: Any,
    repo_root: Path,
    alembic_ini: Path,
    database_url: str,
    pre_migration: bool = False,
) -> dict[str, Any]:
    """Compare a reviewed release specification with observed local identity."""

    manifest = build_run_manifest(cfg)
    commit, dirty = _git_identity(repo_root)
    try:
        applied_revision = _applied_schema_revision(database_url)
    except (OSError, sqlite3.DatabaseError, SQLAlchemyError, RuntimeError, ValueError):
        applied_revision = None
    head_revision = _schema_head(alembic_ini)
    expected_commit = str(spec.get("tested_commit_sha") or "").lower()
    expected_config = str(spec.get("resolved_config_hash") or "").lower()
    expected_profile = str(spec.get("launch_profile_hash") or "").lower()
    expected_schema = str(spec.get("schema_revision") or "")
    mode = str(manifest.get("mode") or "").lower()
    legacy_model_field = _LEGACY_SPEC_FIELD in spec
    checks = {
        "spec_version": spec.get("schema_version") == 1,
        "spec_fields": set(spec) - {_LEGACY_SPEC_FIELD} == _SPEC_FIELDS,
        "tested_commit": len(expected_commit) == 40
        and all(char in "0123456789abcdef" for char in expected_commit)
        and commit == expected_commit,
        "clean_checkout": dirty is False,
        "config_hash": bool(expected_config) and manifest["resolved_config_hash"] == expected_config,
        "profile_hash": bool(expected_profile) and manifest["launch_profile_hash"] == expected_profile,
        "schema_revision": bool(expected_schema)
        and head_revision == expected_schema
        and applied_revision is not None
        and (applied_revision == expected_schema or (
            pre_migration and _schema_path_contains(alembic_ini, applied_revision)
        )),
    }
    return {
        "status": "pass" if all(checks.values()) else "blocked",
        "phase": "pre_migration" if pre_migration else "post_migration",
        "checked_at": datetime.now(UTC).isoformat(),
        "checks": checks,
        "observed": {
            "git_commit_hash": commit,
            "git_dirty": dirty,
            "resolved_config_hash": manifest["resolved_config_hash"],
            "launch_profile_hash": manifest["launch_profile_hash"],
            "schema_revision": applied_revision,
            "schema_head": head_revision,
            "model_identity": "legacy_field_ignored" if legacy_model_field else "absent",
            "mode": mode,
        },
    }


def _database_url_from_runtime() -> str:
    configured = str(get_env("DATABASE_URL", "", cast=str) or "").strip()
    if configured.startswith("postgres://"):
        return f"postgresql+psycopg://{configured[len('postgres://'):]}"
    if configured.startswith("postgresql://"):
        return f"postgresql+psycopg://{configured[len('postgresql://'):]}"
    if configured:
        return configured
    store_path = str(get_env("AI_TRADING_OMS_INTENT_STORE_PATH", "runtime/oms_intents.db", cast=str))
    if "://" in store_path:
        return store_path
    return f"sqlite:///{Path(store_path).expanduser().resolve()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument(
        "--models-root", type=Path, default=None,
        help="Ignored. Accepted while the installed systemd unit still passes it.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pre-migration", action="store_true")
    args = parser.parse_args(argv)
    if args.models_root is not None:
        logger.info(
            "RELEASE_IDENTITY_MODELS_ROOT_IGNORED",
            extra={"reason": "no_runtime_model_pin"},
        )
    repo_root = Path(__file__).resolve().parents[2]
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise ValueError("release spec must be a JSON object")
        result = verify_release_identity(
            spec,
            cfg=get_trading_config(),
            repo_root=repo_root,
            alembic_ini=repo_root / "alembic.ini",
            database_url=_database_url_from_runtime(),
            pre_migration=bool(args.pre_migration),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        result = {
            "status": "blocked", "checked_at": datetime.now(UTC).isoformat(),
            "checks": {"preflight_input": False},
            "reason": exc.__class__.__name__,
        }
    atomic_write_text(args.output, json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
