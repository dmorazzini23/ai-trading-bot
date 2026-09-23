"""Consistent runtime database backup and isolated restore verification.

Bundles contain no environment or credential files. Restoring a bundle does not
grant trading authority; broker reconciliation and owner fencing remain required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text

logger = get_logger(__name__)

_RUNTIME_FILES = (
    "run_manifest.json",
    "release_spec.json",
    "release_identity_pre_migration.json",
    "release_identity_preflight.json",
    "effective_policy.json",
    "live_capital_readiness_latest.json",
    "model_selection_overrides.json",
    "policy_runtime_toggles.json",
    "live_canary_state_latest.json",
    "live_canary_state_latest.initialized",
    "live_canary_events.jsonl",
    "launch_profile_state_latest.json",
    "launch_profile_state_latest.initialized",
    "launch_profile_events.jsonl",
    "oms_events.jsonl",
    "oms_ledger.jsonl",
    "order_events.jsonl",
    "fill_events.jsonl",
    "decision_records.jsonl",
    "tca_records.jsonl",
    "broker_position_boundaries.jsonl",
)
_SCHEMA_VERSION = 1
_MODEL_SUFFIXES = {".pkl", ".joblib", ".json", ".onnx", ".pt", ".safetensors", ".zip"}
_SENSITIVE_NAME_PARTS = {"secret", "credential", "token", "private_key", ".env"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sqlite_integrity(path: Path) -> None:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        result = connection.execute("PRAGMA integrity_check").fetchone()
    if result != ("ok",):
        raise RuntimeError("sqlite_integrity_failed")


def _snapshot_sqlite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as origin:
        with sqlite3.connect(destination) as target:
            origin.backup(target)
    _sqlite_integrity(destination)


def _copy_stable(source: Path, destination: Path) -> None:
    before = source.stat()
    if not source.is_file() or source.is_symlink():
        raise RuntimeError("backup_source_not_regular_file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("backup_source_changed_during_copy")


def _manifest_entry(path: Path, *, root: Path, kind: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "kind": kind,
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _run_identity(runtime_stage: Path) -> dict[str, str | None]:
    manifest_path = runtime_stage / "run_manifest.json"
    if not manifest_path.is_file():
        return {"git_commit_hash": None, "resolved_config_hash": None, "effective_policy_hash": None}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise RuntimeError("run_manifest_invalid") from None
    if not isinstance(payload, dict):
        raise RuntimeError("run_manifest_invalid")
    return {
        name: str(payload[name]) if payload.get(name) not in (None, "") else None
        for name in ("git_commit_hash", "resolved_config_hash", "effective_policy_hash")
    }


def _write_status(status_path: Path, *, status: str, reason: str | None, bundle: str | None) -> None:
    payload = {
        "artifact_type": "runtime_recovery_backup_status",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "reason": reason,
        "bundle": bundle,
    }
    atomic_write_text(
        status_path,
        json.dumps(payload, sort_keys=True) + "\n",
    )


def create_backup(
    data_dir: Path,
    *,
    destination: Path | None = None,
    status_path: Path | None = None,
    retain_days: int = 7,
    max_snapshots: int = 14,
) -> Path:
    """Create a verified local bundle without pausing the running service."""

    runtime_dir = data_dir / "runtime"
    models_dir = data_dir / "models"
    backup_dir = destination or runtime_dir / "recovery_backups"
    report_path = status_path or runtime_dir / "recovery_backup_latest.json"
    backup_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if retain_days < 1 or max_snapshots < 1:
        raise ValueError("retention_must_be_positive")
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix=".recovery-stage-", dir=backup_dir) as raw_stage:
            stage = Path(raw_stage)
            entries: list[dict[str, Any]] = []
            databases = sorted(runtime_dir.glob("*.db"))
            if not databases:
                raise RuntimeError("runtime_databases_missing")
            for source in databases:
                if source.is_symlink() or not source.is_file():
                    raise RuntimeError("database_source_not_regular_file")
                target = stage / "runtime" / source.name
                _snapshot_sqlite(source, target)
                entries.append(_manifest_entry(target, root=stage, kind="sqlite"))
            for name in _RUNTIME_FILES:
                source = runtime_dir / name
                if not source.exists():
                    continue
                target = stage / "runtime" / name
                _copy_stable(source, target)
                entries.append(_manifest_entry(target, root=stage, kind="runtime_evidence"))
            if not models_dir.is_dir():
                raise RuntimeError("models_directory_missing")
            for source in sorted(models_dir.rglob("*")):
                if source.is_symlink() or not source.is_file():
                    continue
                lowered_name = source.name.lower()
                if source.suffix.lower() not in _MODEL_SUFFIXES or any(
                    marker in lowered_name for marker in _SENSITIVE_NAME_PARTS
                ):
                    continue
                target = stage / "models" / source.relative_to(models_dir)
                _copy_stable(source, target)
                entries.append(_manifest_entry(target, root=stage, kind="model"))
            if not any(entry["kind"] == "model" for entry in entries):
                raise RuntimeError("model_metadata_missing")
            manifest = {
                "schema_version": _SCHEMA_VERSION,
                "created_at": datetime.now(UTC).isoformat(),
                "duration_seconds": round(time.monotonic() - started, 3),
                "entries": entries,
                "run_identity": _run_identity(stage / "runtime"),
                "credential_files_included": False,
                "order_authority": "disabled_until_broker_reconciliation",
            }
            (stage / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            bundle = backup_dir / f"recovery.bak.{timestamp}-{uuid.uuid4().hex[:8]}.gz"
            temporary_bundle = backup_dir / f".{bundle.name}.{os.getpid()}.tmp"
            try:
                with tarfile.open(temporary_bundle, mode="w:gz") as archive:
                    for entry in entries:
                        archive.add(
                            stage / entry["path"], arcname=entry["path"], recursive=False
                        )
                    archive.add(stage / "manifest.json", arcname="manifest.json", recursive=False)
                verify_backup(temporary_bundle)
                with temporary_bundle.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary_bundle, bundle)
                bundle.chmod(0o600)
                directory_fd = os.open(backup_dir, os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                temporary_bundle.unlink(missing_ok=True)
        _prune_backups(backup_dir, retain_days=retain_days, max_snapshots=max_snapshots)
        _write_status(report_path, status="ok", reason=None, bundle=bundle.name)
        logger.info("RUNTIME_RECOVERY_BACKUP_OK", extra={"bundle": bundle.name})
        return bundle
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, sqlite3.Error, tarfile.TarError) as exc:
        _write_status(report_path, status="failed", reason=type(exc).__name__, bundle=None)
        logger.error("RUNTIME_RECOVERY_BACKUP_FAILED", extra={"reason": type(exc).__name__})
        raise


def _prune_backups(backup_dir: Path, *, retain_days: int, max_snapshots: int) -> None:
    cutoff = datetime.now(UTC) - timedelta(days=retain_days)
    bundles = sorted(
        backup_dir.glob("recovery.bak.*.gz"),
        key=lambda candidate: candidate.stat().st_mtime_ns,
        reverse=True,
    )
    for index, bundle in enumerate(bundles):
        try:
            timestamp = datetime.strptime(bundle.name[13:29], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            continue
        if index >= max_snapshots or timestamp < cutoff:
            bundle.unlink()


def verify_backup(bundle: Path) -> dict[str, Any]:
    """Check member identity, hashes, and restored SQLite integrity."""

    with tarfile.open(bundle, mode="r:gz") as archive:
        manifest_member = archive.getmember("manifest.json")
        manifest_file = archive.extractfile(manifest_member)
        if manifest_file is None:
            raise RuntimeError("backup_manifest_missing")
        manifest = json.load(manifest_file)
        if not isinstance(manifest, dict):
            raise RuntimeError("backup_manifest_invalid")
        if manifest.get("schema_version") != _SCHEMA_VERSION:
            raise RuntimeError("backup_schema_unsupported")
        entries = manifest.get("entries")
        if not isinstance(entries, list) or not entries:
            raise RuntimeError("backup_manifest_invalid")
        expected: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise RuntimeError("backup_manifest_invalid")
            name = entry.get("path")
            if not isinstance(name, str) or "\\" in name:
                raise RuntimeError("backup_path_unsafe")
            pure = PurePosixPath(name)
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or len(pure.parts) < 2
                or pure.parts[0] not in {"runtime", "models"}
                or pure.as_posix() != name
                or name in expected
            ):
                raise RuntimeError("backup_path_unsafe")
            if (
                entry.get("kind") not in {"sqlite", "runtime_evidence", "model"}
                or not isinstance(entry.get("size"), int)
                or entry["size"] < 0
                or not isinstance(entry.get("sha256"), str)
                or len(entry["sha256"]) != 64
            ):
                raise RuntimeError("backup_manifest_invalid")
            expected[name] = entry
        actual = {member.name: member for member in archive.getmembers()}
        if set(actual) != set(expected) | {"manifest.json"}:
            raise RuntimeError("backup_members_mismatch")
        with tempfile.TemporaryDirectory(prefix="recovery-verify-") as raw_verify:
            verify_dir = Path(raw_verify)
            for name, entry in expected.items():
                member = actual[name]
                if not member.isfile() or member.size != entry["size"]:
                    raise RuntimeError("backup_member_invalid")
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError("backup_member_missing")
                digest = hashlib.sha256()
                with stream:
                    if entry["kind"] == "sqlite":
                        target = verify_dir / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open("wb") as output:
                            while chunk := stream.read(1024 * 1024):
                                digest.update(chunk)
                                output.write(chunk)
                        _sqlite_integrity(target)
                    else:
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
                if digest.hexdigest() != entry["sha256"]:
                    raise RuntimeError("backup_checksum_mismatch")
    return manifest


def restore_to_directory(bundle: Path, target: Path) -> dict[str, Any]:
    """Restore only to a new isolated directory; never activate trading."""

    manifest = verify_backup(bundle)
    if target.exists():
        raise FileExistsError("restore_target_must_not_exist")
    target.mkdir(parents=True, mode=0o700)
    try:
        with tarfile.open(bundle, mode="r:gz") as archive:
            for entry in manifest["entries"]:
                name = entry["path"]
                stream = archive.extractfile(name)
                if stream is None:
                    raise RuntimeError("backup_member_missing")
                output_path = target / name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with stream, output_path.open("wb") as output:
                    shutil.copyfileobj(stream, output)
        atomic_write_text(
            target / "manifest.json",
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        )
        return manifest
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, sqlite3.Error, tarfile.TarError):
        shutil.rmtree(target)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/var/lib/ai-trading-bot"))
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--status-path", type=Path)
    parser.add_argument("--retain-days", type=int, default=7)
    parser.add_argument("--max-snapshots", type=int, default=14)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--restore-to", type=Path)
    args = parser.parse_args()
    if args.restore_to is not None:
        if args.verify is None:
            parser.error("--restore-to requires --verify BUNDLE")
        restore_to_directory(args.verify, args.restore_to)
    elif args.verify is not None:
        verify_backup(args.verify)
    else:
        create_backup(
            args.data_dir,
            destination=args.destination,
            status_path=args.status_path,
            retain_days=args.retain_days,
            max_snapshots=args.max_snapshots,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
