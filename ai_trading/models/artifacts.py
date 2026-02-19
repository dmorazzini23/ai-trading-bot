"""Model artifact manifest helpers and checksum verification."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class ArtifactManifest:
    model_version: str
    checksum_sha256: str
    created_ts: str
    training_data_range: Mapping[str, str] | None = None
    signature: str | None = None
    metadata: Mapping[str, Any] | None = None


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def default_manifest_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    return Path(f"{path}.manifest.json")


def build_artifact_manifest(
    *,
    model_path: str | Path,
    model_version: str,
    training_data_range: Mapping[str, str] | None = None,
    created_ts: datetime | None = None,
    signature: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactManifest:
    path = Path(model_path)
    checksum = _sha256_file(path)
    ts = (created_ts or datetime.now(UTC)).astimezone(UTC).isoformat()
    return ArtifactManifest(
        model_version=str(model_version),
        checksum_sha256=checksum,
        created_ts=ts,
        training_data_range=dict(training_data_range or {}),
        signature=signature,
        metadata=dict(metadata or {}),
    )


def write_artifact_manifest(
    *,
    model_path: str | Path,
    model_version: str,
    training_data_range: Mapping[str, str] | None = None,
    manifest_path: str | Path | None = None,
    signature: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    manifest = build_artifact_manifest(
        model_path=model_path,
        model_version=model_version,
        training_data_range=training_data_range,
        signature=signature,
        metadata=metadata,
    )
    destination = Path(manifest_path) if manifest_path else default_manifest_path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(manifest), sort_keys=True), encoding="utf-8")
    return destination


def load_artifact_manifest(manifest_path: str | Path) -> ArtifactManifest:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return ArtifactManifest(
        model_version=str(payload.get("model_version", "")),
        checksum_sha256=str(payload.get("checksum_sha256", "")),
        created_ts=str(payload.get("created_ts", "")),
        training_data_range=payload.get("training_data_range"),
        signature=payload.get("signature"),
        metadata=payload.get("metadata"),
    )


def verify_artifact(
    *,
    model_path: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[bool, str]:
    model = Path(model_path)
    if not model.is_file():
        return False, "MODEL_FILE_MISSING"

    manifest_file = Path(manifest_path) if manifest_path else default_manifest_path(model)
    if not manifest_file.is_file():
        return False, "MANIFEST_MISSING"

    try:
        manifest = load_artifact_manifest(manifest_file)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False, "MANIFEST_INVALID"

    expected = manifest.checksum_sha256.strip().lower()
    if not expected:
        return False, "MANIFEST_CHECKSUM_MISSING"

    try:
        observed = _sha256_file(model).lower()
    except OSError:
        return False, "MODEL_READ_FAILED"

    if observed != expected:
        return False, "CHECKSUM_MISMATCH"
    return True, "OK"


def manifest_dict(manifest: ArtifactManifest) -> dict[str, Any]:
    return asdict(manifest)
