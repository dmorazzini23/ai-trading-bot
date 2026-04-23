"""Model registry utilities supporting governance workflows.

This module exposes a lightweight file-backed model registry used by tests and
the governance subsystem.  The legacy helper functions (``register_model``,
``set_active_model`` and ``get_active_model_meta``) continue to operate on
``registry.json`` for backwards compatibility, while :class:`ModelRegistry`
provides a richer interface for integration tests and promotion logic.
"""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import hashlib
import inspect
import json
import re
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger
from ai_trading.paths import MODELS_DIR

logger = get_logger(__name__)


_REGISTRY_PATH = MODELS_DIR / "registry.json"
_EVAL_DIR = MODELS_DIR / "eval"


def _load_registry() -> dict[str, Any]:
    try:
        if _REGISTRY_PATH.exists():
            payload = json.loads(_REGISTRY_PATH.read_text())
            return payload if isinstance(payload, dict) else {}
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        logger.debug("MODEL_REGISTRY_LOAD_FAILED", exc_info=True)
    return {}


def _save_registry(reg: dict[str, Any]) -> None:
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _REGISTRY_PATH.write_text(json.dumps(reg, indent=2))
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        logger.debug("MODEL_REGISTRY_SAVE_FAILED", exc_info=True)


def register_model(symbol: str, version: str, path: Path, meta: dict[str, Any] | None = None, activate: bool = True) -> None:
    """Register a model version for a symbol and optionally activate it."""
    reg = _load_registry()
    entry = reg.get(symbol) or {"versions": {}, "active": None}
    entry["versions"][version] = {
        "path": str(path),
        "meta": meta or {},
        "registered_at": datetime.now(UTC).isoformat(),
    }
    if activate:
        entry["active"] = version
    reg[symbol] = entry
    _save_registry(reg)


def set_active_model(symbol: str, version: str) -> None:
    reg = _load_registry()
    if symbol in reg and version in reg[symbol].get("versions", {}):
        reg[symbol]["active"] = version
        _save_registry(reg)


def get_active_model_meta(symbol: str) -> dict[str, Any] | None:
    reg = _load_registry()
    entry = reg.get(symbol)
    if not isinstance(entry, dict):
        return None
    ver = entry.get("active")
    if not isinstance(ver, str) or not ver:
        return None
    versions = entry.get("versions")
    if not isinstance(versions, dict):
        return None
    payload = versions.get(ver)
    return payload if isinstance(payload, dict) else None


def record_evaluation(symbol: str, metrics: dict[str, Any]) -> None:
    """Append evaluation metrics for a symbol in JSONL format."""
    try:
        _EVAL_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"symbol": symbol, "ts": datetime.now(UTC).isoformat(), **metrics}
        with (_EVAL_DIR / f"{symbol}.jsonl").open("a") as f:
            f.write(json.dumps(payload) + "\n")
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        logger.debug("MODEL_EVAL_WRITE_FAILED", exc_info=True)


def list_evaluations(symbol: str, limit: int = 100) -> list[dict[str, Any]]:
    try:
        path = _EVAL_DIR / f"{symbol}.jsonl"
        if not path.exists():
            return []
        lines = path.read_text().splitlines()[-limit:]
        return [json.loads(l) for l in lines if l.strip()]
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        logger.debug("MODEL_EVAL_READ_FAILED", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Rich model registry implementation


class ModelRegistry:
    """Filesystem-backed model registry with metadata and governance helpers."""

    _DEFAULT_INDEX_NAME = "registry_index.json"
    _MODELS_DIRNAME = "models"
    _ARTIFACT_FILENAME = "model.json"
    _META_FILENAME = "meta.json"

    def __init__(
        self,
        base_path: str | Path | None = None,
        *,
        index_filename: str | None = None,
    ) -> None:
        if base_path is None:
            env_override = get_env("MODEL_REGISTRY_DIR", None, cast=str, resolve_aliases=False)
            if env_override:
                base_path = Path(env_override).expanduser()
            else:
                base_path = MODELS_DIR
        self.base_path = Path(base_path)
        self.base_path = self.base_path.resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.base_path / self._MODELS_DIRNAME
        self.models_dir.mkdir(parents=True, exist_ok=True)

        index_name = index_filename or self._DEFAULT_INDEX_NAME
        self.index_path = self.base_path / index_name
        self.model_index: dict[str, dict[str, Any]] = self._load_index()

    # -- Persistence helpers -------------------------------------------------

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if not self.index_path.exists():
            return {}
        try:
            raw = json.loads(self.index_path.read_text())
        except (OSError, JSONDecodeError):
            logger.debug("MODEL_REGISTRY_INDEX_LOAD_FAILED", exc_info=True)
            return {}
        if not isinstance(raw, dict):
            return {}
        cleaned: dict[str, dict[str, Any]] = {}
        for model_id, payload in raw.items():
            if isinstance(model_id, str) and isinstance(payload, dict):
                cleaned[model_id] = payload
        return cleaned

    def _save_index(self) -> None:
        try:
            serialisable = {
                model_id: payload
                for model_id, payload in sorted(
                    self.model_index.items(),
                    key=lambda item: item[1].get("registered_at", ""),
                )
            }
            self.index_path.write_text(json.dumps(serialisable, indent=2, sort_keys=True))
        except OSError:
            logger.debug("MODEL_REGISTRY_INDEX_WRITE_FAILED", exc_info=True)

    @staticmethod
    def _slug(value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
        cleaned = cleaned.strip("._-")
        return cleaned or "model"

    def _generate_model_id(
        self,
        strategy: str,
        model_type: str,
        model_hash: str,
        registered_at: datetime,
    ) -> str:
        timestamp = registered_at.strftime("%Y%m%d%H%M%S")
        entropy = uuid.uuid4().hex[:8]
        return "-".join(
            (
                self._slug(strategy),
                self._slug(model_type),
                model_hash[:8],
                timestamp,
                entropy,
            )
        )

    @staticmethod
    def _normalise_tags(tags: Sequence[str] | None) -> list[str]:
        if not tags:
            return []
        cleaned: list[str] = []
        for tag in tags:
            if tag is None:
                continue
            cleaned.append(str(tag))
        return cleaned

    @staticmethod
    def _artifact_reference_from_mapping(value: Any) -> str | None:
        if not isinstance(value, Mapping):
            return None
        for key in ("artifact_path", "model_path"):
            candidate = str(value.get(key, "") or "").strip()
            if candidate:
                return candidate
        nested_paths = value.get("paths")
        if isinstance(nested_paths, Mapping):
            for key in ("artifact_path", "model_path"):
                candidate = str(nested_paths.get(key, "") or "").strip()
                if candidate:
                    return candidate
        return None

    def _extract_external_artifact_path(
        self,
        model: Any,
        metadata: Mapping[str, Any] | None,
    ) -> str | None:
        for candidate in (
            self._artifact_reference_from_mapping(metadata),
            self._artifact_reference_from_mapping(model),
        ):
            if candidate:
                return candidate
        return None

    @staticmethod
    def _artifact_hash_for_path(raw_path: str) -> str:
        path = Path(raw_path).expanduser()
        try:
            if path.is_file():
                return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            logger.debug("MODEL_REGISTRY_ARTIFACT_HASH_FAILED", exc_info=True)
        return hashlib.sha256(str(path).encode("utf-8")).hexdigest()

    @staticmethod
    def _serialise_inline_model(model: Any) -> tuple[bytes, str]:
        try:
            payload = json.dumps(model, indent=2, sort_keys=True).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Model registry no longer serializes arbitrary Python objects. "
                "Provide a JSON-safe inline model or an approved artifact path via "
                "metadata['model_path'] / metadata['artifact_path']."
            ) from exc
        return payload, "json"

    @staticmethod
    def _convert_metadata_value(value: Any) -> Any:
        if inspect.isclass(value):
            return f"{value.__module__}.{value.__qualname__}"
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, datetime):
            return value.astimezone(UTC).isoformat()
        if isinstance(value, Mapping):
            return {str(k): ModelRegistry._convert_metadata_value(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [ModelRegistry._convert_metadata_value(v) for v in value]
        return value

    def _write_metadata_file(self, model_dir: Path, payload: Mapping[str, Any]) -> None:
        meta_path = model_dir / self._META_FILENAME
        try:
            meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        except OSError:
            logger.debug("MODEL_REGISTRY_META_WRITE_FAILED", exc_info=True)

    def _read_metadata_file(self, model_dir: Path) -> dict[str, Any] | None:
        meta_path = model_dir / self._META_FILENAME
        if not meta_path.exists():
            return None
        try:
            payload = json.loads(meta_path.read_text())
            return payload if isinstance(payload, dict) else None
        except (OSError, JSONDecodeError):
            logger.debug("MODEL_REGISTRY_META_LOAD_FAILED", exc_info=True)
            return None

    @staticmethod
    def _registered_sort_key(info: Mapping[str, Any]) -> tuple[str, int]:
        return (
            str(info.get("registered_at", "") or ""),
            int(info.get("registered_seq", 0) or 0),
        )

    @staticmethod
    def _candidate_production_paths(info: Mapping[str, Any]) -> list[tuple[str, Path]]:
        candidates: list[tuple[str, Path]] = []
        governance = info.get("governance", {})
        if isinstance(governance, Mapping):
            runtime_promotion = governance.get("runtime_promotion", {})
            if isinstance(runtime_promotion, Mapping):
                runtime_model_path = str(runtime_promotion.get("model_path", "") or "").strip()
                if runtime_model_path:
                    candidates.append(("runtime_promotion", Path(runtime_model_path)))
        artifact_path = str(info.get("artifact_path", "") or "").strip()
        if artifact_path:
            candidates.append(("artifact", Path(artifact_path)))
        model_dir = str(info.get("path", "") or "").strip()
        if model_dir:
            candidates.append(("model_dir", Path(model_dir) / ModelRegistry._ARTIFACT_FILENAME))
        return candidates

    def _production_candidates(self, strategy: str) -> list[tuple[str, dict[str, Any]]]:
        candidates: list[tuple[str, dict[str, Any]]] = []
        for model_id, info in self.model_index.items():
            if info.get("strategy") != strategy:
                continue
            governance = info.get("governance", {}) or {}
            if governance.get("status") == "production":
                candidates.append((model_id, dict(info)))
        candidates.sort(
            key=lambda item: self._registered_sort_key(item[1]),
            reverse=True,
        )
        return candidates

    # -- Public API ----------------------------------------------------------

    def register_model(
        self,
        model: Any,
        strategy: str,
        model_type: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        dataset_fingerprint: str | None = None,
        tags: Sequence[str] | None = None,
        activate: bool = True,
    ) -> str:
        registered_at = datetime.now(UTC)
        dataset_fp = str(dataset_fingerprint) if dataset_fingerprint is not None else None
        external_artifact_path = self._extract_external_artifact_path(model, metadata)
        payload: bytes | None = None
        artifact_format: str
        if external_artifact_path:
            artifact_format = "external_path"
            model_hash = self._artifact_hash_for_path(external_artifact_path)
        else:
            payload, artifact_format = self._serialise_inline_model(model)
            model_hash = hashlib.sha256(payload).hexdigest()

        for _attempt in range(64):
            model_id = self._generate_model_id(strategy, model_type, model_hash, registered_at)
            model_dir = self.models_dir / model_id
            try:
                model_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            else:
                break
        else:
            raise RuntimeError("Failed to allocate unique model registry directory")
        if external_artifact_path:
            artifact_path = Path(external_artifact_path).expanduser()
        else:
            artifact_path = model_dir / self._ARTIFACT_FILENAME
            artifact_path.write_bytes(payload or b"")

        sanitised_metadata = {
            str(key): self._convert_metadata_value(val)
            for key, val in (metadata or {}).items()
        }
        tags_list = self._normalise_tags(tags)
        governance = {"status": "registered"}
        try:
            next_seq = max(int(entry.get("registered_seq", 0) or 0) for entry in self.model_index.values()) + 1
        except ValueError:
            next_seq = 1
        except (TypeError, AttributeError):
            next_seq = len(self.model_index) + 1

        index_entry = {
            "model_id": model_id,
            "strategy": strategy,
            "model_type": model_type,
            "registered_at": registered_at.isoformat(),
            "registered_seq": int(next_seq),
            "path": str(model_dir),
            "artifact_path": str(artifact_path),
            "artifact_format": artifact_format,
            "active": bool(activate),
            "model_hash": model_hash,
            "dataset_fingerprint": dataset_fp,
            "tags": tags_list,
            "metadata": sanitised_metadata,
            "governance": governance,
        }

        metadata_payload = {
            **index_entry,
            "metadata": sanitised_metadata,
        }

        self.model_index[model_id] = index_entry
        self._write_metadata_file(model_dir, metadata_payload)
        self._save_index()
        return model_id

    def latest_for(self, strategy: str, model_type: str) -> str | None:
        candidates = [
            (mid, info)
            for mid, info in self.model_index.items()
            if info.get("strategy") == strategy
            and info.get("model_type") == model_type
            and info.get("active", True)
        ]
        if not candidates:
            return None
        latest_id, _ = max(
            candidates,
            key=lambda item: (
                item[1].get("registered_at", ""),
                int(item[1].get("registered_seq", 0) or 0),
            ),
        )
        return latest_id

    def load_model(
        self,
        model_id: str,
        *,
        verify_dataset_hash: bool = False,
        expected_dataset_fingerprint: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        info = self.model_index.get(model_id)
        if info is None:
            raise FileNotFoundError(f"Model {model_id} not found in registry")

        model_dir = Path(info.get("path", self.models_dir / model_id))
        artifact_path = Path(
            info.get("artifact_path", model_dir / self._ARTIFACT_FILENAME)
        )
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact for model {model_id} missing at {artifact_path}")

        meta_payload = self._read_metadata_file(model_dir) or {**info}
        combined_meta = {
            **meta_payload.get("metadata", {}),
            "model_id": model_id,
            "strategy": info.get("strategy"),
            "model_type": info.get("model_type"),
            "registered_at": info.get("registered_at"),
            "dataset_fingerprint": info.get("dataset_fingerprint"),
            "tags": list(info.get("tags", [])),
            "governance": meta_payload.get("governance", info.get("governance", {})),
            "artifact_path": str(artifact_path),
            "artifact_format": str(info.get("artifact_format", "json") or "json"),
        }

        if verify_dataset_hash:
            expected = expected_dataset_fingerprint
            actual = info.get("dataset_fingerprint")
            if not actual:
                raise ValueError("Dataset fingerprint missing for model")
            if expected is None or actual != expected:
                raise ValueError("Dataset fingerprint mismatch")

        artifact_format = str(info.get("artifact_format", "json") or "json").strip().lower()
        if artifact_format == "external_path":
            return None, combined_meta
        if artifact_format == "json":
            try:
                return json.loads(artifact_path.read_text()), combined_meta
            except (OSError, JSONDecodeError) as exc:
                raise RuntimeError(f"Failed to load model {model_id}: {exc}") from exc
        raise RuntimeError(
            f"Failed to load model {model_id}: unsupported artifact format '{artifact_format}'"
        )

    def list_models(
        self,
        *,
        strategy: str | None = None,
        model_type: str | None = None,
        active_only: bool = False,
    ) -> list[dict[str, Any]] | list[str]:
        items: list[dict[str, Any]] = []
        for model_id, info in self.model_index.items():
            if strategy and info.get("strategy") != strategy:
                continue
            if model_type and info.get("model_type") != model_type:
                continue
            if active_only and not info.get("active", True):
                continue
            entry = {
                "model_id": model_id,
                "strategy": info.get("strategy"),
                "model_type": info.get("model_type"),
                "registered_at": info.get("registered_at"),
                "dataset_fingerprint": info.get("dataset_fingerprint"),
                "tags": list(info.get("tags", [])),
                "active": info.get("active", True),
                "governance": dict(info.get("governance", {})),
            }
            items.append(entry)
        items.sort(key=lambda item: item.get("registered_at", ""), reverse=True)
        if strategy is None and model_type is None and not active_only:
            return [entry["model_id"] for entry in items]
        return items

    def get_shadow_models(self, strategy: str) -> list[tuple[str, dict[str, Any]]]:
        results: list[tuple[str, dict[str, Any]]] = []
        for model_id, info in self.model_index.items():
            if info.get("strategy") != strategy:
                continue
            governance = info.get("governance", {}) or {}
            if governance.get("status") == "shadow":
                results.append((model_id, dict(info)))
        return results

    def get_production_model(self, strategy: str) -> tuple[str, dict[str, Any]] | None:
        candidates = self._production_candidates(strategy)
        return candidates[0] if candidates else None

    def get_viable_production_model(self, strategy: str) -> tuple[str, dict[str, Any]] | None:
        for model_id, info in self._production_candidates(strategy):
            for source, candidate_path in self._candidate_production_paths(info):
                try:
                    resolved_path = candidate_path.expanduser()
                except OSError:
                    continue
                if not resolved_path.is_file():
                    continue
                info_with_path = dict(info)
                info_with_path["production_path"] = str(resolved_path)
                info_with_path["production_path_source"] = source
                return model_id, info_with_path
        return None

    def update_governance_status(
        self,
        model_id: str,
        status: str,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        info = self.model_index.get(model_id)
        if info is None:
            raise ValueError(f"Model {model_id} not found")
        model_dir = Path(info.get("path", self.models_dir / model_id))
        meta_payload = self._read_metadata_file(model_dir) or {**info}

        governance = dict(meta_payload.get("governance", info.get("governance", {})))
        governance["status"] = status
        governance["updated_at"] = datetime.now(UTC).isoformat()
        if status == "shadow" and "shadow_start_time" not in governance:
            governance["shadow_start_time"] = governance["updated_at"]
        if extra:
            for key, value in extra.items():
                governance[str(key)] = self._convert_metadata_value(value)

        info["governance"] = governance
        meta_payload["governance"] = governance
        self.model_index[model_id] = info
        self._write_metadata_file(model_dir, meta_payload)
        self._save_index()

    def record_runtime_promotion(
        self,
        model_id: str,
        *,
        model_path: str | Path,
        manifest_path: str | Path | None = None,
    ) -> None:
        runtime_model_path = str(model_path).strip()
        if not runtime_model_path:
            raise ValueError("runtime model path is required")
        runtime_manifest_path = (
            str(manifest_path).strip() if manifest_path is not None else None
        )
        info = self.model_index.get(model_id)
        if info is None:
            raise ValueError(f"Model {model_id} not found")
        governance = info.get("governance", {}) or {}
        status = str(governance.get("status", "registered") or "registered")
        self.update_governance_status(
            model_id,
            status,
            extra={
                "runtime_promotion": {
                    "model_path": runtime_model_path,
                    "manifest_path": runtime_manifest_path,
                    "recorded_at": datetime.now(UTC).isoformat(),
                }
            },
        )


__all__ = [
    "ModelRegistry",
    "register_model",
    "set_active_model",
    "get_active_model_meta",
    "record_evaluation",
    "list_evaluations",
]
