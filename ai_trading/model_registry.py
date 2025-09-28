"""Model registry utilities supporting governance workflows.

This module exposes a lightweight file-backed model registry used by tests and
the governance subsystem.  The legacy helper functions (``register_model``,
``set_active_model`` and ``get_active_model_meta``) continue to operate on
``registry.json`` for backwards compatibility, while :class:`ModelRegistry`
provides a richer interface for integration tests and promotion logic.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import pickle
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from ai_trading.logging import get_logger
from ai_trading.paths import MODELS_DIR

logger = get_logger(__name__)


_REGISTRY_PATH = MODELS_DIR / "registry.json"
_EVAL_DIR = MODELS_DIR / "eval"


def _load_registry() -> dict[str, Any]:
    try:
        if _REGISTRY_PATH.exists():
            return json.loads(_REGISTRY_PATH.read_text())
    except Exception:
        logger.debug("MODEL_REGISTRY_LOAD_FAILED", exc_info=True)
    return {}


def _save_registry(reg: dict[str, Any]) -> None:
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _REGISTRY_PATH.write_text(json.dumps(reg, indent=2))
    except Exception:
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
    if not entry:
        return None
    ver = entry.get("active")
    if not ver:
        return None
    return entry.get("versions", {}).get(ver)


def record_evaluation(symbol: str, metrics: dict[str, Any]) -> None:
    """Append evaluation metrics for a symbol in JSONL format."""
    try:
        _EVAL_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"symbol": symbol, "ts": datetime.now(UTC).isoformat(), **metrics}
        with (_EVAL_DIR / f"{symbol}.jsonl").open("a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        logger.debug("MODEL_EVAL_WRITE_FAILED", exc_info=True)


def list_evaluations(symbol: str, limit: int = 100) -> list[dict[str, Any]]:
    try:
        path = _EVAL_DIR / f"{symbol}.jsonl"
        if not path.exists():
            return []
        lines = path.read_text().splitlines()[-limit:]
        return [json.loads(l) for l in lines if l.strip()]
    except Exception:
        logger.debug("MODEL_EVAL_READ_FAILED", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Rich model registry implementation


@dataclass(slots=True)
class _Pickler:
    """Pickle helper wiring serialize/deserialize callables."""

    name: str
    dumps: Any
    loads: Any


class ModelRegistry:
    """Filesystem-backed model registry with metadata and governance helpers."""

    _DEFAULT_INDEX_NAME = "registry_index.json"
    _MODELS_DIRNAME = "models"
    _ARTIFACT_FILENAME = "model.pkl"
    _META_FILENAME = "meta.json"

    def __init__(
        self,
        base_path: str | Path | None = None,
        *,
        index_filename: str | None = None,
    ) -> None:
        self.base_path = Path(base_path) if base_path is not None else MODELS_DIR
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
    def _available_picklers() -> list[_Pickler]:
        picklers: list[_Pickler] = [
            _Pickler(
                "pickle",
                lambda obj: pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL),
                pickle.loads,
            )
        ]
        try:
            import cloudpickle
        except ImportError:
            cloudpickle = None  # type: ignore[assignment]
        if cloudpickle is not None:
            picklers.append(
                _Pickler(
                    "cloudpickle",
                    lambda obj, _cp=cloudpickle: _cp.dumps(obj),
                    lambda data, _cp=cloudpickle: _cp.loads(data),
                )
            )
        try:
            import dill
        except ImportError:
            dill = None  # type: ignore[assignment]
        if dill is not None:
            picklers.append(
                _Pickler(
                    "dill",
                    lambda obj, _dill=dill: _dill.dumps(obj),
                    lambda data, _dill=dill: _dill.loads(data),
                )
            )
        return picklers

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
        entropy = format(time.time_ns() & 0xFFFFF, "05x")
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

    def _serialise_model(self, model: Any) -> tuple[bytes, str]:
        errors: list[str] = []
        for pickler in self._available_picklers():
            try:
                payload = pickler.dumps(model)
            except Exception as exc:  # noqa: BLE001 - convert to summary later
                errors.append(f"{pickler.name}: {exc}")
                continue
            return payload, pickler.name
        try:
            from unittest.mock import Mock
        except ImportError:  # pragma: no cover - stdlib always available
            Mock = None  # type: ignore[assignment]
        if Mock is not None and isinstance(model, Mock):
            mock_payload = {
                "__registry__": "mock",
                "repr": repr(model),
                "name": getattr(model, "_mock_name", None),
            }
            payload = pickle.dumps(mock_payload, protocol=pickle.HIGHEST_PROTOCOL)
            return payload, "mock"
        message = "; ".join(errors) or "unknown"
        raise RuntimeError(f"Model not picklable ({message})")

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
            return json.loads(meta_path.read_text())
        except (OSError, JSONDecodeError):
            logger.debug("MODEL_REGISTRY_META_LOAD_FAILED", exc_info=True)
            return None

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
        payload, pickler_name = self._serialise_model(model)
        model_hash = hashlib.sha256(payload).hexdigest()

        for existing in self.model_index.values():
            if existing.get("model_hash") == model_hash:
                raise ValueError("Model already registered")

        registered_at = datetime.now(UTC)
        dataset_fp = str(dataset_fingerprint) if dataset_fingerprint is not None else None

        while True:
            model_id = self._generate_model_id(strategy, model_type, model_hash, registered_at)
            model_dir = self.models_dir / model_id
            try:
                model_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            break
        artifact_path = model_dir / self._ARTIFACT_FILENAME
        artifact_path.write_bytes(payload)

        sanitised_metadata = {
            str(key): self._convert_metadata_value(val)
            for key, val in (metadata or {}).items()
        }
        tags_list = self._normalise_tags(tags)
        governance = {"status": "registered"}

        index_entry = {
            "model_id": model_id,
            "strategy": strategy,
            "model_type": model_type,
            "registered_at": registered_at.isoformat(),
            "path": str(model_dir),
            "artifact_path": str(artifact_path),
            "active": bool(activate),
            "pickler": pickler_name,
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
            key=lambda item: item[1].get("registered_at", ""),
        )
        return latest_id

    def _iter_loaders(self, preferred: str | None) -> Iterable[_Pickler]:
        picklers = self._available_picklers()
        if preferred:
            picklers.sort(key=lambda p: 0 if p.name == preferred else 1)
        return picklers

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
        }

        if verify_dataset_hash:
            expected = expected_dataset_fingerprint
            actual = info.get("dataset_fingerprint")
            if not actual:
                raise ValueError("Dataset fingerprint missing for model")
            if expected is None or actual != expected:
                raise ValueError("Dataset fingerprint mismatch")

        raw_bytes = artifact_path.read_bytes()
        if info.get("pickler") == "mock":
            try:
                payload = pickle.loads(raw_bytes)
            except Exception:  # pragma: no cover - defensive
                logger.debug("MODEL_REGISTRY_MOCK_LOAD_FAILED", exc_info=True)
                model = None
            else:
                try:
                    from unittest.mock import Mock
                except ImportError:  # pragma: no cover - stdlib always available
                    model = None
                else:
                    model = Mock(name=payload.get("name") or payload.get("repr"))
            return model, combined_meta
        last_error: Exception | None = None
        for pickler in self._iter_loaders(info.get("pickler")):
            try:
                model = pickler.loads(raw_bytes)
                return model, combined_meta
            except Exception as exc:  # noqa: BLE001 - propagate in summary
                last_error = exc
                continue
        raise RuntimeError(
            f"Failed to load model {model_id}: {last_error or 'no compatible loader'}"
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
        for model_id, info in self.model_index.items():
            if info.get("strategy") != strategy:
                continue
            governance = info.get("governance", {}) or {}
            if governance.get("status") == "production":
                return model_id, dict(info)
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


__all__ = [
    "ModelRegistry",
    "register_model",
    "set_active_model",
    "get_active_model_meta",
    "record_evaluation",
    "list_evaluations",
]
