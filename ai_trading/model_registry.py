"""Clean model registry for storage, versioning, and retrieval.

This module persists arbitrary model objects using ``pickle``. Paths are
validated before loading to guard against unsafe deserialization.
"""
from __future__ import annotations

from ai_trading.logging import get_logger
import hashlib
import json
import os
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_trading.utils.pickle_safe import safe_pickle_load
try:
    from ai_trading.logging import logger
except (ValueError, TypeError):
    import logging
    logger = get_logger(__name__)

class ModelRegistry:
    """Centralized registry for trained models."""

    def __init__(self, base_path: str | None = None):
        base = base_path or os.getenv("MODEL_REGISTRY_DIR") or Path.cwd() / "models"
        self.base_path = Path(base).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.base_path / "registry_index.json"
        self.model_index: dict[str, dict[str, Any]] = self._load_index()
        logger.info('ModelRegistry initialized at %s', self.base_path)

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if not self.index_file.exists():
            return {}
        try:
            return json.loads(self.index_file.read_text(encoding='utf-8'))
        except (ValueError, TypeError) as e:
            logger.warning('Failed to load registry index: %s', e)
            return {}

    def _save_index(self) -> None:
        self.index_file.write_text(json.dumps(self.model_index, indent=2), encoding='utf-8')

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:16]

    def register_model(self, model: Any, strategy: str, model_type: str, metadata: dict[str, Any] | None=None, dataset_fingerprint: str | None=None, tags: list[str] | None=None) -> str:
        """Store model + metadata and return deterministic ID."""
        try:
            blob = pickle.dumps(model)
        except (ValueError, TypeError) as e:
            raise RuntimeError(f'Model not picklable: {e}') from e
        content_hash = self._hash_bytes(blob)
        id_components = [strategy, model_type, content_hash]
        if dataset_fingerprint:
            id_components.append(dataset_fingerprint[:16])
        model_id = '-'.join(id_components)
        model_dir = (self.base_path / model_id).resolve()
        if not model_dir.is_relative_to(self.base_path):
            raise RuntimeError(f"Model path escapes base directory: {model_dir}")
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.pkl").write_bytes(blob)
        meta = {'strategy': strategy, 'model_type': model_type, 'registration_time': datetime.now(UTC).isoformat(), 'dataset_fingerprint': dataset_fingerprint, 'tags': tags or []}
        if metadata:
            meta.update(metadata)
        (model_dir / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        self.model_index[model_id] = meta
        self._save_index()
        logger.info('Registered model %s', model_id)
        return model_id

    def load_model(self, model_id: str, verify_dataset_hash: bool=False, expected_dataset_fingerprint: str | None=None) -> tuple[Any, dict[str, Any]]:
        """Return (model, metadata); optionally verify dataset fingerprint."""
        model_dir = (self.base_path / model_id).resolve()
        if not model_dir.is_relative_to(self.base_path):
            raise RuntimeError(f"Model path escapes base directory: {model_dir}")
        model_path = model_dir / "model.pkl"
        meta_path = model_dir / "meta.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Model {model_id} not found in registry")
        model = safe_pickle_load(model_path, [self.base_path])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if verify_dataset_hash and expected_dataset_fingerprint:
            got = meta.get('dataset_fingerprint')
            if got != expected_dataset_fingerprint:
                raise ValueError(f'Dataset fingerprint mismatch: expected {expected_dataset_fingerprint}, got {got}')
        return (model, meta)

    def latest_for(self, strategy: str, model_type: str) -> str | None:
        """Return most recently registered ID for (strategy, model_type)."""
        candidates = [(mid, m.get('registration_time', '')) for mid, m in self.model_index.items() if m.get('strategy') == strategy and m.get('model_type') == model_type]
        if not candidates:
            return None
        return sorted(candidates, key=lambda t: t[1])[-1][0]