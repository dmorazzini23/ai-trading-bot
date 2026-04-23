"""JSON-safe model persistence helpers for simple research artifacts."""

from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import json
from pathlib import Path
from typing import Any

from ai_trading.logging import get_logger

logger = get_logger(__name__)


def save_model(model: Any, path: str | Path) -> Path:
    """Serialize a JSON-safe ``model`` to ``path``.

    The parent directory is created if needed. Any serialization error results
    in a :class:`RuntimeError` with the original exception chained.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:  # pragma: no cover - defensive
        logger.error(
            "MODEL_SAVE_ERROR", extra={"path": str(p), "error": str(exc)}
        )
        raise RuntimeError(
            f"Failed to save model at '{p}': only JSON-safe artifacts are supported ({exc})"
        ) from exc
    return p


def load_model(path: str | Path) -> Any:
    """Deserialize and return a JSON-safe model from ``path``.

    A :class:`RuntimeError` is raised if the file does not exist or cannot be
    deserialized.
    """
    p = Path(path)
    if not p.exists():
        logger.error("MODEL_FILE_MISSING", extra={"path": str(p)})
        raise RuntimeError(f"Model file not found: '{p}'")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:  # pragma: no cover - defensive
        logger.error(
            "MODEL_LOAD_ERROR", extra={"path": str(p), "error": str(exc)}
        )
        raise RuntimeError(
            f"Failed to load model from '{p}': only JSON-safe artifacts are supported ({exc})"
        ) from exc
