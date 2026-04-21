from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_trading.logging import get_logger

logger = get_logger(__name__)


def save_checkpoint(data: dict[str, Any], filepath: str) -> dict[str, Any]:
    """Serialize ``data`` to ``filepath`` as JSON.

    Returns a shallow copy of ``data`` or an empty dict if saving fails.
    """
    try:
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(data, fh, sort_keys=True, default=str)
        logger.info("CHECKPOINT_SAVED", extra={"path": filepath})
        return dict(data)
    except (OSError, TypeError, ValueError) as exc:  # pragma: no cover - error path
        logger.error("Failed to save checkpoint: %s", exc, exc_info=True)
        return {}


def load_checkpoint(filepath: str) -> dict[str, Any]:
    """Load a checkpoint dictionary from ``filepath``.

    Returns the loaded mapping or ``{}`` when the file is missing, invalid or
    does not contain a mapping.
    """
    p = Path(filepath)
    if not p.exists():
        logger.warning("Checkpoint file missing: %s", p)
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:  # pragma: no cover - error path
        logger.error("Failed to load checkpoint: %s", exc, exc_info=True)
        return {}
    if not isinstance(obj, dict):
        logger.error("Checkpoint file %s did not contain a dict", p)
        return {}
    logger.info("CHECKPOINT_LOADED", extra={"path": str(p)})
    return obj
