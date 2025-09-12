from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from ai_trading.logging import get_logger
from ai_trading.utils.pickle_safe import safe_pickle_load

logger = get_logger(__name__)


def save_checkpoint(data: dict[str, Any], filepath: str) -> dict[str, Any]:
    """Serialize ``data`` to ``filepath`` with :mod:`pickle`.

    Returns a shallow copy of ``data`` or an empty dict if saving fails.
    """
    try:
        with open(filepath, "wb") as fh:
            pickle.dump(data, fh)
        logger.info("CHECKPOINT_SAVED", extra={"path": filepath})
        return dict(data)
    except (OSError, pickle.PickleError) as exc:  # pragma: no cover - error path
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
        obj = safe_pickle_load(p, [p.parent])
    except RuntimeError as exc:  # pragma: no cover - error path
        logger.error("Failed to load checkpoint: %s", exc, exc_info=True)
        return {}
    if not isinstance(obj, dict):
        logger.error("Checkpoint file %s did not contain a dict", p)
        return {}
    logger.info("CHECKPOINT_LOADED", extra={"path": str(p)})
    return obj
