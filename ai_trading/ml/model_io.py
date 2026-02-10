"""Model persistence helpers using :mod:`dill`/``cloudpickle`` when available.

The default :mod:`pickle` module cannot serialize lambda functions. To support
models that may reference lambdas, this module attempts to use :mod:`dill`
first, then ``cloudpickle``. If neither is installed the standard
:mod:`pickle` module is used as a fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_trading.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - import error path exercised in tests
    import dill as _pickle
except Exception:  # pragma: no cover - fallback when dill missing
    try:
        import cloudpickle as _pickle  # type: ignore
    except Exception:
        import pickle as _pickle  # type: ignore


def save_model(model: Any, path: str | Path) -> Path:
    """Serialize ``model`` to ``path``.

    The parent directory is created if needed. Any serialization error results
    in a :class:`RuntimeError` with the original exception chained.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with p.open("wb") as fh:
            _pickle.dump(model, fh)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "MODEL_SAVE_ERROR", extra={"path": str(p), "error": str(exc)}
        )
        raise RuntimeError(f"Failed to save model at '{p}': {exc}") from exc
    return p


def load_model(path: str | Path) -> Any:
    """Deserialize and return a model from ``path``.

    A :class:`RuntimeError` is raised if the file does not exist or cannot be
    deserialized.
    """
    p = Path(path)
    if not p.exists():
        logger.error("MODEL_FILE_MISSING", extra={"path": str(p)})
        raise RuntimeError(f"Model file not found: '{p}'")
    try:
        with p.open("rb") as fh:
            return _pickle.load(fh)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "MODEL_LOAD_ERROR", extra={"path": str(p), "error": str(exc)}
        )
        raise RuntimeError(f"Failed to load model from '{p}': {exc}") from exc
