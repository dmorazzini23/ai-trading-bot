from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, cast

# Defer Flask dependency: module must import cleanly without Flask installed
_flask_blueprint: Any = None
_flask_current_app: Any = None
_flask_jsonify: Any = None
try:  # pragma: no cover - exercised via tests that stub Flask
    from flask import Blueprint as _FlaskBlueprint
    from flask import current_app as _FlaskCurrentApp
    from flask import jsonify as _FlaskJsonify
    _flask_blueprint = _FlaskBlueprint
    _flask_current_app = _FlaskCurrentApp
    _flask_jsonify = _FlaskJsonify
except Exception:  # pragma: no cover - allow import without Flask
    pass

from .env_diag import gather_env_diag


logger = logging.getLogger(__name__)

# Create blueprint only when Flask is available; otherwise leave as None so
# callers (e.g., app factory) can skip registration gracefully.
diag_bp = _flask_blueprint("diag", __name__) if _flask_blueprint is not None else None


def _invoke_snapshot(snapshot_fn: Callable[[], Mapping[str, Any]] | None) -> Mapping[str, Any] | None:
    if not callable(snapshot_fn):
        return None
    try:
        snapshot = snapshot_fn()
    except Exception:  # pragma: no cover - diagnostics should never raise
        logger.debug("BROKER_DIAG_SNAPSHOT_FAILED", exc_info=True)
        return {"error": "snapshot_failed"}
    if isinstance(snapshot, Mapping):
        return dict(snapshot)
    return {"error": "snapshot_invalid"}


if diag_bp is not None:
    @diag_bp.get("/diag")
    def diag() -> Any:  # pragma: no cover - exercised via app wiring tests
        """Return environment diagnostics plus an optional broker snapshot."""

        payload = gather_env_diag()
        try:
            current = cast(Any, _flask_current_app)
            app = current._get_current_object() if current is not None else None
        except Exception:  # pragma: no cover - defensive guard in tests
            app = None
        snapshot_fn = app.config.get("broker_snapshot_fn") if getattr(app, "config", None) else None
        broker_snapshot = _invoke_snapshot(snapshot_fn)
        if broker_snapshot:
            payload["broker"] = broker_snapshot
        if callable(_flask_jsonify):
            return _flask_jsonify(payload)
        return payload
else:
    # Fallback callable to aid direct import usage in environments without Flask.
    def diag() -> Any:  # pragma: no cover - import safety
        return gather_env_diag()
