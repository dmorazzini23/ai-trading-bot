from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

from flask import Blueprint, current_app, jsonify

from .env_diag import gather_env_diag


logger = logging.getLogger(__name__)

diag_bp = Blueprint("diag", __name__)


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


@diag_bp.get("/diag")
def diag() -> Any:
    """Return environment diagnostics plus an optional broker snapshot."""

    payload = gather_env_diag()
    app = current_app._get_current_object()
    snapshot_fn = app.config.get("broker_snapshot_fn") if hasattr(app, "config") else None
    broker_snapshot = _invoke_snapshot(snapshot_fn)
    if broker_snapshot:
        payload["broker"] = broker_snapshot
    return jsonify(payload)

