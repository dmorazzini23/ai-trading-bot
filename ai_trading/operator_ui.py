from __future__ import annotations

from typing import Any

try:
    import flask as _flask
except ImportError:  # pragma: no cover - import smoke fallback
    _flask = None

from ai_trading.logging import get_logger
from ai_trading.operator_presets import PresetValidationError, build_plan, list_presets

logger = get_logger(__name__)

operator_bp = None
Blueprint = getattr(_flask, "Blueprint", None) if _flask is not None else None
jsonify = getattr(_flask, "jsonify", None) if _flask is not None else None
request = getattr(_flask, "request", None) if _flask is not None else None

if callable(Blueprint) and callable(jsonify):
    operator_bp = Blueprint("operator_ui", __name__, url_prefix="/operator")

    @operator_bp.get("/presets")
    def get_presets() -> Any:
        """List no-code presets for operator selection."""

        payload = {
            "ok": True,
            "presets": list_presets(),
        }
        return jsonify(payload), 200

    @operator_bp.post("/plan")
    def build_operator_plan() -> Any:
        """Validate selected preset and optional risk guardrail overrides."""

        body = request.get_json(silent=True) if request is not None else {}
        if not isinstance(body, dict):
            body = {}
        preset = body.get("preset")
        overrides = body.get("overrides")
        if not isinstance(overrides, dict):
            overrides = {}
        try:
            plan = build_plan(str(preset or ""), overrides)
        except PresetValidationError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except (TypeError, ValueError) as exc:
            logger.warning("OPERATOR_PLAN_BUILD_FAILED", extra={"error": str(exc)})
            return jsonify({"ok": False, "error": "invalid payload"}), 400

        return jsonify({"ok": True, "plan": plan}), 200


__all__ = ["operator_bp"]
