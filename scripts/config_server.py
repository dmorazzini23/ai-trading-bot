"""Small authenticated runtime configuration update service."""

from __future__ import annotations

import hmac
import json
from collections.abc import Mapping
from typing import Any

try:
    from flask import Flask, jsonify, request
except ImportError:
    from flask import Flask, jsonify

    request = None  # type: ignore[assignment]

from ai_trading.config import management as config
from ai_trading.config.management import get_env
from ai_trading.logging import get_logger

logger = get_logger(__name__)
app = Flask(__name__)
app_config = getattr(app, "config", None)
if isinstance(app_config, dict):
    app_config["MAX_CONTENT_LENGTH"] = int(
        get_env("AI_TRADING_CONFIG_SERVER_MAX_CONTENT_LENGTH", "65536", cast=int) or 65536
    )


def _parse_operator_token_map() -> dict[str, str]:
    raw = str(get_env("AI_TRADING_OPERATOR_TOKEN_MAP", "") or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, Mapping):
        return {
            str(operator_id).strip().lower(): str(token).strip()
            for operator_id, token in parsed.items()
            if str(operator_id).strip() and str(token).strip()
        }
    result: dict[str, str] = {}
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if "=" in token:
            operator_id, secret = token.split("=", 1)
        elif ":" in token:
            operator_id, secret = token.split(":", 1)
        else:
            continue
        operator_key = operator_id.strip().lower()
        secret_value = secret.strip()
        if operator_key and secret_value:
            result[operator_key] = secret_value
    return result


def _require_operator_auth() -> tuple[dict[str, Any] | None, int]:
    if request is None:
        logger.error("CONFIG_SERVER_REQUEST_CONTEXT_UNAVAILABLE")
        return {"error": "request context unavailable"}, 503
    token_map = _parse_operator_token_map()
    configured_token = str(get_env("AI_TRADING_CONFIG_SERVER_TOKEN", "") or "").strip()
    if not token_map and not configured_token:
        logger.error("CONFIG_SERVER_AUTH_MISCONFIGURED")
        return {"error": "configuration server authentication is not configured"}, 503

    authorization = str(request.headers.get("Authorization", "") or "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return {"error": "operator authentication required"}, 401

    operator_id = str(
        request.headers.get("X-AI-Trading-Operator-Id")
        or request.headers.get("X-Operator-Id")
        or ""
    ).strip()
    token_value = token.strip()
    if token_map:
        expected = token_map.get(operator_id.lower())
        if not operator_id or expected is None:
            return {"error": "operator is not authorized for this action"}, 403
        if not hmac.compare_digest(token_value, expected):
            return {"error": "operator authentication rejected"}, 403
        return None, 200

    if hmac.compare_digest(token_value, configured_token):
        return None, 200
    return {"error": "operator authentication rejected"}, 403


def _validated_payload(data: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    try:
        volume_thr = float(data.get("volume_spike_threshold", 1.5))
        ml_thr = float(data.get("ml_confidence_threshold", 0.5))
    except (TypeError, ValueError):
        return None, {"error": "Invalid number format"}
    if not 0.1 <= volume_thr <= 10.0:
        return None, {"error": "volume_spike_threshold must be between 0.1 and 10.0"}
    if not 0.0 <= ml_thr <= 1.0:
        return None, {"error": "ml_confidence_threshold must be between 0.0 and 1.0"}

    pyramid_levels = data.get(
        "pyramid_levels",
        {"high": 0.4, "medium": 0.25, "low": 0.15},
    )
    if not isinstance(pyramid_levels, Mapping) or len(pyramid_levels) > 20:
        return None, {"error": "pyramid_levels must be an object with at most 20 entries"}
    validated_levels: dict[str, float] = {}
    for raw_level, raw_value in pyramid_levels.items():
        level = str(raw_level).strip()
        if not level or len(level) > 64:
            return None, {"error": "Invalid pyramid level name"}
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None, {"error": f"Invalid pyramid level {level}: {raw_value}"}
        if not 0.0 <= value <= 1.0:
            return None, {"error": f"Invalid pyramid level {level}: {raw_value}"}
        validated_levels[level] = value
    return (
        {
            "volume_spike_threshold": volume_thr,
            "ml_confidence_threshold": ml_thr,
            "pyramid_levels": validated_levels,
        },
        None,
    )


@app.route("/update_config", methods=["POST"])
def update_config():
    if request is None:
        return jsonify({"error": "request context unavailable"}), 503
    auth_error, auth_status = _require_operator_auth()
    if auth_error is not None:
        return jsonify(auth_error), auth_status
    data = request.get_json(silent=True)
    if not isinstance(data, Mapping):
        return jsonify({"error": "JSON object required"}), 400
    validated, error = _validated_payload(data)
    if error is not None or validated is None:
        return jsonify(error), 400

    set_runtime_config = getattr(config, "set_runtime_config", None)
    if not callable(set_runtime_config):
        logger.error("CONFIG_SERVER_SET_RUNTIME_CONFIG_UNAVAILABLE")
        return jsonify({"error": "Runtime configuration updates unavailable"}), 503
    try:
        set_runtime_config(
            validated["volume_spike_threshold"],
            validated["ml_confidence_threshold"],
            validated["pyramid_levels"],
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("CONFIG_SERVER_UPDATE_FAILED", extra={"error": str(exc)})
        return jsonify({"error": "Runtime configuration update failed"}), 503
    return jsonify({"status": "updated", **validated})


if __name__ == "__main__":
    host = str(get_env("AI_TRADING_CONFIG_SERVER_HOST", "127.0.0.1") or "127.0.0.1")
    port = int(get_env("AI_TRADING_CONFIG_SERVER_PORT", "5002", cast=int) or 5002)
    app.run(host=host, port=port)
