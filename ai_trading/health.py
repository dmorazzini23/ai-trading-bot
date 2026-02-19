from __future__ import annotations

"""Lightweight health check HTTP server utilities."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping

from ai_trading.logging import get_logger
from ai_trading.telemetry import runtime_state
from ai_trading.app import _install_route_tracker, _ensure_test_client

try:  # pragma: no cover - optional dependency
    from flask import Flask, jsonify
except Exception:  # pragma: no cover - stub for tests when flask missing
    class Flask:  # type: ignore
        def __init__(self, *a, **k):
            self.config = {}

        def route(self, *a, **k):
            def deco(func):
                return func
            return deco

        def run(self, *a, **k):
            return None

    def jsonify(payload):  # type: ignore
        return payload


@dataclass(slots=True)
class HealthCheck:
    """Expose simple `/healthz` endpoint via an internal Flask app.

    Parameters
    ----------
    ctx:
        Optional context object. Attribute access is guarded so that missing
        properties never raise :class:`AttributeError`.
    config:
        Optional configuration mapping passed to the internal Flask
        application. If omitted, an empty dict is used which avoids `None`
        lookups when applying configuration.
    """

    ctx: Any | None = None
    config: dict[str, Any] | None = None
    app: Flask = field(init=False)

    def __post_init__(self) -> None:
        # Ensure we always work with a mapping to avoid ``None`` handling logic
        # throughout the class.
        self.config = dict(self.config or {})

        # Create the Flask app and apply the supplied configuration. Flask's
        # ``config`` attribute is a mutable mapping so ``update`` is safe even
        # when the dict is empty.
        self.app = Flask(__name__)
        if not hasattr(self.app, "config") or not isinstance(self.app.config, dict):
            setattr(self.app, "config", dict(getattr(self.app, "config", {}) or {}))
        self.app.config.update(self.config)

        route_registry = _install_route_tracker(self.app)
        self._register_routes()
        _ensure_test_client(self.app, route_registry)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_ctx_attr(self, name: str, default: Any = None) -> Any:
        """Safely fetch an attribute from the provided context object."""
        if self.ctx is None:
            return default
        return getattr(self.ctx, name, default)

    def _register_routes(self) -> None:
        """Register default health route."""

        def _emit_response(body: dict, status: int = 200):
            response = jsonify(body)
            if response is None or isinstance(response, Mapping):
                return body if status == 200 else (body, status)
            if not (
                callable(getattr(response, "get_data", None))
                or callable(getattr(response, "get_json", None))
                or hasattr(response, "status_code")
            ):
                return body if status == 200 else (body, status)
            try:
                response.status_code = status
            except Exception:
                pass
            return response

        @self.app.route("/healthz")
        def _healthz() -> Any:  # pragma: no cover - simple glue
            try:
                ok = True
                err: str | None = None
                try:
                    service_name = self._get_ctx_attr("service", "ai-trading")
                except Exception as exc:  # pragma: no cover - defensive
                    ok = False
                    service_name = "ai-trading"
                    err = str(exc) or exc.__class__.__name__

                ctx_alpaca = self._get_ctx_attr("alpaca", None)
                alpaca_ctx = dict(ctx_alpaca) if isinstance(ctx_alpaca, Mapping) else {}
                alpaca_payload = {
                    "sdk_ok": False,
                    "initialized": False,
                    "client_attached": False,
                    "has_key": False,
                    "has_secret": False,
                    "base_url": "",
                    "paper": False,
                    "shadow_mode": False,
                }
                try:
                    for key, value in alpaca_ctx.items():
                        if key in alpaca_payload:
                            if isinstance(alpaca_payload[key], bool):
                                alpaca_payload[key] = bool(value)
                            else:
                                alpaca_payload[key] = value or ""
                except Exception:
                    pass

                # Fallback to resolved runtime env when context does not provide
                # Alpaca details (common for the dedicated health server path).
                try:
                    from ai_trading.utils.env import (
                        alpaca_credential_status,
                        get_alpaca_base_url,
                    )

                    has_key, has_secret = alpaca_credential_status()
                    if has_key:
                        alpaca_payload["has_key"] = True
                    if has_secret:
                        alpaca_payload["has_secret"] = True
                    if not alpaca_payload.get("base_url"):
                        alpaca_payload["base_url"] = str(get_alpaca_base_url() or "")
                except Exception:
                    logger.debug("HEALTH_ALPACA_ENV_RESOLVE_FAILED", exc_info=True)

                if alpaca_payload.get("base_url"):
                    alpaca_payload["paper"] = "paper" in str(
                        alpaca_payload["base_url"]
                    ).lower()

                try:
                    from ai_trading.alpaca_api import ALPACA_AVAILABLE as _alpaca_sdk_ok

                    alpaca_payload["sdk_ok"] = bool(_alpaca_sdk_ok)
                except Exception:
                    logger.debug("HEALTH_ALPACA_SDK_RESOLVE_FAILED", exc_info=True)

                try:
                    provider_state = runtime_state.observe_data_provider_state()
                except Exception:
                    provider_state = {}
                try:
                    quote_state = runtime_state.observe_quote_status()
                except Exception:
                    quote_state = {}
                try:
                    broker_state = runtime_state.observe_broker_status()
                except Exception:
                    broker_state = {}

                provider_section = {
                    "primary": provider_state.get("primary"),
                    "active": provider_state.get("active"),
                    "backup": provider_state.get("backup"),
                    "using_backup": bool(provider_state.get("using_backup")),
                    "reason": provider_state.get("reason"),
                    "updated": provider_state.get("updated"),
                    "cooldown_seconds_remaining": provider_state.get("cooldown_sec"),
                    "status": provider_state.get("status"),
                    "consecutive_failures": provider_state.get("consecutive_failures"),
                    "last_error_at": provider_state.get("last_error_at"),
                    "gap_ratio_recent": provider_state.get("gap_ratio_recent"),
                    "quote_fresh_ms": provider_state.get("quote_fresh_ms"),
                    "safe_mode": bool(provider_state.get("safe_mode")),
                }
                gap_ratio_recent = provider_state.get("gap_ratio_recent")
                if gap_ratio_recent is not None:
                    try:
                        provider_section["gap_ratio_pct"] = float(gap_ratio_recent) * 100.0
                    except (TypeError, ValueError):
                        provider_section["gap_ratio_pct"] = None
                broker_section = {
                    "connected": bool(broker_state.get("connected")),
                    "status": broker_state.get("status") or ("reachable" if broker_state.get("connected") else "unreachable"),
                    "latency_ms": broker_state.get("latency_ms"),
                    "last_error": broker_state.get("last_error"),
                    "last_order_ack_ms": broker_state.get("last_order_ack_ms"),
                }

                provider_status = provider_section.get("status")
                provider_disabled = provider_status in {"down", "disabled"}
                degraded = provider_disabled or provider_section.get("using_backup") or (
                    provider_status not in {None, "healthy"}
                )
                if broker_section["status"] == "unreachable":
                    degraded = True

                ok = ok and not provider_disabled and broker_section["status"] != "unreachable"

                payload = {
                    "ok": bool(ok),
                    "alpaca": alpaca_payload,
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "service": service_name,
                    "primary_data_provider": provider_section,
                    "provider_state": provider_state,
                    "fallback_active": bool(provider_section.get("using_backup")),
                    "quotes_status": quote_state,
                    "broker_connectivity": broker_section,
                    "status": "degraded" if degraded else "healthy",
                    "gap_ratio_recent": provider_section.get("gap_ratio_recent"),
                    "quote_fresh_ms": provider_section.get("quote_fresh_ms"),
                    "safe_mode": provider_section.get("safe_mode"),
                    "cooldown_seconds_remaining": provider_section.get("cooldown_seconds_remaining"),
                }
                reason_text = provider_section.get("reason")
                if degraded and reason_text:
                    payload["reason"] = reason_text
                if err:
                    payload["error"] = err
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "HEALTH_ENDPOINT_ERROR",
                    extra={"error": str(exc)},
                )
                payload = _emit_response(
                    {
                        "ok": False,
                        "service": "ai-trading",
                        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                        "status": "degraded",
                        "error": str(exc) or exc.__class__.__name__,
                    },
                    status=500,
                )
                return payload
            return _emit_response(payload, status=200)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run the health check web server.

        Any missing context attributes (``host``/``port``) are guarded to
        prevent ``AttributeError`` exceptions during startup.
        """

        host = self._get_ctx_attr("host", "0.0.0.0")
        port = int(self._get_ctx_attr("port", 9001))
        try:
            self.app.run(host=host, port=port)
        except OSError as exc:
            logger.warning(
                "HEALTHCHECK_PORT_CONFLICT",
                extra={"host": host, "port": port, "error": str(exc)},
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "HEALTH_SERVER_START_FAILED",
                extra={"host": host, "port": port, "error": str(exc)},
            )


__all__ = ["HealthCheck"]
logger = get_logger(__name__)
