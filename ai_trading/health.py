from __future__ import annotations

"""Lightweight health check HTTP server utilities."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from ai_trading.health_payload import (
    build_canonical_healthz_payload,
    build_health_exception_payload,
)
from ai_trading.logging import get_logger
from ai_trading.app import _install_route_tracker, _ensure_test_client

try:  # pragma: no cover - optional dependency
    from flask import Flask, jsonify
except Exception:  # pragma: no cover - stub for tests when flask missing
    class Flask:  # type: ignore
        def __init__(self, *a: Any, **k: Any) -> None:
            self.config: dict[str, Any] = {}

        def route(self, *a: Any, **k: Any) -> Any:
            def deco(func: Any) -> Any:
                return func
            return deco

        def run(self, *a: Any, **k: Any) -> None:
            return None

    def jsonify(payload: Any) -> Any:  # type: ignore
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

        def _emit_response(body: dict[str, Any], status: int = 200) -> Any:
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
                err: str | None = None
                try:
                    service_name = self._get_ctx_attr("service", "ai-trading")
                except Exception as exc:  # pragma: no cover - defensive
                    service_name = "ai-trading"
                    err = str(exc) or exc.__class__.__name__

                ctx_alpaca = self._get_ctx_attr("alpaca", None)
                alpaca_ctx = dict(ctx_alpaca) if isinstance(ctx_alpaca, Mapping) else None
                payload = build_canonical_healthz_payload(
                    service_name=str(service_name or "ai-trading"),
                    force_ok_for_pytest=False,
                    healthy_status_mode="healthy",
                    ok_mode="connectivity",
                    alpaca_context=alpaca_ctx,
                    error=err,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "HEALTH_ENDPOINT_ERROR",
                    extra={"error": str(exc)},
                )
                payload = _emit_response(
                    build_health_exception_payload(exc, service_name="ai-trading"),
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
