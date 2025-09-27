from __future__ import annotations

"""Lightweight health check HTTP server utilities."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

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
        self.app.config.update(self.config)

        self._register_routes()

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

        @self.app.route("/healthz")
        def _healthz() -> Any:  # pragma: no cover - simple glue
            errors: list[dict[str, str]] = []
            ok = True
            try:
                service_name = self._get_ctx_attr("service", "ai-trading")
            except ImportError as exc:  # pragma: no cover - defensive
                ok = False
                service_name = "ai-trading"
                errors.append({
                    "type": "ImportError",
                    "detail": str(exc),
                })
            except Exception as exc:  # pragma: no cover - defensive
                ok = False
                service_name = "ai-trading"
                errors.append({
                    "type": exc.__class__.__name__,
                    "detail": str(exc),
                })

            payload = {
                "ok": ok,
                "errors": errors,
                "ts": datetime.now(UTC).isoformat(),
                "service": service_name,
            }
            return jsonify(payload)

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
        self.app.run(host=host, port=port)


__all__ = ["HealthCheck"]
