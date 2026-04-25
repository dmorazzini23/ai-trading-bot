from __future__ import annotations

"""Lightweight health check HTTP server utilities."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from ai_trading.app import build_standalone_healthcheck_app, run_standalone_healthcheck_app
from ai_trading.logging import get_logger
from flask import Flask


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
        self.config = dict(self.config or {})
        service_name = str(self._get_ctx_attr("service", "ai-trading") or "ai-trading")
        ctx_alpaca = self._get_ctx_attr("alpaca", None)
        alpaca_context = dict(ctx_alpaca) if isinstance(ctx_alpaca, Mapping) else None
        self.app = build_standalone_healthcheck_app(
            fail_fast_env=False,
            service_name=service_name,
            alpaca_context=alpaca_context,
            force_ok_for_pytest=False,
            healthy_status_mode="healthy",
        )
        if self.config:
            self.app.config.update(self.config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_ctx_attr(self, name: str, default: Any = None) -> Any:
        """Safely fetch an attribute from the provided context object."""
        if self.ctx is None:
            return default
        return getattr(self.ctx, name, default)

    def run(self) -> None:
        """Run the health check web server.

        Any missing context attributes (``host``/``port``) are guarded to
        prevent ``AttributeError`` exceptions during startup.
        """

        host = self._get_ctx_attr("host", "0.0.0.0")
        port = int(self._get_ctx_attr("port", 8081))
        run_standalone_healthcheck_app(self.app, host=host, port=port, logger=logger)


__all__ = ["HealthCheck"]
logger = get_logger(__name__)
