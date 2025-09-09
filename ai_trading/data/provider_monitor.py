from __future__ import annotations

"""Monitoring utilities for external data providers.

This module tracks repeated provider failures (authentication errors,
rate limits, network timeouts) and triggers alerts when a provider is
degraded. When failures exceed a configurable threshold the provider is
temporarily disabled and downstream code can fall back to secondary
providers.

The monitor integrates with :mod:`ai_trading.monitoring.alerts` so
production deployments receive notifications about outages.
"""

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Callable

from ai_trading.logging import get_logger
from ai_trading.monitoring.alerts import AlertManager, AlertSeverity, AlertType


logger = get_logger(__name__)


class ProviderMonitor:
    """Simple failure counter with alerting and disable hooks."""

    def __init__(
        self,
        *,
        threshold: int = 3,
        cooldown: int = 300,
        alert_manager: AlertManager | None = None,
    ) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self.alert_manager = alert_manager or AlertManager()
        self.fail_counts: dict[str, int] = defaultdict(int)
        self.disabled_until: dict[str, datetime] = {}
        self._callbacks: dict[str, Callable[[timedelta], None]] = {}

    def register_disable_callback(
        self, provider: str, cb: Callable[[timedelta], None]
    ) -> None:
        """Register ``cb`` to disable ``provider`` for a duration.

        The callback receives the cooldown period as a ``timedelta`` so the
        caller can implement provider-specific disabling logic.
        """

        self._callbacks[provider] = cb

    def record_failure(self, provider: str, reason: str) -> None:
        """Record a failure for ``provider`` and alert on threshold."""

        count = self.fail_counts[provider] + 1
        self.fail_counts[provider] = count
        if count >= self.threshold:
            logger.error(
                "DATA_PROVIDER_FAILURE",
                extra={"provider": provider, "reason": reason, "count": count},
            )
            try:
                self.alert_manager.create_alert(
                    AlertType.SYSTEM,
                    AlertSeverity.CRITICAL,
                    f"Data provider {provider} failure",
                    metadata={"reason": reason, "failures": count},
                )
            except Exception:  # pragma: no cover - alerting best effort
                logger.exception("ALERT_FAILURE", extra={"provider": provider})
            self.disable(provider)

    def record_success(self, provider: str) -> None:
        """Reset failure counter for ``provider``."""

        self.fail_counts.pop(provider, None)

    def disable(self, provider: str) -> None:
        """Disable ``provider`` for the configured cooldown period."""

        until = datetime.now(UTC) + timedelta(seconds=self.cooldown)
        self.disabled_until[provider] = until
        cb = self._callbacks.get(provider)
        if cb:
            try:
                cb(timedelta(seconds=self.cooldown))
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "PROVIDER_DISABLE_CALLBACK_ERROR", extra={"provider": provider}
                )

    def is_disabled(self, provider: str) -> bool:
        """Return ``True`` if ``provider`` is currently disabled."""

        until = self.disabled_until.get(provider)
        return bool(until and datetime.now(UTC) < until)


provider_monitor = ProviderMonitor()

__all__ = ["provider_monitor", "ProviderMonitor"]

