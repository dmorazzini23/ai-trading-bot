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

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger
from ai_trading.monitoring.alerts import AlertManager, AlertSeverity, AlertType
from ai_trading.data.metrics import (
    provider_disabled,
    provider_disable_total,
    provider_disable_duration_seconds,
    provider_failure_duration_seconds,
)


logger = get_logger(__name__)


class ProviderMonitor:
    """Simple failure counter with alerting and disable hooks."""

    def __init__(
        self,
        *,
        threshold: int = 3,
        cooldown: int = 300,
        alert_manager: AlertManager | None = None,
        switchover_threshold: int = 5,
        backoff_factor: float | None = None,
        max_cooldown: int | None = None,
    ) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self.alert_manager = alert_manager or AlertManager()
        self.fail_counts: dict[str, int] = defaultdict(int)
        self.disabled_until: dict[str, datetime] = {}
        self.disabled_since: dict[str, datetime] = {}
        self.disable_counts: dict[str, int] = defaultdict(int)
        self.outage_start: dict[str, datetime] = {}
        self._callbacks: dict[str, Callable[[timedelta], None]] = {}
        # Track provider switchovers for diagnostics
        self.switch_counts: dict[tuple[str, str], int] = defaultdict(int)
        self.consecutive_switches = 0
        self._last_switch_time: datetime | None = None
        self._alert_cooldown_until: datetime | None = None
        self._switchover_disable_count = 0
        self._current_switch_cooldown = cooldown
        self.switchover_threshold = switchover_threshold
        self.backoff_factor = (
            backoff_factor
            if backoff_factor is not None
            else float(get_env("DATA_PROVIDER_BACKOFF_FACTOR", "2", cast=float))
        )
        self.max_cooldown = (
            max_cooldown
            if max_cooldown is not None
            else int(get_env("DATA_PROVIDER_MAX_COOLDOWN", "3600", cast=int))
        )

    def register_disable_callback(self, provider: str, cb: Callable[[timedelta], None]) -> None:
        """Register ``cb`` to disable ``provider`` for a duration.

        The callback receives the cooldown period as a ``timedelta`` so the
        caller can implement provider-specific disabling logic.
        """

        self._callbacks[provider] = cb

    def record_failure(self, provider: str, reason: str, error: str | None = None) -> None:
        """Record a failure for ``provider`` and alert on threshold.

        Parameters
        ----------
        provider:
            Name of the failing data provider.
        reason:
            High level classification of the failure (``timeout``,
            ``connection_error`` …).
        error:
            Optional detailed error message derived from the underlying
            exception. When provided it is included in logs and alert metadata
            to aid debugging of provider outages.
        """

        count = self.fail_counts[provider] + 1
        self.fail_counts[provider] = count
        if count >= self.threshold:
            extra = {"provider": provider, "reason": reason, "count": count}
            if error:
                extra["error"] = error
            logger.error("DATA_PROVIDER_FAILURE", extra=extra)
            try:
                metadata = {"reason": reason, "failures": count}
                if error:
                    metadata["error"] = error
                self.alert_manager.create_alert(
                    AlertType.SYSTEM,
                    AlertSeverity.CRITICAL,
                    f"Data provider {provider} failure",
                    metadata=metadata,
                )
            except Exception:  # pragma: no cover - alerting best effort
                logger.exception("ALERT_FAILURE", extra={"provider": provider})
            self.disable(provider)

    def record_success(self, provider: str) -> None:
        """Reset failure counter and backoff state for ``provider``."""

        self.fail_counts.pop(provider, None)
        self.disable_counts.pop(provider, None)
        self.outage_start.pop(provider, None)

    def record_switchover(self, from_provider: str, to_provider: str) -> None:
        """Record a switchover from one provider to another.

        The call increments an in-memory counter and emits an INFO log with the
        running count so operators can diagnose frequent provider churn. It also
        tracks consecutive switchovers and raises an alert when a threshold is
        exceeded.
        """

        now = datetime.now(UTC)
        if self._last_switch_time:
            elapsed = (now - self._last_switch_time).total_seconds()
            if elapsed >= self._current_switch_cooldown:
                self.consecutive_switches = 0
                self._switchover_disable_count = 0
                self._current_switch_cooldown = self.cooldown
                self._alert_cooldown_until = None
        self._last_switch_time = now
        self._switchover_disable_count += 1
        self._current_switch_cooldown = min(
            self.cooldown * (self.backoff_factor ** (self._switchover_disable_count - 1)),
            self.max_cooldown,
        )
        key = (from_provider, to_provider)
        self.switch_counts[key] += 1
        self.consecutive_switches += 1
        logger.info(
            "DATA_PROVIDER_SWITCHOVER",
            extra={
                "from_provider": from_provider,
                "to_provider": to_provider,
                "count": self.switch_counts[key],
            },
        )
        disabled_since = self.disabled_since.get(from_provider)
        if disabled_since:
            duration = (now - disabled_since).total_seconds()
            provider_failure_duration_seconds.labels(provider=from_provider).inc(duration)
            logger.info(
                "DATA_PROVIDER_FAILURE_DURATION",
                extra={"provider": from_provider, "duration": duration},
            )
        if (
            self.consecutive_switches >= self.switchover_threshold
            and (not self._alert_cooldown_until or now >= self._alert_cooldown_until)
        ):
            try:
                self.alert_manager.create_alert(
                    AlertType.SYSTEM,
                    AlertSeverity.WARNING,
                    "Consecutive provider switchovers",
                    metadata={"count": self.consecutive_switches},
                )
            except Exception:  # pragma: no cover - alerting best effort
                logger.exception("ALERT_FAILURE", extra={"provider": to_provider})
            self._alert_cooldown_until = now + timedelta(
                seconds=self._current_switch_cooldown
            )

    def disable(self, provider: str, *, duration: float | None = None) -> None:
        """Disable ``provider`` for ``duration`` seconds with exponential backoff.

        When ``duration`` is ``None`` the base cooldown is used and scaled by
        ``backoff_factor`` for consecutive disables. The cooldown is capped by
        ``max_cooldown`` so a flapping provider eventually gets a longer
        recovery window but never exceeds the configured limit.
        """

        now = datetime.now(UTC)
        count = self.disable_counts[provider] + 1
        self.disable_counts[provider] = count
        self.outage_start.setdefault(provider, now)
        if duration is None:
            duration = self.cooldown * (self.backoff_factor ** (count - 1))
        cooldown_s = min(duration, self.max_cooldown)
        until = now + timedelta(seconds=cooldown_s)
        self.disabled_until[provider] = until
        self.disabled_since[provider] = now
        provider_disable_total.labels(provider=provider).inc()
        provider_disabled.labels(provider=provider).set(1)
        logger.warning(
            "DATA_PROVIDER_DISABLED",
            extra={"provider": provider, "cooldown": cooldown_s, "disable_count": count},
        )
        cb = self._callbacks.get(provider)
        if cb:
            try:
                cb(timedelta(seconds=cooldown_s))
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "PROVIDER_DISABLE_CALLBACK_ERROR", extra={"provider": provider}
                )

    def is_disabled(self, provider: str) -> bool:
        """Return ``True`` if ``provider`` is currently disabled."""

        until = self.disabled_until.get(provider)
        if until and datetime.now(UTC) < until:
            return True
        if provider in self.disabled_until:
            self.disabled_until.pop(provider, None)
            provider_disabled.labels(provider=provider).set(0)
            since = self.disabled_since.pop(provider, None)
            if since:
                duration = (datetime.now(UTC) - since).total_seconds()
                provider_disable_duration_seconds.labels(provider=provider).inc(duration)
            total_count = self.disable_counts.get(provider, 0)
            start = self.outage_start.pop(provider, since or datetime.now(UTC))
            outage_dur = (datetime.now(UTC) - start).total_seconds()
            if total_count:
                try:
                    self.alert_manager.create_alert(
                        AlertType.SYSTEM,
                        AlertSeverity.WARNING,
                        f"Data provider {provider} restored",
                        metadata={
                            "duration": round(outage_dur, 2),
                            "disable_count": total_count,
                        },
                    )
                except Exception:  # pragma: no cover - alerting best effort
                    logger.exception("ALERT_FAILURE", extra={"provider": provider})
            logger.info(
                "DATA_PROVIDER_RECOVERED",
                extra={
                    "provider": provider,
                    "duration": outage_dur,
                    "disable_count": total_count,
                },
            )
            self.disable_counts.pop(provider, None)
        return False


provider_monitor = ProviderMonitor()

__all__ = ["provider_monitor", "ProviderMonitor"]
