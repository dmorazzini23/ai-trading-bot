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
from typing import Callable, Dict, Tuple

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
        self.consecutive_switches_by_provider: dict[str, int] = defaultdict(int)
        self._last_switch_time: dict[str, datetime] = {}
        self._alert_cooldown_until: dict[str, datetime | None] = {}
        self._switchover_disable_counts: dict[str, int] = defaultdict(int)
        self._current_switch_cooldowns: dict[str, float] = defaultdict(lambda: float(cooldown))
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
        self._pair_states: Dict[Tuple[str, str], dict[str, object]] = {}

    def register_disable_callback(self, provider: str, cb: Callable[[timedelta], None]) -> None:
        """Register ``cb`` to disable ``provider`` for a duration.

        The callback receives the cooldown period as a ``timedelta`` so the
        caller can implement provider-specific disabling logic.
        """

        self._callbacks[provider] = cb

    def record_failure(
        self,
        provider: str,
        reason: str,
        error: str | None = None,
        *,
        exception: Exception | None = None,
        retry_after: float | None = None,
    ) -> None:
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
        exception:
            Optional exception instance for structured diagnostics. When
            supplied the exception type and representation are captured in the
            diagnostic log event and alert metadata.
        retry_after:
            Optional retry hint (in seconds) returned by the provider. The value
            is emitted with the diagnostics to highlight throttling or
            backoff-oriented failures.
        """

        count = self.fail_counts[provider] + 1
        self.fail_counts[provider] = count
        diagnostics: dict[str, object] = {
            "provider": provider,
            "reason": reason,
            "count": count,
            "threshold": self.threshold,
            "backoff_factor": self.backoff_factor,
            "max_cooldown": self.max_cooldown,
        }
        if retry_after is not None:
            diagnostics["retry_after"] = retry_after
        if error:
            diagnostics["error"] = error
        if exception is not None:
            diagnostics["exception_type"] = type(exception).__name__
            diagnostics["exception_repr"] = repr(exception)
        disable_count = self.disable_counts.get(provider, 0)
        projected_cooldown = min(
            self.cooldown * (self.backoff_factor ** disable_count),
            self.max_cooldown,
        )
        diagnostics["projected_cooldown"] = projected_cooldown
        logger.warning("DATA_PROVIDER_FAILURE_DIAGNOSTIC", extra=diagnostics)
        if count >= self.threshold:
            extra = {"provider": provider, "reason": reason, "count": count}
            if error:
                extra["error"] = error
            if exception is not None:
                extra["exception_type"] = type(exception).__name__
                extra["exception_repr"] = repr(exception)
            if retry_after is not None:
                extra["retry_after"] = retry_after
            logger.error("DATA_PROVIDER_FAILURE", extra=extra)
            try:
                metadata = {"reason": reason, "failures": count}
                if error:
                    metadata["error"] = error
                if exception is not None:
                    metadata["exception_type"] = type(exception).__name__
                if retry_after is not None:
                    metadata["retry_after"] = retry_after
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
        self.consecutive_switches_by_provider.pop(provider, None)
        self._switchover_disable_counts.pop(provider, None)
        self._current_switch_cooldowns.pop(provider, None)
        self._alert_cooldown_until.pop(provider, None)
        self._last_switch_time.pop(provider, None)
        if "_" not in provider:
            # Reset feed-specific keys when a base provider recovers
            to_clear = [key for key in self.consecutive_switches_by_provider if key.startswith(f"{provider}_")]
            for key in to_clear:
                self.consecutive_switches_by_provider.pop(key, None)
                self._switchover_disable_counts.pop(key, None)
                self._current_switch_cooldowns.pop(key, None)
                self._alert_cooldown_until.pop(key, None)
                self._last_switch_time.pop(key, None)
            disable_clear = [key for key in self.disabled_until if key.startswith(f"{provider}_")]
            for key in disable_clear:
                self.disabled_until.pop(key, None)
                self.disabled_since.pop(key, None)
                try:
                    provider_disabled.labels(provider=key).set(0)
                except Exception:  # pragma: no cover - defensive
                    pass
        if not self.consecutive_switches_by_provider:
            self.consecutive_switches = 0
        self.disabled_until.pop(provider, None)
        self.disabled_since.pop(provider, None)
        try:
            provider_disabled.labels(provider=provider).set(0)
        except Exception:  # pragma: no cover - defensive
            pass

    def record_switchover(self, from_provider: str, to_provider: str) -> None:
        """Record a switchover from one provider to another.

        The call increments an in-memory counter and emits an INFO log with the
        running count so operators can diagnose frequent provider churn. It also
        tracks consecutive switchovers and raises an alert when a threshold is
        exceeded.
        """

        now = datetime.now(UTC)
        last = self._last_switch_time.get(from_provider)
        cooldown_window = self._current_switch_cooldowns.get(from_provider, float(self.cooldown))
        if last:
            elapsed = (now - last).total_seconds()
            if elapsed >= cooldown_window:
                self.consecutive_switches_by_provider.pop(from_provider, None)
                self._switchover_disable_counts.pop(from_provider, None)
                self._current_switch_cooldowns[from_provider] = float(self.cooldown)
                self._alert_cooldown_until.pop(from_provider, None)
                if not self.consecutive_switches_by_provider:
                    self.consecutive_switches = 0
        self._last_switch_time[from_provider] = now
        count = self._switchover_disable_counts[from_provider] + 1
        self._switchover_disable_counts[from_provider] = count
        cooldown_window = min(
            self.cooldown * (self.backoff_factor ** (count - 1)),
            self.max_cooldown,
        )
        self._current_switch_cooldowns[from_provider] = cooldown_window
        key = (from_provider, to_provider)
        self.switch_counts[key] += 1
        streak = self.consecutive_switches_by_provider[from_provider] + 1
        self.consecutive_switches_by_provider[from_provider] = streak
        self.consecutive_switches = streak
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
        cooldown_until = self._alert_cooldown_until.get(from_provider)
        if streak >= self.switchover_threshold and (cooldown_until is None or now >= cooldown_until):
            logger.warning(
                "PRIMARY_DATA_FEED_UNAVAILABLE",
                extra={
                    "from_provider": from_provider,
                    "to_provider": to_provider,
                    "consecutive": streak,
                },
            )
            try:
                self.alert_manager.create_alert(
                    AlertType.SYSTEM,
                    AlertSeverity.WARNING,
                    "Consecutive provider switchovers",
                    metadata={"count": streak},
                )
            except Exception:  # pragma: no cover - alerting best effort
                logger.exception("ALERT_FAILURE", extra={"provider": to_provider})
            self._alert_cooldown_until[from_provider] = now + timedelta(seconds=cooldown_window)
            # Back off the failing provider more aggressively to avoid thrashing
            try:
                self.disable(from_provider, duration=self.max_cooldown)
            except Exception:  # pragma: no cover - defensive disable
                logger.exception(
                    "PROVIDER_DISABLE_FAILED",
                    extra={"provider": from_provider},
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
        try:
            from ai_trading.data.fetch.metrics import register_provider_disable
            register_provider_disable(provider)
        except Exception:  # pragma: no cover - defensive
            pass
        provider_disabled.labels(provider=provider).set(1)
        logger.warning(
            "DATA_PROVIDER_DISABLED",
            extra={
                "provider": provider,
                "cooldown": cooldown_s,
                "disable_count": count,
                "backoff_factor": self.backoff_factor,
                "max_cooldown": self.max_cooldown,
            },
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

    def active_provider(self, primary: str, backup: str) -> str:
        """Return the currently active provider for ``(primary, backup)``."""

        key = (primary, backup)
        state = self._pair_states.get(key)
        if state is None:
            state = {"active": primary, "last_switch": None, "consecutive_passes": 0}
            self._pair_states[key] = state
        return str(state.get("active", primary))

    def update_data_health(
        self,
        primary: str,
        backup: str,
        *,
        healthy: bool,
        reason: str,
    ) -> str:
        """Update health state and potentially switch providers.

        Returns the provider that should remain active after evaluating health.
        """

        cooldown_default = int(get_env("DATA_COOLDOWN_SECONDS", "120", cast=int))
        key = (primary, backup)
        state = self._pair_states.setdefault(
            key,
            {"active": primary, "last_switch": None, "consecutive_passes": 0},
        )
        now = datetime.now(UTC)
        active = str(state.get("active", primary))
        last_switch = state.get("last_switch")
        consecutive = int(state.get("consecutive_passes", 0))

        if healthy:
            consecutive += 1
            state["consecutive_passes"] = consecutive
            if active == backup:
                if consecutive >= 2:
                    cooldown_seconds = max(0, int(state.get("cooldown", cooldown_default)))
                    last_switch_dt = last_switch if isinstance(last_switch, datetime) else now
                    elapsed = (now - last_switch_dt).total_seconds()
                    if elapsed >= cooldown_seconds:
                        state["active"] = primary
                        state["last_switch"] = now
                        state["consecutive_passes"] = 0
                        state["cooldown"] = cooldown_default
                        logger.info(
                            "DATA_PROVIDER_SWITCHOVER | from=%s to=%s reason=%s",
                            backup,
                            primary,
                            reason or "recovered",
                        )
                        return primary
                    logger.info(
                        "DATA_PROVIDER_STAY | provider=%s reason=cooldown_active",
                        backup,
                    )
                    return backup
                logger.info(
                    "DATA_PROVIDER_STAY | provider=%s reason=insufficient_health_passes",
                    backup,
                )
                return backup
            logger.info(
                "DATA_PROVIDER_STAY | provider=%s reason=healthy",
                active,
            )
            return active

        # Unhealthy path
        state["consecutive_passes"] = 0
        if active != backup:
            state["active"] = backup
            state["last_switch"] = now
            state["cooldown"] = cooldown_default
            logger.info(
                "DATA_PROVIDER_SWITCHOVER | from=%s to=%s reason=%s",
                active,
                backup,
                reason or "unhealthy",
            )
            return backup
        logger.info(
            "DATA_PROVIDER_STAY | provider=%s reason=unhealthy",
            backup,
        )
        return backup


provider_monitor = ProviderMonitor()

__all__ = ["provider_monitor", "ProviderMonitor"]
