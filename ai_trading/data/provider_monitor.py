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

import logging
import os
import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, Mapping, MutableMapping, Tuple

from ai_trading.config.management import get_env
from ai_trading.config.settings import get_settings
from ai_trading.logging import (
    get_logger,
    log_throttled_event,
    provider_log_deduper,
    record_provider_log_suppressed,
)
from ai_trading.monitoring.alerts import AlertManager, AlertSeverity, AlertType
from ai_trading.data.metrics import (
    provider_disabled,
    provider_disable_total,
    provider_disable_duration_seconds,
    provider_failure_duration_seconds,
)
from ai_trading.utils.time import monotonic_time


logger = get_logger(__name__)

_PROVIDER_CONFIG_LOGGED: bool = False
_FIRST_DECISION = True

def _resolve_switch_cooldown_seconds() -> int:
    """Return the minimum cooldown before switching back to the primary feed."""

    candidate: int | None = None
    try:
        settings = get_settings()
    except Exception:
        settings = None
    if settings is not None:
        for attr in ("provider_switch_cooldown_sec", "provider_switch_cooldown_seconds"):
            value = getattr(settings, attr, None)
            if value is not None:
                try:
                    candidate = int(value)
                except (TypeError, ValueError):
                    candidate = None
                else:
                    break
    if candidate is None:
        try:
            candidate = get_env("AI_TRADING_PROVIDER_SWITCH_COOLDOWN_SEC", None, cast=int)
        except Exception:
            candidate = None
    if candidate is None:
        raw = os.getenv("AI_TRADING_PROVIDER_SWITCH_COOLDOWN_SEC", "").strip()
        if raw:
            try:
                candidate = int(raw)
            except Exception:
                candidate = None
    if candidate is None:
        candidate = 900
    try:
        return max(int(candidate), 0)
    except Exception:
        return 900


def _resolve_health_passes_required() -> int:
    """Return the number of consecutive healthy passes required for recovery."""

    candidate: int | None = None
    try:
        settings = get_settings()
    except Exception:
        settings = None
    if settings is not None:
        for attr in ("provider_health_passes_required", "provider_health_passes"):
            value = getattr(settings, attr, None)
            if value is not None:
                try:
                    candidate = int(value)
                except (TypeError, ValueError):
                    candidate = None
                else:
                    break
    if candidate is None:
        try:
            candidate = get_env("AI_TRADING_PROVIDER_HEALTH_PASSES_REQUIRED", None, cast=int)
        except Exception:
            candidate = None
    if candidate is None:
        raw = os.getenv("AI_TRADING_PROVIDER_HEALTH_PASSES_REQUIRED", "").strip()
        if raw:
            try:
                candidate = int(raw)
            except Exception:
                candidate = None
    if candidate is None:
        candidate = 4
    try:
        return max(int(candidate), 1)
    except Exception:
        return 4


_MIN_RECOVERY_SECONDS = _resolve_switch_cooldown_seconds()
_MIN_RECOVERY_PASSES = _resolve_health_passes_required()
_DEFAULT_SWITCH_QUIET_SECONDS = 15.0

_HALT_EVENT_WINDOW_SECONDS = 600.0
_SIP_AUTH_FAIL_THRESHOLD = 3
_GAP_EVENT_THRESHOLD = 3
_HALT_SUPPRESS_SECONDS = 60.0

_sip_auth_events: Deque[float] = deque()
_gap_events: Deque[float] = deque()
_last_halt_reason: str | None = None
_last_halt_ts: float = 0.0
_SAFE_MODE_ACTIVE = False
_SAFE_MODE_REASON: str | None = None


def _current_intraday_feed() -> str:
    """Return the configured intraday feed identifier."""

    env_feed = os.getenv("DATA_FEED_INTRADAY")
    if env_feed not in (None, ""):
        normalized = env_feed.strip().lower()
        if normalized:
            return normalized
    try:
        from ai_trading.config import DATA_FEED_INTRADAY as _CFG_FEED  # local import to avoid cycles
    except Exception:
        _CFG_FEED = None
    feed = _CFG_FEED
    if feed in (None, ""):
        try:
            settings = get_settings()
        except Exception:
            settings = None
        if settings is not None:
            feed = getattr(settings, "data_feed_intraday", None) or getattr(settings, "alpaca_data_feed", None)
    if feed in (None, ""):
        feed = os.getenv("ALPACA_DATA_FEED")
    normalized = str(feed or "iex").strip().lower()
    return normalized or "iex"


def _intraday_feed_is_sip() -> bool:
    """Return ``True`` when SIP is the active intraday feed."""

    return _current_intraday_feed() == "sip"


def _resolve_halt_flag_path() -> str:
    """Return the configured halt flag path with sensible fallbacks."""

    try:
        settings = get_settings()
    except Exception:
        settings = None
    if settings is not None:
        path = getattr(settings, "halt_flag_path", None)
        if isinstance(path, str) and path:
            return path
    env_path = os.getenv("AI_TRADING_HALT_FLAG_PATH")
    if env_path:
        return env_path
    return "halt.flag"


def _write_halt_flag(reason: str, metadata: Mapping[str, Any] | None = None) -> None:
    path = _resolve_halt_flag_path()
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem guard
            logger.error(
                "HALT_FLAG_DIR_CREATE_FAILED",
                extra={"path": path, "error": str(exc)},
            )
            return
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"{reason} {datetime.now(UTC).isoformat()}")
    except OSError as exc:  # pragma: no cover - filesystem guard
        logger.error(
            "HALT_FLAG_WRITE_FAILED",
            extra={"path": path, "error": str(exc)},
        )
        return
    payload: dict[str, Any] = {"path": path, "reason": reason}
    if metadata:
        payload.update({k: v for k, v in metadata.items() if k not in payload})
    logger.warning("HALT_FLAG_WRITTEN", extra=payload)


def is_safe_mode_active() -> bool:
    """Return ``True`` if the provider monitor has triggered safe mode."""

    return _SAFE_MODE_ACTIVE


def safe_mode_reason() -> str | None:
    """Return the most recent safe-mode trigger reason, if any."""

    return _SAFE_MODE_REASON


def _trigger_provider_safe_mode(
    reason: str,
    *,
    count: int,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    global _last_halt_reason, _last_halt_ts, _SAFE_MODE_ACTIVE, _SAFE_MODE_REASON

    now = monotonic_time()
    if _last_halt_reason == reason and (now - _last_halt_ts) < _HALT_SUPPRESS_SECONDS:
        return
    _last_halt_reason = reason
    _last_halt_ts = now
    _SAFE_MODE_ACTIVE = True
    _SAFE_MODE_REASON = reason

    metadata_payload: dict[str, Any] = {"provider": "alpaca", "reason": reason, "events": count}
    if metadata:
        metadata_payload.update({k: v for k, v in metadata.items() if k not in metadata_payload})

    monitor = globals().get("provider_monitor")
    alert_manager: AlertManager | None = None
    if monitor is not None:
        alert_manager = getattr(monitor, "alert_manager", None)

    if alert_manager is not None:
        try:
            alert_manager.create_alert(
                AlertType.PROVIDER_OUTAGE,
                AlertSeverity.CRITICAL,
                "Alpaca minute feed outage detected",
                metadata=metadata_payload,
            )
        except Exception:  # pragma: no cover - alerting best effort
            logger.exception(
                "PROVIDER_OUTAGE_ALERT_FAILED",
                extra={"reason": reason},
            )

    logger.error(
        "PROVIDER_SAFE_MODE_TRIGGERED",
        extra=metadata_payload,
    )

    _write_halt_flag(reason, metadata=metadata_payload)

    if monitor is not None:
        for provider in ("alpaca", "alpaca_sip"):
            try:
                monitor.disable(provider)
            except Exception:  # pragma: no cover - disable guard
                logger.exception(
                    "PROVIDER_SAFE_MODE_DISABLE_FAILED",
                    extra={"provider": provider, "reason": reason},
                )


def _record_event(
    bucket: Deque[float],
    *,
    threshold: int,
    reason: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    if reason == "minute_gap" and metadata is not None:
        provider_name = str(metadata.get("provider", "") or "").strip().lower()
        residual_flag = metadata.get("residual_gap")
        if provider_name == "yahoo":
            return
        if residual_flag is False:
            return
    now = monotonic_time()
    bucket.append(now)
    cutoff = now - _HALT_EVENT_WINDOW_SECONDS
    while bucket and bucket[0] < cutoff:
        bucket.popleft()
    count = len(bucket)
    if count >= threshold:
        _trigger_provider_safe_mode(reason, count=count, metadata=metadata)
        bucket.clear()


def record_unauthorized_sip_event(metadata: Mapping[str, Any] | None = None) -> None:
    """Record an UNAUTHORIZED_SIP event for safe-mode tracking."""

    if not _intraday_feed_is_sip():
        return
    _record_event(
        _sip_auth_events,
        threshold=_SIP_AUTH_FAIL_THRESHOLD,
        reason="unauthorized_sip",
        metadata=metadata,
    )


def record_minute_gap_event(metadata: Mapping[str, Any] | None = None) -> None:
    """Record a minute coverage gap event for safe-mode tracking."""

    _record_event(
        _gap_events,
        threshold=_GAP_EVENT_THRESHOLD,
        reason="minute_gap",
        metadata=metadata,
    )


def _resolve_max_cooldown() -> float:
    """Resolve the maximum provider cooldown from settings/env with bounds."""

    candidate: float | None = None
    try:
        settings = get_settings()
    except Exception:
        settings = None
    if settings is not None:
        raw_value = getattr(settings, "provider_max_cooldown_seconds", None)
        try:
            candidate = float(raw_value) if raw_value is not None else None
        except (TypeError, ValueError):  # pragma: no cover - defensive parsing
            candidate = None
    if candidate is None:
        try:
            new_env = get_env("PROVIDER_MAX_COOLDOWN_SECONDS", None, cast=float)
        except Exception:
            new_env = None
        if new_env is not None:
            try:
                candidate = float(new_env)
            except (TypeError, ValueError):  # pragma: no cover - defensive parsing
                candidate = None
    if candidate is None:
        try:
            env_value = get_env("DATA_PROVIDER_MAX_COOLDOWN", None, cast=float)
        except Exception:
            env_value = None
        if env_value is not None:
            try:
                candidate = float(env_value)
            except (TypeError, ValueError):  # pragma: no cover - defensive parsing
                candidate = None
    if candidate is None:
        raw_env = os.getenv("PROVIDER_MAX_COOLDOWN_SECONDS")
        if raw_env is not None:
            try:
                candidate = float(raw_env)
            except (TypeError, ValueError):  # pragma: no cover - defensive parsing
                candidate = None
    if candidate is None:
        raw_env = os.getenv("DATA_PROVIDER_MAX_COOLDOWN")
        if raw_env is not None:
            try:
                candidate = float(raw_env)
            except (TypeError, ValueError):  # pragma: no cover - defensive parsing
                candidate = None
    if candidate is None:
        candidate = 600.0
    candidate = float(candidate)
    if candidate < 60.0:
        candidate = 60.0
    return candidate


def _resolve_switch_quiet_seconds() -> float:
    """Return the configured provider switchover quiet window in seconds."""

    try:
        settings = get_settings()
    except Exception:
        settings = None
    if settings is not None:
        raw_value = getattr(settings, "provider_switch_quiet_seconds", None)
        try:
            value = float(raw_value) if raw_value is not None else None
        except (TypeError, ValueError):  # pragma: no cover - defensive parsing
            value = None
        if value is not None:
            return max(value, 0.0)

    try:
        env_value = get_env("PROVIDER_SWITCH_QUIET_SECONDS", None, cast=float)
    except Exception:
        env_value = None
    if env_value is not None:
        try:
            return max(float(env_value), 0.0)
        except (TypeError, ValueError):  # pragma: no cover - defensive parsing
            pass

    raw_env = os.getenv("PROVIDER_SWITCH_QUIET_SECONDS")
    if raw_env is not None:
        try:
            return max(float(raw_env), 0.0)
        except (TypeError, ValueError):  # pragma: no cover - defensive parsing
            pass

    return _DEFAULT_SWITCH_QUIET_SECONDS


def _logging_dedupe_ttl() -> int:
    try:
        settings = get_settings()
    except Exception:
        return 0
    ttl = getattr(settings, "logging_dedupe_ttl_s", 0)
    try:
        return int(ttl)
    except (TypeError, ValueError):  # pragma: no cover - defensive parsing
        return 0


class ProviderAction(Enum):
    """Decision outcome for a provider health evaluation tick."""

    STAY = "stay"
    SWITCH = "switch"
    DISABLE = "disable"


def _policy_lookup(policy: Mapping[str, Any] | object | None, key: str, default: Any) -> Any:
    """Helper to read ``key`` from ``policy`` regardless of mapping or attribute."""

    if policy is None:
        return default
    if isinstance(policy, Mapping):
        return policy.get(key, default)
    return getattr(policy, key, default)


def decide_provider_action(
    health: Mapping[str, Any] | bool,
    cooldown_ok: bool,
    consecutive_switches: int,
    policy: Mapping[str, Any] | object | None,
    *,
    from_provider: str | None = None,
    to_provider: str | None = None,
    cooldown: int | None = None,
    context: MutableMapping[str, Any] | None = None,
) -> ProviderAction:
    """Return the desired :class:`ProviderAction` for the current tick.

    Parameters
    ----------
    health:
        Either a mapping with contextual flags or a boolean that evaluates to
        ``True`` when the provider is healthy. When a mapping is supplied the
        function looks for ``is_healthy``/``healthy`` and ``using_backup``
        entries.
    cooldown_ok:
        Indicates whether switching providers is allowed because cooldown and
        debounce conditions are satisfied.
    consecutive_switches:
        Current streak of switchovers for the provider pair. Policies may use
        this to decide when to disable a provider.
    policy:
        Optional mapping/object describing heuristics. Supported keys:

        * ``prefer_primary`` (default ``True``) – when healthy, prefer the
          primary provider if recovery conditions are met.
        * ``allow_recovery`` (default ``False``) – whether recovery back to the
          primary provider is permissible this tick.
        * ``disable_after`` – when provided and the provider is unhealthy,
          disable after ``consecutive_switches`` reaches this threshold.

    Returns
    -------
    ProviderAction
        The action that should be taken for this tick.
    """

    if isinstance(health, Mapping):
        is_healthy = bool(health.get("is_healthy", health.get("healthy", False)))
        using_backup = bool(health.get("using_backup", False))
        allow_recovery = bool(health.get("allow_recovery", False))
    else:
        is_healthy = bool(health)
        using_backup = False
        allow_recovery = False

    prefer_primary = bool(_policy_lookup(policy, "prefer_primary", True))
    policy_allow_recovery = bool(_policy_lookup(policy, "allow_recovery", allow_recovery))
    disable_after = _policy_lookup(policy, "disable_after", None)

    if context is not None and "stay_logged" not in context:
        context["stay_logged"] = False

    if not is_healthy:
        if disable_after is not None and consecutive_switches >= int(disable_after):
            return ProviderAction.DISABLE
        action = ProviderAction.STAY if using_backup else ProviderAction.SWITCH
    elif using_backup and prefer_primary and policy_allow_recovery and cooldown_ok:
        action = ProviderAction.SWITCH
    else:
        action = ProviderAction.STAY

    if (
        action is ProviderAction.SWITCH
        and from_provider is not None
        and to_provider is not None
    ):
        from_key = _normalize_provider(from_provider)
        to_key = _normalize_provider(to_provider)
        if from_key == to_key:
            provider_for_log = from_key or to_key or to_provider or from_provider
            record_stay(
                provider=provider_for_log,
                reason="redundant_request",
                cooldown=cooldown if cooldown is not None else _DEFAULT_DECISION_SECS,
            )
            if context is not None:
                context["stay_reason"] = "redundant_request"
                context["stay_logged"] = True
            return ProviderAction.STAY

    return action


def _normalize_provider(name: str) -> str:
    normalized = (name or "").strip().lower().replace("-", "_")
    if normalized == "alpaca_yahoo":
        return "yahoo"
    return normalized or name


def record_stay(*, provider: str, reason: str, cooldown: int) -> None:
    """Log a stay decision for ``provider`` with the supplied ``reason``."""

    normalized = _normalize_provider(provider)
    logger.info(
        "DATA_PROVIDER_STAY | provider=%s reason=%s cooldown=%ss",
        normalized,
        reason,
        cooldown,
    )


_DEFAULT_DECISION_SECS = 120


def _decision_window_seconds() -> int:
    try:
        value = get_env("AI_TRADING_PROVIDER_DECISION_SECS", None, cast=int)
    except Exception:
        value = None
    if value is None:
        try:
            value = int(os.getenv("AI_TRADING_PROVIDER_DECISION_SECS", ""))
        except Exception:  # pragma: no cover - defensive env parsing fallback
            value = None
    try:
        seconds = int(value) if value is not None else _DEFAULT_DECISION_SECS
    except Exception:  # pragma: no cover - defensive conversion
        seconds = _DEFAULT_DECISION_SECS
    return max(seconds, 0)


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
            float(max_cooldown)
            if max_cooldown is not None
            else float(_resolve_max_cooldown())
        )
        self.switch_quiet_seconds = max(float(_resolve_switch_quiet_seconds()), 0.0)
        self._pair_states: Dict[Tuple[str, str], dict[str, object]] = {}
        self._last_switch_logged: tuple[str, str] | None = None
        self._last_switch_ts: float | None = None
        self.decision_window_seconds = _decision_window_seconds()
        global _PROVIDER_CONFIG_LOGGED
        if not _PROVIDER_CONFIG_LOGGED:
            _PROVIDER_CONFIG_LOGGED = True
            try:
                decision_secs = int(self.decision_window_seconds)
            except (TypeError, ValueError):
                decision_secs = 0
            try:
                switch_cd = int(float(self.cooldown))
            except (TypeError, ValueError):
                switch_cd = 0
            try:
                max_cd = int(float(self.max_cooldown))
            except (TypeError, ValueError):
                max_cd = switch_cd
            try:
                logger.info(
                    "PROVIDER_MONITOR_CONFIG",
                    extra={
                        "decision_window_secs": decision_secs,
                        "switch_cooldown_secs": switch_cd,
                        "max_cooldown_secs": max_cd,
                    },
                )
            except Exception:
                pass
        self._last_switchover_provider: str | None = None
        self._last_switchover_ts: float = 0.0
        self._last_switchover_passes: int = 0
        self._last_sip_warn_ts: float = 0.0
        self._pair_switch_history: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)
        self.min_recovery_seconds = _MIN_RECOVERY_SECONDS
        self.recovery_passes_required = _MIN_RECOVERY_PASSES

    def register_disable_callback(self, provider: str, cb: Callable[[timedelta], None]) -> None:
        """Register ``cb`` to disable ``provider`` for a duration.

        The callback receives the cooldown period as a ``timedelta`` so the
        caller can implement provider-specific disabling logic.
        """

        self._callbacks[provider] = cb

    def reset(self) -> None:
        """Reset mutable provider state for tests and controlled call sites."""

        self.fail_counts.clear()
        self.disabled_until.clear()
        self.disabled_since.clear()
        self.disable_counts.clear()
        self.outage_start.clear()
        self.switch_counts.clear()
        self.consecutive_switches = 0
        self.consecutive_switches_by_provider.clear()
        self._last_switch_time.clear()
        self._alert_cooldown_until.clear()
        self._switchover_disable_counts.clear()
        self._current_switch_cooldowns.clear()
        self._pair_states.clear()
        self._last_switch_logged = None
        self._last_switch_ts = None
        self._last_switchover_provider = None
        self._last_switchover_ts = 0.0
        self._last_switchover_passes = 0
        self._last_sip_warn_ts = 0.0
        self._pair_switch_history.clear()
        global _MIN_RECOVERY_SECONDS, _MIN_RECOVERY_PASSES
        _MIN_RECOVERY_SECONDS = _resolve_switch_cooldown_seconds()
        _MIN_RECOVERY_PASSES = _resolve_health_passes_required()
        self.min_recovery_seconds = _MIN_RECOVERY_SECONDS
        self.recovery_passes_required = _MIN_RECOVERY_PASSES

    def _refresh_runtime_limits(self) -> None:
        """Refresh cached cooldown and quiet window values from configuration."""

        configured_max = float(_resolve_max_cooldown())
        if configured_max != self.max_cooldown:
            self.max_cooldown = configured_max
        configured_quiet = max(float(_resolve_switch_quiet_seconds()), 0.0)
        if configured_quiet != self.switch_quiet_seconds:
            self.switch_quiet_seconds = configured_quiet
        configured_recovery = _resolve_switch_cooldown_seconds()
        if configured_recovery != self.min_recovery_seconds:
            global _MIN_RECOVERY_SECONDS
            _MIN_RECOVERY_SECONDS = configured_recovery
            self.min_recovery_seconds = configured_recovery
        configured_passes = _resolve_health_passes_required()
        if configured_passes != self.recovery_passes_required:
            global _MIN_RECOVERY_PASSES
            _MIN_RECOVERY_PASSES = configured_passes
            self.recovery_passes_required = configured_passes

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
            self.cooldown * (self.backoff_factor**disable_count),
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
        if provider.startswith("alpaca"):
            global _SAFE_MODE_ACTIVE, _SAFE_MODE_REASON
            _SAFE_MODE_ACTIVE = False
            _SAFE_MODE_REASON = None

    def _migrate_provider_state(self, old: str, new: str) -> None:
        if not old or old == new:
            return
        for mapping in (
            self._last_switch_time,
            self._current_switch_cooldowns,
            self._switchover_disable_counts,
            self.consecutive_switches_by_provider,
            self._alert_cooldown_until,
            self.disabled_since,
            self.disabled_until,
        ):
            if old in mapping and new not in mapping:
                mapping[new] = mapping.pop(old)
            elif old in mapping:
                mapping.pop(old)

    def _enforce_switchover_quiet_period(
        self,
        from_key: str,
        to_key: str,
        *,
        now_monotonic: float,
        now_wall: datetime,
    ) -> bool:
        self._refresh_runtime_limits()
        window = max(float(self.switch_quiet_seconds), 0.0)
        if window <= 0.0:
            return False
        history = self._pair_switch_history[(from_key, to_key)]
        history.append(now_monotonic)
        cutoff = now_monotonic - window
        while history and history[0] < cutoff:
            history.popleft()
        if len(history) < 3:
            return False
        history.clear()
        window_for_log = int(round(window))
        logger.warning(
            "DATA_PROVIDER_SWITCHOVER_BLOCKED",
            extra={
                "from_provider": from_key,
                "to_provider": to_key,
                "reason": "repeated_switchovers",
                "window_seconds": window_for_log,
            },
        )
        try:
            self.disable(from_key, duration=self.max_cooldown)
        except Exception:  # pragma: no cover - defensive disable
            logger.exception(
                "PROVIDER_DISABLE_FAILED",
                extra={"provider": from_key},
            )
        cooldown_for_log = int(round(self.max_cooldown))
        record_stay(
            provider=to_key,
            reason="repeated_switchovers",
            cooldown=cooldown_for_log,
        )
        self.consecutive_switches_by_provider.pop(from_key, None)
        if not self.consecutive_switches_by_provider:
            self.consecutive_switches = 0
        self._alert_cooldown_until[from_key] = now_wall + timedelta(seconds=self.cooldown)
        self._last_switchover_provider = None
        self._last_switchover_passes = 0
        self._last_switchover_ts = 0.0
        return True

    def record_switchover(self, from_provider: str, to_provider: str) -> ProviderAction | None:
        """Record a switchover from one provider to another.

        The call increments an in-memory counter and emits an INFO log with the
        running count so operators can diagnose frequent provider churn. It also
        tracks consecutive switchovers and raises an alert when a threshold is
        exceeded. When a pair thrashes (three switches within the configured
        quiet window) the switchover is blocked, the provider is disabled for
        the configured maximum cooldown, and :class:`ProviderAction.DISABLE` is
        returned.
        """

        from_key = _normalize_provider(from_provider)
        to_key = _normalize_provider(to_provider)
        if from_key != from_provider:
            self._migrate_provider_state(from_provider, from_key)
        if to_key != to_provider:
            self._migrate_provider_state(to_provider, to_key)
        if not from_key or from_key == to_key:
            stay_provider = to_key or from_key
            logger.info(
                "DATA_PROVIDER_STAY | provider=%s reason=%s cooldown=%ss",
                stay_provider,
                "redundant_request",
                self.cooldown,
            )
            return
        now_wall = time.time()
        if self._last_switchover_provider == from_key:
            elapsed = (
                now_wall - self._last_switchover_ts
                if self._last_switchover_ts
                else float("inf")
            )
            passes = self._last_switchover_passes
            required_passes = max(int(self.recovery_passes_required), 1)
            required_seconds = max(float(self.min_recovery_seconds), 0.0)
            if passes < required_passes or elapsed < required_seconds:
                reason = "insufficient_health_passes" if passes < required_passes else "cooldown_active"
                logger.info(
                    "DATA_PROVIDER_STAY | provider=%s reason=%s cooldown=%ss",
                    from_key,
                    reason,
                    int(max(self.cooldown, required_seconds)),
                )
                return
        now = datetime.now(UTC)
        last = self._last_switch_time.get(from_key)
        cooldown_window = self._current_switch_cooldowns.get(from_key, float(self.cooldown))
        if last:
            elapsed = (now - last).total_seconds()
            if elapsed >= cooldown_window:
                self.consecutive_switches_by_provider.pop(from_key, None)
                self._switchover_disable_counts.pop(from_key, None)
                self._current_switch_cooldowns[from_key] = float(self.cooldown)
                self._alert_cooldown_until.pop(from_key, None)
                if not self.consecutive_switches_by_provider:
                    self.consecutive_switches = 0
        self._last_switch_time[from_key] = now
        now_monotonic = monotonic_time()
        if self._enforce_switchover_quiet_period(
            from_key,
            to_key,
            now_monotonic=now_monotonic,
            now_wall=now,
        ):
            return ProviderAction.DISABLE
        count = self._switchover_disable_counts[from_key] + 1
        self._switchover_disable_counts[from_key] = count
        cooldown_window = min(
            self.cooldown * (self.backoff_factor ** (count - 1)),
            self.max_cooldown,
        )
        self._current_switch_cooldowns[from_key] = cooldown_window
        key = (from_key, to_key)
        if key not in self.switch_counts:
            # migrate existing counter from raw key if present
            raw_key = (from_provider, to_provider)
            if raw_key in self.switch_counts:
                self.switch_counts[key] = self.switch_counts.pop(raw_key)
        self.switch_counts[key] += 1
        streak = self.consecutive_switches_by_provider[from_key] + 1
        self.consecutive_switches_by_provider[from_key] = streak
        self.consecutive_switches = streak
        dedupe_ttl = _logging_dedupe_ttl()
        switchover_key = f"DATA_PROVIDER_SWITCHOVER:{from_key}->{to_key}"
        if not (
            self._last_switch_logged == key
            and self._last_switch_ts is not None
            and now_monotonic - self._last_switch_ts < 1.0
        ):
            if provider_log_deduper.should_log(switchover_key, dedupe_ttl):
                log_throttled_event(
                    logger,
                    "DATA_PROVIDER_SWITCHOVER",
                    level=logging.WARNING,
                    extra={
                        "from_provider": from_key,
                        "to_provider": to_key,
                        "count": self.switch_counts[key],
                    },
                )
            else:
                record_provider_log_suppressed("DATA_PROVIDER_SWITCHOVER")
            self._last_switch_logged = key
            self._last_switch_ts = now_monotonic
        disabled_since = self.disabled_since.get(from_key)
        if disabled_since:
            duration = (now - disabled_since).total_seconds()
            provider_failure_duration_seconds.labels(provider=from_key).inc(duration)
            failure_key = f"DATA_PROVIDER_FAILURE_DURATION:{from_key}"
            if provider_log_deduper.should_log(failure_key, dedupe_ttl):
                logger.info(
                    "DATA_PROVIDER_FAILURE_DURATION",
                    extra={"provider": from_key, "duration": duration},
                )
            else:
                record_provider_log_suppressed("DATA_PROVIDER_FAILURE_DURATION")
        cooldown_until = self._alert_cooldown_until.get(from_key)
        if streak >= self.switchover_threshold and (cooldown_until is None or now >= cooldown_until):
            logger.warning(
                "PRIMARY_DATA_FEED_UNAVAILABLE",
                extra={
                    "from_provider": from_key,
                    "to_provider": to_key,
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
                logger.exception("ALERT_FAILURE", extra={"provider": to_key})
            self._alert_cooldown_until[from_key] = now + timedelta(seconds=cooldown_window)
            # Back off the failing provider more aggressively to avoid thrashing
            try:
                self.disable(from_key, duration=self.max_cooldown)
            except Exception:  # pragma: no cover - defensive disable
                logger.exception(
                    "PROVIDER_DISABLE_FAILED",
                    extra={"provider": from_key},
                )
        if self._last_switchover_provider == from_key:
            self._last_switchover_provider = None
            self._last_switchover_passes = 0
            self._last_switchover_ts = 0.0
        else:
            self._last_switchover_provider = to_key
            self._last_switchover_passes = 0
            self._last_switchover_ts = now_wall

    def record_health_pass(self, success: bool, provider: str | None = None) -> None:
        """Record the outcome of a provider health probe for switchover dwell."""

        target = provider or self._last_switchover_provider
        if not target:
            return
        normalized = _normalize_provider(target)
        if normalized != self._last_switchover_provider:
            if not success and self._last_switchover_provider == normalized:
                self._last_switchover_passes = 0
            return
        if success:
            self._last_switchover_passes = min(self._last_switchover_passes + 1, 10_000)
        else:
            self._last_switchover_passes = 0

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
                logger.exception("PROVIDER_DISABLE_CALLBACK_ERROR", extra={"provider": provider})

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

    def _sip_switch_allowed(self, provider: str) -> bool:
        """Return ``True`` if switching to ``provider`` is permitted."""

        normalized = _normalize_provider(provider or "")
        if not normalized.endswith("_sip"):
            return True

        strict_enabled = False
        try:
            from ai_trading.data import fetch
        except Exception:  # pragma: no cover - defensive import guard
            fetch = None
        if fetch is not None:
            strict_checker = getattr(fetch, "_strict_data_gating_enabled", None)
            if callable(strict_checker):
                try:
                    strict_enabled = bool(strict_checker())
                except Exception:  # pragma: no cover - defensive gating check
                    strict_enabled = False
        env_allowed = os.getenv("ALPACA_SIP_ENABLED", "0") in {"1", "true", "True"}
        allowed = strict_enabled and env_allowed
        if not allowed:
            now_monotonic = monotonic_time()
            cooldown_window = max(float(self.cooldown), 1.0)
            if (
                self._last_sip_warn_ts <= 0.0
                or now_monotonic - self._last_sip_warn_ts >= cooldown_window
            ):
                logger.warning(
                    "UNAUTHORIZED_SIP",
                    extra={
                        "provider": normalized,
                        "feed": "sip",
                        "status": "sip_disabled",
                        "reason": "sip_disabled",
                    },
                )
                self._last_sip_warn_ts = now_monotonic
        return allowed

    def update_data_health(
        self,
        primary: str,
        backup: str,
        *,
        healthy: bool,
        reason: str,
        severity: str | None = None,
    ) -> str:
        """Update health state and potentially switch providers.

        Returns the provider that should remain active after evaluating health.
        """

        cooldown_default = int(get_env("DATA_COOLDOWN_SECONDS", "120", cast=int))
        key = (primary, backup)
        state = self._pair_states.setdefault(
            key,
            {
                "active": primary,
                "last_switch": None,
                "consecutive_passes": 0,
                "decision_until": None,
                "decision_provider": primary,
                "decision_severity": "good",
            },
        )
        global _FIRST_DECISION
        if _FIRST_DECISION:
            _FIRST_DECISION = False
            preferred = os.getenv("DATA_PROVIDER", "").strip()
            preferred_norm = _normalize_provider(preferred) if preferred else ""
            primary_norm = _normalize_provider(primary)
            fail_count = self.fail_counts.get(primary, 0)
            if fail_count <= 0:
                fail_count = self.fail_counts.get(primary_norm, 0)
            if (
                preferred_norm
                and preferred_norm == primary_norm
                and not self.is_disabled(primary)
                and fail_count <= 0
            ):
                state["active"] = primary
                state["decision_provider"] = primary
                state["decision_severity"] = "good"
                state["decision_until"] = None
        now = datetime.now(UTC)
        active = str(state.get("active", primary))
        self.record_health_pass(bool(healthy), provider=active)
        last_switch = state.get("last_switch")
        consecutive = int(state.get("consecutive_passes", 0))
        decision_until = state.get("decision_until")
        decision_provider = str(state.get("decision_provider", active))
        decision_severity = str(state.get("decision_severity", "good"))
        window_seconds = max(int(self.decision_window_seconds), 0)
        window_active = False
        if isinstance(decision_until, datetime) and window_seconds > 0:
            if now < decision_until:
                window_active = True
            else:
                state["decision_until"] = None
                decision_until = None
        normalized_severity = (severity or ("good" if healthy else "degraded")).strip().lower()
        if normalized_severity not in {"good", "degraded", "hard_fail"}:
            normalized_severity = "good" if healthy else "degraded"

        using_backup = active == backup
        cooldown_seconds = max(0, int(state.get("cooldown", cooldown_default)))
        last_switch_dt = last_switch if isinstance(last_switch, datetime) else now
        allow_recovery = False
        cooldown_ok = False
        stay_reason = "healthy" if healthy else (reason or "unhealthy")
        switch_reason = reason or ("recovered" if healthy else "unhealthy")
        window_locked = window_active and decision_provider == active
        hard_fail = normalized_severity == "hard_fail" and not healthy
        if window_locked and not hard_fail:
            state["decision_provider"] = decision_provider
            state["decision_severity"] = decision_severity or normalized_severity
            stay_reason = "decision_window_active"
            cooldown_ok = False
            allow_recovery = allow_recovery or False

        if healthy:
            consecutive += 1
            state["consecutive_passes"] = consecutive
            if using_backup:
                allow_recovery = consecutive >= max(int(self.recovery_passes_required), 1)
                elapsed = (now - last_switch_dt).total_seconds()
                required_stay = max(cooldown_seconds, int(max(float(self.min_recovery_seconds), 0.0)))
                cooldown_ok = allow_recovery and elapsed >= required_stay
                if window_locked and not hard_fail:
                    cooldown_ok = False
                if not allow_recovery:
                    stay_reason = "insufficient_health_passes"
                elif not cooldown_ok:
                    stay_reason = "cooldown_active"
                else:
                    switch_reason = reason or "recovered"
        else:
            state["consecutive_passes"] = 0
            cooldown_ok = not using_backup and (not window_locked or hard_fail)
            stay_reason = reason or "unhealthy"
            switch_reason = reason or "unhealthy"

        health_state: Mapping[str, Any] = {
            "is_healthy": healthy,
            "using_backup": using_backup,
            "allow_recovery": allow_recovery,
        }
        policy: Mapping[str, Any] = {
            "prefer_primary": True,
            "allow_recovery": allow_recovery,
        }
        normalized_active = _normalize_provider(active)
        consecutive_switches = int(self.consecutive_switches_by_provider.get(normalized_active, 0))
        decision_context: dict[str, Any] = {}
        target_provider = primary if using_backup else backup
        action = decide_provider_action(
            health_state,
            cooldown_ok,
            consecutive_switches,
            policy,
            from_provider=active,
            to_provider=target_provider,
            cooldown=cooldown_default,
            context=decision_context,
        )
        stay_reason = decision_context.get("stay_reason", stay_reason)
        stay_logged = bool(decision_context.get("stay_logged"))

        if action is ProviderAction.SWITCH and not self._sip_switch_allowed(target_provider):
            action = ProviderAction.STAY
            stay_reason = "sip_disabled"
            stay_logged = False

        if action is ProviderAction.SWITCH:
            if healthy and using_backup:
                state["active"] = primary
                state["last_switch"] = now
                state["consecutive_passes"] = 0
                state["cooldown"] = cooldown_default
                if window_seconds > 0:
                    state["decision_until"] = now + timedelta(seconds=window_seconds)
                    state["decision_provider"] = primary
                    state["decision_severity"] = "good"
                logger.info(
                    "DATA_PROVIDER_SWITCHOVER | from=%s to=%s reason=%s cooldown=%ss",
                    _normalize_provider(backup),
                    _normalize_provider(primary),
                    switch_reason,
                    cooldown_seconds,
                )
                return primary
            if not healthy and not using_backup:
                state["active"] = backup
                state["last_switch"] = now
                state["consecutive_passes"] = 0
                state["cooldown"] = cooldown_default
                if window_seconds > 0:
                    state["decision_until"] = now + timedelta(seconds=window_seconds)
                    state["decision_provider"] = backup
                    state["decision_severity"] = normalized_severity
                logger.info(
                    "DATA_PROVIDER_SWITCHOVER | from=%s to=%s reason=%s cooldown=%ss",
                    _normalize_provider(active),
                    _normalize_provider(backup),
                    switch_reason,
                    cooldown_default,
                )
                return backup
            action = ProviderAction.STAY

        if action is ProviderAction.DISABLE:
            target = backup if using_backup else active
            try:
                self.disable(target)
            finally:
                return target

        state["cooldown"] = cooldown_default
        stay_provider = active if not using_backup else backup
        cooldown_for_log = (
            max(cooldown_seconds, int(max(float(self.min_recovery_seconds), 0.0)))
            if using_backup and healthy
            else cooldown_default
        )
        if not stay_logged:
            record_stay(
                provider=stay_provider,
                reason=stay_reason,
                cooldown=cooldown_for_log,
            )
        state["active"] = stay_provider
        if window_seconds > 0 and (hard_fail or not window_locked):
            state["decision_until"] = now + timedelta(seconds=window_seconds)
            state["decision_provider"] = stay_provider
            state["decision_severity"] = normalized_severity
        return stay_provider


provider_monitor = ProviderMonitor()

__all__ = [
    "provider_monitor",
    "ProviderMonitor",
    "ProviderAction",
    "decide_provider_action",
    "is_safe_mode_active",
    "safe_mode_reason",
    "record_unauthorized_sip_event",
    "record_minute_gap_event",
]
