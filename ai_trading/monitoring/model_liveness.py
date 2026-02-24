from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger

logger = get_logger(__name__)

_METRIC_ML_SIGNAL = "ml_signal"
_METRIC_RL_SIGNAL = "rl_signal"
_METRIC_AFTER_HOURS = "after_hours_training"
_CRITICAL_CANARY_METRICS = {_METRIC_ML_SIGNAL, _METRIC_RL_SIGNAL}


@dataclass(frozen=True, slots=True)
class _LivenessBreach:
    metric: str
    event: str
    age_seconds: float
    threshold_seconds: float
    severity: str
    reason: str

    def as_payload(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "event": self.event,
            "age_seconds": round(self.age_seconds, 3),
            "threshold_seconds": round(self.threshold_seconds, 3),
            "severity": self.severity,
            "reason": self.reason,
        }


def _env_bool(name: str, default: bool) -> bool:
    try:
        return bool(get_env(name, default, cast=bool))
    except Exception:
        raw = str(get_env(name, default) or "").strip().lower()
        return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(get_env(name, default, cast=float))
    except Exception:
        try:
            return float(get_env(name, default))
        except Exception:
            return float(default)


def _event_for_metric(metric: str) -> str:
    if metric == _METRIC_ML_SIGNAL:
        return "ML_SIGNAL"
    if metric == _METRIC_RL_SIGNAL:
        return "RL_SIGNALS_EMITTED"
    if metric == _METRIC_AFTER_HOURS:
        return "AFTER_HOURS_TRAINING_COMPLETE"
    return metric


def _severity_for_metric(metric: str) -> str:
    if metric == _METRIC_AFTER_HOURS:
        return "warning"
    return "critical"


class _ModelLivenessMonitor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = datetime.now(UTC)
        self._last_seen: dict[str, datetime] = {}
        self._last_alert_at: dict[str, datetime] = {}
        self._last_canary_rollback_at: datetime | None = None

    def record(self, metric: str, now: datetime | None = None) -> None:
        ts = now or datetime.now(UTC)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        with self._lock:
            self._last_seen[metric] = ts.astimezone(UTC)

    def evaluate(
        self,
        *,
        market_open: bool,
        now: datetime | None = None,
    ) -> list[_LivenessBreach]:
        if not _env_bool("AI_TRADING_MODEL_LIVENESS_ENABLED", True):
            return []
        now_utc = now or datetime.now(UTC)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=UTC)
        now_utc = now_utc.astimezone(UTC)
        enforce_only_when_market_open = _env_bool(
            "AI_TRADING_MODEL_LIVENESS_REQUIRE_MARKET_OPEN",
            True,
        )
        alert_cooldown = max(
            0.0,
            _env_float("AI_TRADING_MODEL_LIVENESS_ALERT_COOLDOWN_SECONDS", 300.0),
        )
        thresholds: dict[str, float] = {
            _METRIC_ML_SIGNAL: max(
                1.0,
                _env_float("AI_TRADING_ML_SIGNAL_MAX_AGE_SECONDS", 5400.0),
            ),
            _METRIC_RL_SIGNAL: max(
                1.0,
                _env_float("AI_TRADING_RL_SIGNAL_MAX_AGE_SECONDS", 5400.0),
            ),
        }
        if _env_bool("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", False):
            thresholds[_METRIC_AFTER_HOURS] = max(
                60.0,
                _env_float("AI_TRADING_AFTER_HOURS_TRAINING_MAX_AGE_SECONDS", 129600.0),
            )

        breaches: list[_LivenessBreach] = []
        with self._lock:
            for metric, threshold_seconds in thresholds.items():
                if (
                    enforce_only_when_market_open
                    and not market_open
                    and metric in {_METRIC_ML_SIGNAL, _METRIC_RL_SIGNAL}
                ):
                    continue
                last_seen = self._last_seen.get(metric)
                baseline = last_seen if last_seen is not None else self._started_at
                age_seconds = max(0.0, (now_utc - baseline).total_seconds())
                if age_seconds <= threshold_seconds:
                    continue
                last_alert_at = self._last_alert_at.get(metric)
                if last_alert_at is not None:
                    since_last_alert = max(0.0, (now_utc - last_alert_at).total_seconds())
                    if since_last_alert < alert_cooldown:
                        continue
                self._last_alert_at[metric] = now_utc
                breaches.append(
                    _LivenessBreach(
                        metric=metric,
                        event=_event_for_metric(metric),
                        age_seconds=age_seconds,
                        threshold_seconds=threshold_seconds,
                        severity=_severity_for_metric(metric),
                        reason="never_observed" if last_seen is None else "stale",
                    )
                )
        return breaches

    def maybe_trigger_canary_rollback(
        self,
        breaches: list[_LivenessBreach],
        *,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        if not breaches:
            return None
        if not _env_bool("AI_TRADING_CANARY_AUTO_ROLLBACK_ENABLED", True):
            return None
        if not _env_bool("AI_TRADING_CANARY_ROLLBACK_ON_MODEL_LIVENESS_BREACH", True):
            return None
        canary_symbols_raw = str(get_env("AI_TRADING_CANARY_SYMBOLS", "") or "").strip()
        if not canary_symbols_raw:
            return None
        critical_breaches = [b for b in breaches if b.metric in _CRITICAL_CANARY_METRICS]
        if not critical_breaches:
            return None

        now_utc = now or datetime.now(UTC)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=UTC)
        now_utc = now_utc.astimezone(UTC)
        cooldown_seconds = max(
            0.0,
            _env_float("AI_TRADING_CANARY_ROLLBACK_COOLDOWN_SECONDS", 1800.0),
        )
        with self._lock:
            if self._last_canary_rollback_at is not None:
                since_last = max(0.0, (now_utc - self._last_canary_rollback_at).total_seconds())
                if since_last < cooldown_seconds:
                    return {
                        "triggered": False,
                        "status": "cooldown",
                        "cooldown_seconds": cooldown_seconds,
                        "since_last_seconds": round(since_last, 3),
                    }
            self._last_canary_rollback_at = now_utc

        payload: dict[str, Any] = {
            "triggered": True,
            "status": "triggered",
            "reason": "model_liveness_breach",
            "at": now_utc.isoformat(),
            "metrics": [b.metric for b in critical_breaches],
            "events": [b.event for b in critical_breaches],
        }
        flag_path = Path(
            str(
                get_env(
                    "AI_TRADING_CANARY_ROLLBACK_FLAG_PATH",
                    "runtime/canary_rollback.flag",
                )
                or "runtime/canary_rollback.flag"
            )
        )
        try:
            flag_path.parent.mkdir(parents=True, exist_ok=True)
            flag_path.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            payload["flag_path"] = str(flag_path)
        except Exception as exc:
            logger.error(
                "CANARY_AUTO_ROLLBACK_FLAG_WRITE_FAILED",
                extra={"path": str(flag_path), "error": str(exc)},
            )

        if _env_bool("AI_TRADING_CANARY_ROLLBACK_SET_KILL_SWITCH", True):
            kill_switch_path = Path(
                str(get_env("AI_TRADING_KILL_SWITCH_PATH", "runtime/kill_switch") or "runtime/kill_switch")
            )
            try:
                kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
                kill_switch_path.write_text(
                    f"canary_auto_rollback {now_utc.isoformat()}\n",
                    encoding="utf-8",
                )
                payload["kill_switch_path"] = str(kill_switch_path)
            except Exception as exc:
                logger.error(
                    "CANARY_AUTO_ROLLBACK_KILL_SWITCH_WRITE_FAILED",
                    extra={"path": str(kill_switch_path), "error": str(exc)},
                )

        rollback_command = str(get_env("AI_TRADING_CANARY_ROLLBACK_COMMAND", "") or "").strip()
        if rollback_command:
            command_timeout = max(
                1.0,
                _env_float("AI_TRADING_CANARY_ROLLBACK_COMMAND_TIMEOUT_SECONDS", 30.0),
            )
            try:
                completed = subprocess.run(
                    rollback_command,
                    shell=True,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=command_timeout,
                )
                payload["command"] = rollback_command
                payload["command_exit_code"] = int(completed.returncode)
            except Exception as exc:
                payload["command"] = rollback_command
                payload["command_error"] = str(exc)
                logger.error(
                    "CANARY_AUTO_ROLLBACK_COMMAND_FAILED",
                    extra={"command": rollback_command, "error": str(exc)},
                )
        return payload

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return {metric: ts.isoformat() for metric, ts in self._last_seen.items()}


_MONITOR = _ModelLivenessMonitor()


def note_ml_signal(*, now: datetime | None = None) -> None:
    _MONITOR.record(_METRIC_ML_SIGNAL, now=now)


def note_rl_signals_emitted(*, now: datetime | None = None) -> None:
    _MONITOR.record(_METRIC_RL_SIGNAL, now=now)


def note_after_hours_training_complete(*, now: datetime | None = None) -> None:
    _MONITOR.record(_METRIC_AFTER_HOURS, now=now)


def check_model_liveness(
    *,
    market_open: bool,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    return [breach.as_payload() for breach in _MONITOR.evaluate(market_open=market_open, now=now)]


def maybe_trigger_canary_auto_rollback(
    breaches: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    normalized: list[_LivenessBreach] = []
    for entry in breaches:
        try:
            normalized.append(
                _LivenessBreach(
                    metric=str(entry.get("metric") or ""),
                    event=str(entry.get("event") or ""),
                    age_seconds=float(entry.get("age_seconds") or 0.0),
                    threshold_seconds=float(entry.get("threshold_seconds") or 0.0),
                    severity=str(entry.get("severity") or "critical"),
                    reason=str(entry.get("reason") or "stale"),
                )
            )
        except Exception:
            continue
    return _MONITOR.maybe_trigger_canary_rollback(normalized, now=now)


def get_model_liveness_snapshot() -> dict[str, str]:
    return _MONITOR.snapshot()


def _reset_model_liveness_state_for_tests() -> None:
    global _MONITOR
    _MONITOR = _ModelLivenessMonitor()


__all__ = [
    "check_model_liveness",
    "get_model_liveness_snapshot",
    "maybe_trigger_canary_auto_rollback",
    "note_after_hours_training_complete",
    "note_ml_signal",
    "note_rl_signals_emitted",
]
