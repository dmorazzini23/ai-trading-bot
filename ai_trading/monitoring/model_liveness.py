from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

logger = get_logger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]

_METRIC_ML_SIGNAL = "ml_signal"
_METRIC_RL_SIGNAL = "rl_signal"
_METRIC_AFTER_HOURS = "after_hours_training"
_CRITICAL_CANARY_METRICS = {_METRIC_ML_SIGNAL, _METRIC_RL_SIGNAL}
_CANARY_ROLLBACK_FLAG_DEFAULT = "runtime/canary_rollback.flag"
_KILL_SWITCH_DEFAULT = "runtime/kill_switch"


@dataclass(frozen=True)
class _LivenessBreach:
    metric: str
    event: str
    age_seconds: float
    threshold_seconds: float
    severity: str
    reason: str
    context: dict[str, Any]

    def as_payload(self) -> dict[str, Any]:
        payload = {
            "metric": self.metric,
            "event": self.event,
            "age_seconds": round(self.age_seconds, 3),
            "threshold_seconds": round(self.threshold_seconds, 3),
            "severity": self.severity,
            "reason": self.reason,
        }
        payload.update(self.context)
        return payload


def _env_bool(name: str, default: bool) -> bool:
    try:
        return bool(get_env(name, default, cast=bool))
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        raw = str(get_env(name, default) or "").strip().lower()
        return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(get_env(name, default, cast=float))
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        try:
            return float(get_env(name, default))
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            return float(default)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_fh:
        if fcntl is not None:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_fh:
                tmp_fh.write(content)
                tmp_fh.flush()
                os.fsync(tmp_fh.fileno())
            os.replace(tmp_name, path)
            dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    logger.debug("CANARY_AUTO_ROLLBACK_TMP_CLEANUP_FAILED", exc_info=True)
            if fcntl is not None:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _parse_rollback_command(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        command = [str(item).strip() for item in parsed if str(item).strip()]
        if command:
            return command
        raise ValueError("rollback command list is empty")
    command = [part.strip() for part in shlex.split(text) if part.strip()]
    if command:
        return command
    raise ValueError("rollback command is empty")


def _resolve_liveness_runtime_path(
    env_key: str,
    default_relative: str,
    *,
    for_write: bool,
) -> Path:
    configured = str(get_env(env_key, default_relative) or default_relative)
    return resolve_runtime_artifact_path(
        configured,
        default_relative=default_relative,
        for_write=for_write,
    )


def _event_for_metric(metric: str) -> str:
    if metric == _METRIC_ML_SIGNAL:
        return "ML_SIGNAL"
    if metric == _METRIC_RL_SIGNAL:
        return "RL_SIGNALS_EMITTED"
    if metric == _METRIC_AFTER_HOURS:
        return "AFTER_HOURS_TRAINING_COMPLETE"
    return metric


def _severity_for_metric(metric: str) -> str:
    _ = metric
    return "warning"


def _ml_liveness_expected_default() -> bool:
    """Return whether ML signal heartbeat should be enforced by default."""

    if not _env_bool("AI_TRADING_MODEL_LIVENESS_ENFORCE_ML", True):
        return False
    bot_engine_module = sys.modules.get("ai_trading.core.bot_engine")
    if bot_engine_module is None:
        return True
    use_ml = getattr(bot_engine_module, "USE_ML", None)
    if use_ml is None:
        return True
    return bool(use_ml)


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
        signals_expected_now: bool,
        phase: str | None = None,
        execution_gate_open: bool | None = None,
        warmup_complete: bool | None = None,
        ml_expected: bool | None = None,
        rl_expected: bool | None = None,
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
        enforce_signal_liveness = bool(signals_expected_now) and (
            bool(market_open) or not enforce_only_when_market_open
        )
        alert_cooldown = max(
            0.0,
            _env_float("AI_TRADING_MODEL_LIVENESS_ALERT_COOLDOWN_SECONDS", 300.0),
        )
        if ml_expected is None:
            enforce_ml = _ml_liveness_expected_default()
        else:
            enforce_ml = bool(ml_expected) and _env_bool(
                "AI_TRADING_MODEL_LIVENESS_ENFORCE_ML",
                True,
            )
        enforce_rl_cfg = _env_bool("USE_RL_AGENT", False)
        if rl_expected is None:
            enforce_rl = bool(enforce_rl_cfg)
        else:
            enforce_rl = bool(enforce_rl_cfg) and bool(rl_expected)
        thresholds: dict[str, float] = {}
        if enforce_signal_liveness and enforce_ml:
            thresholds[_METRIC_ML_SIGNAL] = max(
                1.0,
                _env_float("AI_TRADING_ML_SIGNAL_MAX_AGE_SECONDS", 5400.0),
            )
        if enforce_signal_liveness and enforce_rl:
            thresholds[_METRIC_RL_SIGNAL] = max(
                1.0,
                _env_float("AI_TRADING_RL_SIGNAL_MAX_AGE_SECONDS", 5400.0),
            )
        if _env_bool("AI_TRADING_AFTER_HOURS_TRAINING_ENABLED", False):
            thresholds[_METRIC_AFTER_HOURS] = max(
                60.0,
                _env_float("AI_TRADING_AFTER_HOURS_TRAINING_MAX_AGE_SECONDS", 129600.0),
            )

        breaches: list[_LivenessBreach] = []
        with self._lock:
            last_ml_signal_ts = self._last_seen.get(_METRIC_ML_SIGNAL)
            last_rl_signal_ts = self._last_seen.get(_METRIC_RL_SIGNAL)
            ml_age_s = (
                max(0.0, (now_utc - last_ml_signal_ts).total_seconds())
                if last_ml_signal_ts is not None
                else None
            )
            rl_age_s = (
                max(0.0, (now_utc - last_rl_signal_ts).total_seconds())
                if last_rl_signal_ts is not None
                else None
            )
            ml_since_start_s = (
                None
                if last_ml_signal_ts is not None
                else max(0.0, (now_utc - self._started_at).total_seconds())
            )
            rl_since_start_s = (
                None
                if last_rl_signal_ts is not None
                else max(0.0, (now_utc - self._started_at).total_seconds())
            )
            base_context = {
                "last_ml_signal_ts": (
                    last_ml_signal_ts.isoformat()
                    if isinstance(last_ml_signal_ts, datetime)
                    else None
                ),
                "last_rl_signal_ts": (
                    last_rl_signal_ts.isoformat()
                    if isinstance(last_rl_signal_ts, datetime)
                    else None
                ),
                "ml_age_s": round(float(ml_age_s), 3) if ml_age_s is not None else None,
                "rl_age_s": round(float(rl_age_s), 3) if rl_age_s is not None else None,
                "ml_since_start_s": (
                    round(float(ml_since_start_s), 3)
                    if ml_since_start_s is not None
                    else None
                ),
                "rl_since_start_s": (
                    round(float(rl_since_start_s), 3)
                    if rl_since_start_s is not None
                    else None
                ),
                "ml_max_age_s": round(
                    float(thresholds.get(_METRIC_ML_SIGNAL, 0.0)),
                    3,
                ),
                "rl_max_age_s": round(
                    float(thresholds.get(_METRIC_RL_SIGNAL, 0.0)),
                    3,
                ),
                "market_open": bool(market_open),
                "signals_expected_now": bool(signals_expected_now),
                "phase": str(phase or "unknown"),
                "execution_gate_open": bool(execution_gate_open),
                "warmup_complete": bool(warmup_complete),
                "ml_enforced": bool(enforce_ml),
                "rl_enforced": bool(enforce_rl),
            }
            for metric, threshold_seconds in thresholds.items():
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
                        context=dict(base_context),
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
        flag_path: Path | str = _CANARY_ROLLBACK_FLAG_DEFAULT
        try:
            flag_path = _resolve_liveness_runtime_path(
                "AI_TRADING_CANARY_ROLLBACK_FLAG_PATH",
                _CANARY_ROLLBACK_FLAG_DEFAULT,
                for_write=True,
            )
            _atomic_write_text(
                flag_path,
                json.dumps(payload, sort_keys=True) + "\n",
            )
            payload["flag_path"] = str(flag_path)
        except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
            logger.error(
                "CANARY_AUTO_ROLLBACK_FLAG_WRITE_FAILED",
                extra={"path": str(flag_path), "error": str(exc)},
            )

        if _env_bool("AI_TRADING_CANARY_ROLLBACK_SET_KILL_SWITCH", True):
            kill_switch_path: Path | str = _KILL_SWITCH_DEFAULT
            try:
                kill_switch_path = _resolve_liveness_runtime_path(
                    "AI_TRADING_KILL_SWITCH_PATH",
                    _KILL_SWITCH_DEFAULT,
                    for_write=True,
                )
                _atomic_write_text(
                    kill_switch_path,
                    f"canary_auto_rollback {now_utc.isoformat()}\n",
                )
                payload["kill_switch_path"] = str(kill_switch_path)
            except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
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
                command_args = _parse_rollback_command(rollback_command)
                completed = subprocess.run(
                    command_args,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=command_timeout,
                )
                payload["command"] = rollback_command
                payload["command_args"] = list(command_args)
                payload["command_exit_code"] = int(completed.returncode)
            except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
                payload["command"] = rollback_command
                payload["command_error"] = str(exc)
                logger.error(
                    "CANARY_AUTO_ROLLBACK_COMMAND_FAILED",
                    extra={"command": rollback_command, "error": str(exc)},
                )
        return payload

    def snapshot(self, *, now: datetime | None = None) -> dict[str, Any]:
        now_utc = now or datetime.now(UTC)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=UTC)
        now_utc = now_utc.astimezone(UTC)
        with self._lock:
            ml_ts = self._last_seen.get(_METRIC_ML_SIGNAL)
            rl_ts = self._last_seen.get(_METRIC_RL_SIGNAL)
            ml_age_s = (
                max(0.0, (now_utc - ml_ts).total_seconds())
                if ml_ts is not None
                else None
            )
            rl_age_s = (
                max(0.0, (now_utc - rl_ts).total_seconds())
                if rl_ts is not None
                else None
            )
            ml_since_start_s = (
                None
                if ml_ts is not None
                else max(0.0, (now_utc - self._started_at).total_seconds())
            )
            rl_since_start_s = (
                None
                if rl_ts is not None
                else max(0.0, (now_utc - self._started_at).total_seconds())
            )
            return {
                "last_ml_signal_ts": ml_ts.isoformat() if ml_ts is not None else None,
                "last_rl_signal_ts": rl_ts.isoformat() if rl_ts is not None else None,
                "ml_age_s": round(float(ml_age_s), 3) if ml_age_s is not None else None,
                "rl_age_s": round(float(rl_age_s), 3) if rl_age_s is not None else None,
                "ml_since_start_s": (
                    round(float(ml_since_start_s), 3)
                    if ml_since_start_s is not None
                    else None
                ),
                "rl_since_start_s": (
                    round(float(rl_since_start_s), 3)
                    if rl_since_start_s is not None
                    else None
                ),
                "ml_max_age_s": round(
                    max(1.0, _env_float("AI_TRADING_ML_SIGNAL_MAX_AGE_SECONDS", 5400.0)),
                    3,
                ),
                "rl_max_age_s": round(
                    max(1.0, _env_float("AI_TRADING_RL_SIGNAL_MAX_AGE_SECONDS", 5400.0)),
                    3,
                ),
            }


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
    signals_expected_now: bool = True,
    phase: str | None = None,
    execution_gate_open: bool | None = None,
    warmup_complete: bool | None = None,
    ml_expected: bool | None = None,
    rl_expected: bool | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    return [
        breach.as_payload()
        for breach in _MONITOR.evaluate(
            market_open=market_open,
            signals_expected_now=signals_expected_now,
            phase=phase,
            execution_gate_open=execution_gate_open,
            warmup_complete=warmup_complete,
            ml_expected=ml_expected,
            rl_expected=rl_expected,
            now=now,
        )
    ]


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
                    severity=str(entry.get("severity") or "warning"),
                    reason=str(entry.get("reason") or "stale"),
                    context={},
                )
            )
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            continue
    return _MONITOR.maybe_trigger_canary_rollback(normalized, now=now)


def get_model_liveness_snapshot() -> dict[str, Any]:
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
