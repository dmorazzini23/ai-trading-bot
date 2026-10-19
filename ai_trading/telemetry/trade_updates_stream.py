"""Shadow recording and cached state for the Alpaca trade-updates stream.

The stream runs in shadow mode: every broker-pushed trade update is appended
to ``runtime/trade_updates_stream.jsonl`` and nothing else.  This module never
writes the OMS event store or ``fill_events.jsonl`` and never feeds a trading
decision.  Health surfaces read :func:`observe_trade_updates_stream_state`,
which only copies in-memory state.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from ai_trading.config.management import get_env
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
from ai_trading.logging import get_logger
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

_log = get_logger(__name__)

STREAM_ARTIFACT_ENV = "AI_TRADING_TRADE_UPDATES_STREAM_PATH"
STREAM_ARTIFACT_DEFAULT = "runtime/trade_updates_stream.jsonl"
PARITY_LATEST_ENV = "AI_TRADING_TRADE_UPDATES_PARITY_LATEST_PATH"
PARITY_LATEST_DEFAULT = "runtime/trade_updates_parity_latest.json"
ENABLED_ENV = "AI_TRADING_TRADE_UPDATES_STREAM_ENABLED"
GRACE_ENV = "AI_TRADING_TRADE_UPDATES_STREAM_GRACE_SEC"

SESSION_TZ = ZoneInfo("America/New_York")
_DEDUPE_MAX_KEYS = 20_000
_WRITE_FAILURE_LOG_INTERVAL_SEC = 60.0

_ORDER_TIMESTAMP_FIELDS = (
    "created_at",
    "updated_at",
    "submitted_at",
    "filled_at",
    "canceled_at",
    "expired_at",
    "failed_at",
    "replaced_at",
)

__all__ = [
    "TradeUpdatesShadowRecorder",
    "mark_stream_connected",
    "mark_stream_disconnected",
    "mark_stream_reconnecting",
    "mark_stream_started",
    "mark_stream_stopped",
    "normalize_trade_update",
    "observe_trade_updates_stream_state",
    "record_parity_summary",
    "reset_trade_updates_stream_state",
    "resolve_parity_latest_path",
    "resolve_stream_artifact_path",
    "session_date_for",
    "trade_updates_stream_enabled",
    "trade_updates_stream_paper",
]


def trade_updates_stream_enabled() -> bool:
    """Return the kill switch ``AI_TRADING_TRADE_UPDATES_STREAM_ENABLED`` (default on)."""

    return bool(get_env(ENABLED_ENV, True, cast=bool))


def trade_updates_stream_paper() -> bool:
    """Derive paper vs live from process configuration.

    Mirrors the trading client: ``ALPACA_TRADING_BASE_URL`` containing
    ``paper`` means paper.  Without a base URL, ``EXECUTION_MODE=live`` means
    live and anything else defaults to paper.
    """

    base_url = str(get_env("ALPACA_TRADING_BASE_URL", "", cast=str) or "").strip().lower()
    if base_url:
        return "paper" in base_url
    execution_mode = str(get_env("EXECUTION_MODE", "paper", cast=str) or "").strip().lower()
    return execution_mode != "live"


def _grace_seconds() -> float:
    try:
        return max(0.0, float(get_env(GRACE_ENV, 120.0, cast=float)))
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return 120.0


def resolve_stream_artifact_path(*, for_write: bool = False) -> Path:
    raw = str(get_env(STREAM_ARTIFACT_ENV, "", cast=str) or "").strip()
    return resolve_runtime_artifact_path(
        raw or None, default_relative=STREAM_ARTIFACT_DEFAULT, for_write=for_write
    )


def resolve_parity_latest_path(*, for_write: bool = False) -> Path:
    raw = str(get_env(PARITY_LATEST_ENV, "", cast=str) or "").strip()
    return resolve_runtime_artifact_path(
        raw or None, default_relative=PARITY_LATEST_DEFAULT, for_write=for_write
    )


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_ts(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def session_date_for(value: Any) -> str | None:
    """Return the America/New_York calendar date (YYYY-MM-DD) for a timestamp."""

    parsed = _parse_ts(value)
    if parsed is None:
        return None
    return parsed.astimezone(SESSION_TZ).date().isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float)):
        return enum_value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_safe(model_dump())
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            return str(value)
    return str(value)


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    safe = _json_safe(value)
    return dict(safe) if isinstance(safe, Mapping) else {}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(getattr(value, "value", value)).strip()
    return text or None


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_trade_update(raw: Any, *, received_at: str | None = None) -> dict[str, Any]:
    """Flatten a trade update (raw websocket message, data dict or model) to a record."""

    message = _as_mapping(raw)
    data = message.get("data") if isinstance(message.get("data"), Mapping) else message
    data = _as_mapping(data)
    order = _as_mapping(data.get("order"))
    order_timestamps = {
        field: _text(order.get(field))
        for field in _ORDER_TIMESTAMP_FIELDS
        if order.get(field) not in (None, "")
    }
    return {
        "received_at": received_at or _utc_now_iso(),
        "event": (_text(data.get("event")) or "").lower() or None,
        "event_timestamp": _text(data.get("timestamp")),
        "order_timestamps": order_timestamps,
        "order_id": _text(order.get("id")),
        "client_order_id": _text(order.get("client_order_id")),
        "symbol": _text(order.get("symbol")),
        "side": _text(order.get("side")),
        "order_status": _text(order.get("status")),
        "order_qty": _float(order.get("qty")),
        "qty": _float(data.get("qty")),
        "filled_qty": _float(order.get("filled_qty")),
        "price": _float(data.get("price")),
        "filled_avg_price": _float(order.get("filled_avg_price")),
        "position_qty": _float(data.get("position_qty")),
        "execution_id": _text(data.get("execution_id")),
        "raw": _json_safe(message),
    }


def dedupe_key(record: Mapping[str, Any]) -> tuple[str, ...]:
    discriminator = record.get("execution_id")
    if not discriminator:
        discriminator = f"filled_qty={record.get('filled_qty')}"
    return (
        str(record.get("order_id") or ""),
        str(record.get("event") or ""),
        str(record.get("event_timestamp") or ""),
        str(discriminator),
    )


class _StreamState:
    def __init__(self) -> None:
        self.lock = RLock()
        self.reset()

    def reset(self) -> None:
        self.enabled = False
        self.started = False
        self.paper: bool | None = None
        self.connected = False
        self.started_monotonic: float | None = None
        self.disconnected_monotonic: float | None = None
        self.started_at: str | None = None
        self.last_connected_at: str | None = None
        self.last_disconnected_at: str | None = None
        self.last_disconnect_reason: str | None = None
        self.last_event_at: str | None = None
        self.last_event_type: str | None = None
        self.events_session: str | None = None
        self.events_today = 0
        self.duplicates_skipped = 0
        self.reconnects = 0
        self.write_failures = 0
        self.parity_summary: dict[str, Any] | None = None


_STATE = _StreamState()


def reset_trade_updates_stream_state() -> None:
    with _STATE.lock:
        _STATE.reset()


def mark_stream_started(*, enabled: bool, paper: bool | None) -> None:
    with _STATE.lock:
        _STATE.enabled = bool(enabled)
        _STATE.started = True
        _STATE.paper = paper
        _STATE.connected = False
        _STATE.started_monotonic = time.monotonic()
        _STATE.disconnected_monotonic = None
        _STATE.started_at = _utc_now_iso()


def mark_stream_connected() -> None:
    with _STATE.lock:
        _STATE.connected = True
        _STATE.last_connected_at = _utc_now_iso()


def mark_stream_disconnected(reason: str) -> None:
    with _STATE.lock:
        _STATE.connected = False
        _STATE.disconnected_monotonic = time.monotonic()
        _STATE.last_disconnected_at = _utc_now_iso()
        _STATE.last_disconnect_reason = str(reason)


def mark_stream_reconnecting() -> None:
    with _STATE.lock:
        _STATE.reconnects += 1


def mark_stream_stopped() -> None:
    with _STATE.lock:
        _STATE.connected = False
        _STATE.started = False


def record_parity_summary(summary: Mapping[str, Any]) -> None:
    with _STATE.lock:
        _STATE.parity_summary = dict(summary)


def _note_event(record: Mapping[str, Any]) -> None:
    session = session_date_for(record.get("received_at"))
    with _STATE.lock:
        if session != _STATE.events_session:
            _STATE.events_session = session
            _STATE.events_today = 0
        _STATE.events_today += 1
        _STATE.last_event_at = str(record.get("received_at") or "")
        _STATE.last_event_type = record.get("event")


def _note_duplicate() -> None:
    with _STATE.lock:
        _STATE.duplicates_skipped += 1


def _note_write_failure() -> None:
    with _STATE.lock:
        _STATE.write_failures += 1


_PARITY_HEALTH_FIELDS = (
    "session",
    "generated_at",
    "available",
    "stream_fill_count",
    "polled_fill_count",
    "matched",
    "stream_only",
    "polled_only",
    "qty_mismatch",
    "price_delta_bps_median",
    "price_delta_bps_p95",
    "latency_seconds_median",
)


def observe_trade_updates_stream_state() -> dict[str, Any]:
    """Return the cached health section; no I/O, never blocks on the stream."""

    with _STATE.lock:
        degraded = False
        if _STATE.enabled and _STATE.started and not _STATE.connected:
            # The grace period runs from the later of stream start and the last
            # disconnect, so a reconnect after startup gets the same allowance.
            marks = [m for m in (_STATE.started_monotonic, _STATE.disconnected_monotonic) if m is not None]
            if marks and time.monotonic() - max(marks) >= _grace_seconds():
                degraded = True
        today = datetime.now(UTC).astimezone(SESSION_TZ).date().isoformat()
        events_today = _STATE.events_today if _STATE.events_session == today else 0
        parity = deepcopy(_STATE.parity_summary) if _STATE.parity_summary else None
        payload: dict[str, Any] = {
            "enabled": bool(_STATE.enabled),
            "started": bool(_STATE.started),
            "connected": bool(_STATE.connected),
            "degraded": degraded,
            "shadow_only": True,
            "paper": _STATE.paper,
            "started_at": _STATE.started_at,
            "last_connected_at": _STATE.last_connected_at,
            "last_disconnected_at": _STATE.last_disconnected_at,
            "last_disconnect_reason": _STATE.last_disconnect_reason,
            "last_event_at": _STATE.last_event_at,
            "last_event_type": _STATE.last_event_type,
            "events_today": events_today,
            "duplicates_skipped": _STATE.duplicates_skipped,
            "reconnects": _STATE.reconnects,
            "write_failures": _STATE.write_failures,
        }
    payload["parity"] = (
        {field: parity.get(field) for field in _PARITY_HEALTH_FIELDS} if parity else None
    )
    return payload


class TradeUpdatesShadowRecorder:
    """Append trade updates to the shadow artifact with bounded de-duplication."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        max_keys: int = _DEDUPE_MAX_KEYS,
    ) -> None:
        self._path = path
        self._max_keys = max(1, int(max_keys))
        self._seen: OrderedDict[tuple[str, ...], None] = OrderedDict()
        self._lock = RLock()
        self._last_failure_log = float("-inf")

    @property
    def path(self) -> Path:
        if self._path is None:
            self._path = resolve_stream_artifact_path(for_write=True)
        return self._path

    def _remember(self, key: tuple[str, ...]) -> bool:
        with self._lock:
            if key in self._seen:
                self._seen.move_to_end(key)
                return False
            self._seen[key] = None
            while len(self._seen) > self._max_keys:
                self._seen.popitem(last=False)
            return True

    def _append(self, record: Mapping[str, Any]) -> None:
        line = (json.dumps(record, sort_keys=True, default=str) + "\n").encode("utf-8")
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        # One O_APPEND write per record keeps concurrent appends line-atomic.
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
        try:
            view = memoryview(line)
            while view:
                written = os.write(fd, view)
                view = view[written:]
        finally:
            os.close(fd)

    def record(self, raw: Any) -> dict[str, Any] | None:
        """Normalize and append ``raw``; return the record, or ``None`` if skipped."""

        record = normalize_trade_update(raw)
        key = dedupe_key(record)
        if not self._remember(key):
            _note_duplicate()
            return None
        try:
            self._append(record)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            # Forget the key so a broker replay of this update can still land.
            with self._lock:
                self._seen.pop(key, None)
            _note_write_failure()
            now = time.monotonic()
            if now - self._last_failure_log >= _WRITE_FAILURE_LOG_INTERVAL_SEC:
                self._last_failure_log = now
                _log.warning(
                    "TRADE_UPDATES_STREAM_WRITE_FAILED",
                    extra={"error_type": exc.__class__.__name__, "detail": str(exc)},
                )
        _note_event(record)
        return record
