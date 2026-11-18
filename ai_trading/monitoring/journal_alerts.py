"""Slack alerts derived from the ai-trading.service journal.

Three alert kinds, all read from ``journalctl`` (the aiuser account is in the
``systemd-journal`` group, so no sudo is needed):

* service lifecycle: unit start, failure, or restart;
* trading stopped / resumed: halt and kill-switch markers and the drawdown
  breaker markers ``DRAWDOWN_BREAKER_TRIPPED`` / ``DRAWDOWN_BREAKER_RESET``;
* new ERROR / CRITICAL log line types, de-duplicated by message type.

Detection and message building are pure functions over journal entries. The
sender and the state file live in ``tools.mcp_slack_alerts_server`` so the
alerts share the incident dispatcher's webhook, lock and state directory.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from ai_trading.monitoring import slack_format as fmt

SERVICE_UNIT = "ai-trading.service"
JOURNAL_TIMEOUT_S = 30.0
MAX_ENTRIES = 20000
ERROR_COOLDOWN = timedelta(hours=1)
TRADING_COOLDOWN = timedelta(minutes=15)
SERVICE_COOLDOWN = timedelta(minutes=10)
STATE_RETENTION = timedelta(days=7)

# Markers that mean new openings are blocked, and the markers that lift them.
TRADING_STOP_MARKERS = (
    "DRAWDOWN_BREAKER_TRIPPED",
    "DISASTER_HALT_TRIGGERED",
    "ALERT_HALT_TRADING",
    "HALT_FLAG_WRITTEN",
    "DATA_KILL_SWITCH_ACTIVATED",
)
TRADING_RESUME_MARKERS = (
    "DRAWDOWN_BREAKER_RESET",
    "HALT_FLAG_REMOVED",
    "TRANSIENT_HALT_CLEARED",
)
_TRADING_MARKERS = frozenset(TRADING_STOP_MARKERS) | frozenset(TRADING_RESUME_MARKERS)
_ERROR_LEVELS = frozenset({"ERROR", "CRITICAL"})

_EXIT_RE = re.compile(r"Main process exited, code=(\w+), status=(\S+)")
_FAILED_RE = re.compile(r"Failed with result '([^']+)'")
_RESTART_RE = re.compile(r"Scheduled restart job, restart counter is at (\d+)")
_NUMBER_RE = re.compile(r"\b0x[0-9a-fA-F]+\b|\b\d[\d.:,-]*\b")

JournalRunner = Callable[[Sequence[str]], str]


class JournalReadError(RuntimeError):
    """journalctl could not be read."""


def run_journalctl(argv: Sequence[str]) -> str:
    """Run journalctl read-only and return stdout."""

    try:
        proc = subprocess.run(
            ["journalctl", *argv],
            capture_output=True,
            text=True,
            timeout=JOURNAL_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise JournalReadError(f"journalctl failed: {type(exc).__name__}") from exc
    if proc.returncode != 0:
        raise JournalReadError(f"journalctl exit {proc.returncode}: {proc.stderr.strip()[:200]}")
    return proc.stdout


def _parse_entries(raw: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            entries.append(row)
    return entries[-MAX_ENTRIES:]


def read_entries(
    *,
    cursor: str | None,
    since_minutes: int,
    unit: str = SERVICE_UNIT,
    runner: JournalRunner = run_journalctl,
) -> list[dict[str, Any]]:
    """Journal entries for the unit after ``cursor`` (or in the last ``since_minutes``)."""

    base = ["-u", unit, "-o", "json", "--no-pager"]
    if cursor:
        try:
            return _parse_entries(runner([*base, f"--after-cursor={cursor}"]))
        except JournalReadError:
            pass  # stale cursor after rotation: fall back to the time window
    return _parse_entries(runner([*base, "--since", f"-{max(1, int(since_minutes))} minutes"]))


def tail_cursor(*, unit: str = SERVICE_UNIT, runner: JournalRunner = run_journalctl) -> str | None:
    """Cursor of the newest entry, used to start without replaying history."""

    rows = _parse_entries(runner(["-u", unit, "-o", "json", "--no-pager", "-n", "1"]))
    return str(rows[-1].get("__CURSOR") or "") or None if rows else None


def _entry_time(entry: Mapping[str, Any]) -> datetime:
    try:
        return datetime.fromtimestamp(int(entry["__REALTIME_TIMESTAMP"]) / 1_000_000, UTC)
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return datetime.now(UTC)


def _structured(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    message = entry.get("MESSAGE")
    if isinstance(message, str) and message.startswith("{"):
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _is_systemd_line(entry: Mapping[str, Any]) -> bool:
    return str(entry.get("_SYSTEMD_UNIT") or "") == "init.scope"


def normalize_error_type(logger_name: str, message: str) -> str:
    """Stable key for an error line: numbers and ids collapse, so repeats group."""

    text = _NUMBER_RE.sub("#", message.strip().splitlines()[0] if message.strip() else "")
    text = re.sub(r"\s+", " ", text)[:80]
    return f"{logger_name}:{text}"


def detect(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Classify entries into alert events, oldest first.

    Each event: ``alert`` (service|trading|error), ``kind``, ``key`` (dedupe key),
    ``at`` (datetime) and ``detail`` fields for the message.
    """

    events: list[dict[str, Any]] = []
    for entry in entries:
        at = _entry_time(entry)
        message = str(entry.get("MESSAGE") or "")
        if _is_systemd_line(entry):
            event = _service_event(message, at)
            if event:
                events.append(event)
            continue
        payload = _structured(entry)
        if payload is not None:
            marker = str(payload.get("msg") or "")
            if marker in _TRADING_MARKERS:
                events.append(_trading_event(marker, payload, at))
                continue
            level = str(payload.get("level") or "").upper()
            if level in _ERROR_LEVELS:
                logger_name = str(payload.get("name") or "unknown")
                events.append(_error_event(level, logger_name, marker, payload, at))
            continue
        try:
            priority = int(entry.get("PRIORITY", 6))
        except (TypeError, ValueError):
            priority = 6
        if priority <= 3 and message.strip():
            events.append(_error_event("ERROR", "stderr", message, {}, at))
    return events


def _service_event(message: str, at: datetime) -> dict[str, Any] | None:
    if message.startswith("Started "):
        return {"alert": "service", "kind": "start", "key": "service:start", "at": at, "detail": ""}
    restart = _RESTART_RE.search(message)
    if restart:
        detail = f"scheduled restart #{restart.group(1)}"
        return {"alert": "service", "kind": "restart", "key": "service:failed", "at": at, "detail": detail}
    failed = _FAILED_RE.search(message)
    if failed:
        detail = f"failed with result {failed.group(1)}"
        return {"alert": "service", "kind": "failed", "key": "service:failed", "at": at, "detail": detail}
    exited = _EXIT_RE.search(message)
    if exited and not (exited.group(1) == "exited" and exited.group(2).startswith(("0", "SUCCESS"))):
        detail = f"main process {exited.group(1)} status {exited.group(2)}"
        return {"alert": "service", "kind": "failed", "key": "service:failed", "at": at, "detail": detail}
    return None


_TRADING_DETAIL_FIELDS = ("reason", "drawdown", "peak", "equity", "path", "provider")


def _trading_event(marker: str, payload: Mapping[str, Any], at: datetime) -> dict[str, Any]:
    stopped = marker in TRADING_STOP_MARKERS
    detail = {k: payload[k] for k in _TRADING_DETAIL_FIELDS if payload.get(k) is not None}
    return {
        "alert": "trading",
        "kind": "stopped" if stopped else "resumed",
        "key": f"trading:{marker}",
        "marker": marker,
        "at": at,
        "detail": detail,
    }


def _error_event(
    level: str, logger_name: str, message: str, payload: Mapping[str, Any], at: datetime
) -> dict[str, Any]:
    extras = {
        k: v
        for k, v in payload.items()
        if k not in {"ts", "level", "name", "msg"} and isinstance(v, str | int | float | bool)
    }
    return {
        "alert": "error",
        "kind": level.lower(),
        "key": f"error:{normalize_error_type(logger_name, message)}",
        "logger": logger_name,
        "message": message,
        "at": at,
        "detail": extras,
    }


def _parse_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _cooldown_for(alert: str) -> timedelta:
    return {"error": ERROR_COOLDOWN, "trading": TRADING_COOLDOWN}.get(alert, SERVICE_COOLDOWN)


def apply_dedupe(
    events: Sequence[Mapping[str, Any]],
    alerts_state: Mapping[str, Any],
    *,
    now: datetime,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Group events by dedupe key and apply the per-key cool-down.

    Returns ``(to_send, new_alerts_state)``. A key sends when it has never been
    seen or its cool-down has elapsed; otherwise its occurrences accumulate in
    ``suppressed`` and are reported with the next send as a repeat count.
    Service start events are never suppressed (a manual deploy must show).
    """

    state: dict[str, Any] = {k: dict(v) for k, v in alerts_state.items() if isinstance(v, Mapping)}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        grouped.setdefault(str(event["key"]), []).append(event)
    to_send: list[dict[str, Any]] = []
    for key, group in grouped.items():
        first, last = group[0], group[-1]
        record = state.get(key, {})
        last_sent = _parse_time(record.get("last_sent"))
        suppressed = int(record.get("suppressed", 0) or 0)
        cooled = last_sent is None or now - last_sent >= _cooldown_for(str(first["alert"]))
        if cooled or key == "service:start":
            repeats = len(group) + suppressed
            details = list(dict.fromkeys(str(e["detail"]) for e in group if isinstance(e["detail"], str)))
            to_send.append(
                {
                    **last,
                    "count": repeats,
                    "first_seen": first["at"],
                    "new": last_sent is None,
                    "details": details,
                }
            )
            state[key] = {"last_sent": now.isoformat(), "suppressed": 0, "last_seen": now.isoformat()}
        else:
            state[key] = {
                "last_sent": record.get("last_sent"),
                "suppressed": suppressed + len(group),
                "last_seen": now.isoformat(),
            }
    cutoff = now - STATE_RETENTION
    state = {
        k: v
        for k, v in state.items()
        if (_parse_time(v.get("last_seen")) or now) >= cutoff
    }
    to_send.sort(key=lambda e: e["at"])
    return to_send, state


def _ts(value: datetime) -> str:
    return fmt.fmt_time(value)


def _service_message(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = [e for e in events if e["kind"] in {"failed", "restart"}]
    starts = [e for e in events if e["kind"] == "start"]
    if failures:
        severity = fmt.Severity.CRITICAL
        title = "ai-trading.service failed or restarted unexpectedly"
        lines = [
            f"{_ts(e['at'])}: {'; '.join(e['details'])}"
            + (f" (x{e['count']})" if e["count"] > 1 else "")
            for e in failures
        ]
        action = "Check `journalctl -u ai-trading.service` and `/healthz`."
    else:
        severity = fmt.Severity.INFO
        title = "ai-trading.service started"
        lines = [f"Started at {_ts(e['at'])}" for e in starts]
        action = "A start with no failure above is a normal or manual (deploy) start."
    blocks = [
        fmt.header_block(severity, title),
        fmt.section_block("*What happened*\n" + fmt.bullets(lines)),
    ]
    if failures and starts:
        blocks.append(fmt.section_block(f"Service started again at {_ts(starts[-1]['at'])}."))
    blocks.append(fmt.context_block(action, fmt.now_label()))
    return fmt.message(fmt.fallback_text(severity, title, *[d for e in failures for d in e["details"]]), blocks)


def _drawdown_text(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return fmt.fmt_pct(number) if abs(number) <= 1 else f"{number:.2f}%"


def _trading_message(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stopped = [e for e in events if e["kind"] == "stopped"]
    resumed = [e for e in events if e["kind"] == "resumed"]
    last = events[-1]
    if last["kind"] == "stopped":
        severity, title = fmt.Severity.CRITICAL, "Trading stopped: new openings are blocked"
    else:
        severity, title = fmt.Severity.OK, "Trading resumed"
    lines = []
    for event in [*stopped, *resumed]:
        detail = dict(event["detail"])
        parts = [f"{fmt.humanize(event['marker'])}"]
        if "reason" in detail:
            parts.append(f"reason {detail['reason']}")
        if "drawdown" in detail:
            parts.append(f"drawdown {_drawdown_text(detail['drawdown'])}")
        for name in ("peak", "equity"):
            if name in detail:
                parts.append(f"{name} {fmt.fmt_money(detail[name], signed=False)}")
        lines.append(f"{_ts(event['at'])}: " + ", ".join(parts))
    blocks = [
        fmt.header_block(severity, title),
        fmt.section_block("*Markers*\n" + fmt.bullets(lines)),
        fmt.context_block("Existing positions are not touched by this alert.", fmt.now_label()),
    ]
    return fmt.message(fmt.fallback_text(severity, title, lines[-1] if lines else ""), blocks)


def _error_message(event: Mapping[str, Any]) -> dict[str, Any]:
    severity = fmt.Severity.CRITICAL if event["kind"] == "critical" else fmt.Severity.WARNING
    new = bool(event.get("new"))
    title = f"{'New' if new else 'Repeated'} {event['kind'].upper()}: {fmt.truncate(event['message'], 90)}"
    count = int(event["count"])
    blocks: list[dict[str, Any]] = [
        fmt.header_block(severity, title),
        fmt.fields_block(
            [
                ("Logger", event["logger"]),
                ("Occurrences", f"{count} since {_ts(event['first_seen'])}" if count > 1 else "1"),
                ("Last seen", _ts(event["at"])),
            ]
        ),
    ]
    if event["detail"]:
        extra = ", ".join(f"{k}={fmt.truncate(v, 60)}" for k, v in list(event["detail"].items())[:6])
        blocks.append(fmt.section_block(f"*Details*\n{fmt.truncate(extra, 500)}"))
    blocks.append(
        fmt.context_block(
            "Same error type is held back for 1 hour; repeats report a count.",
            fmt.now_label(),
        )
    )
    return fmt.message(fmt.fallback_text(severity, title, f"x{count}"), blocks)


def build_messages(to_send: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One message for service events, one for trading markers, one per error type."""

    messages: list[dict[str, Any]] = []
    service = [e for e in to_send if e["alert"] == "service"]
    trading = [e for e in to_send if e["alert"] == "trading"]
    if service:
        messages.append(_service_message(service))
    if trading:
        messages.append(_trading_message(trading))
    messages.extend(_error_message(e) for e in to_send if e["alert"] == "error")
    return messages


def load_alert_state(state: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
    cursor = state.get("cursor")
    alerts = state.get("alerts")
    return (str(cursor) if cursor else None), (dict(alerts) if isinstance(alerts, Mapping) else {})
