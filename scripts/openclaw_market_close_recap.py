#!/usr/bin/env python3
"""Generate a robust market-close recap for OpenClaw cron delivery.

The script is intentionally read-only and best-effort: missing logs, stale
artifacts, or unavailable health endpoints are reported in the recap instead of
failing the cron job.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


RUNTIME_DIR = Path(os.environ.get("AI_TRADING_RECAP_RUNTIME_DIR", "/var/lib/ai-trading-bot/runtime"))
HEALTH_URL = os.environ.get("AI_TRADING_RECAP_HEALTH_URL", "http://127.0.0.1:9001/healthz")
JOURNAL_PATTERN = re.compile(
    r"ORDER_SUBMITTED|ORDER_FILLED|fill_recorded|ERROR|Exception|Traceback|BUDGET_OVER|CRITICAL",
    re.IGNORECASE,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _utc_today() -> str:
    return os.environ.get("AI_TRADING_RECAP_DATE") or datetime.now(UTC).date().isoformat()


def _parse_ts(raw: Any) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _is_fresh_date(raw: Any, day: str) -> bool:
    ts = _parse_ts(raw)
    return bool(ts and ts.date().isoformat() == day)


def _run_command(args: list[str], *, timeout: float = 12.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    output = (proc.stdout or proc.stderr or "").strip()
    return int(proc.returncode), output


def _service_summary() -> str:
    if os.environ.get("AI_TRADING_RECAP_SKIP_SYSTEM") == "1":
        return "service status not checked in test/dry mode"
    code, output = _run_command(["systemctl", "is-active", "ai-trading.service"], timeout=5)
    if code == 0:
        return "ai-trading.service active"
    return f"ai-trading.service status check returned {code}: {output or 'no output'}"


def _health_summary() -> tuple[str, dict[str, Any] | None]:
    if os.environ.get("AI_TRADING_RECAP_SKIP_SYSTEM") == "1":
        return "health not checked in test/dry mode", None
    try:
        with urlopen(HEALTH_URL, timeout=8) as resp:  # nosec B310 - local operator endpoint
            body = resp.read().decode("utf-8", errors="replace")
    except (OSError, URLError, TimeoutError) as exc:
        return f"health unavailable: {exc}", None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return "health returned non-json payload", None
    if not isinstance(payload, dict):
        return "health returned non-object payload", None
    broker = payload.get("broker") if isinstance(payload.get("broker"), dict) else {}
    provider = payload.get("provider_state") if isinstance(payload.get("provider_state"), dict) else {}
    return (
        "health "
        f"ok={payload.get('ok')} status={payload.get('status')} reason={payload.get('reason')} "
        f"broker_connected={broker.get('connected')} "
        f"open_orders={broker.get('open_orders_count')} positions={broker.get('positions_count')} "
        f"provider={provider.get('active')} provider_status={provider.get('status')}",
        payload,
    )


def _journal_summary(day: str) -> str:
    if os.environ.get("AI_TRADING_RECAP_SKIP_SYSTEM") == "1":
        return "journal not checked in test/dry mode"
    since = os.environ.get("AI_TRADING_RECAP_JOURNAL_SINCE")
    if not since:
        since_dt = datetime.fromisoformat(f"{day}T19:20:00+00:00") - timedelta(minutes=0)
        since = since_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    code, output = _run_command(
        ["journalctl", "-u", "ai-trading.service", "--since", since, "--no-pager", "-n", "400"],
        timeout=15,
    )
    if code != 0:
        return f"journal unavailable exit={code}: {output[:240] or 'no output'}"
    matches = [line for line in output.splitlines() if JOURNAL_PATTERN.search(line)]
    if not matches:
        return "no matching journal lines for order/fill/error patterns"
    tail = matches[-5:]
    return f"{len(matches)} matching journal lines; latest: " + " | ".join(line[-180:] for line in tail)


def _fill_summary(day: str) -> dict[str, Any]:
    path = RUNTIME_DIR / "fill_events.jsonl"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return {"available": False, "reason": "fill_events.jsonl missing"}
    last_ts = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        ts_text = str(row.get("ts") or row.get("entry_time") or "")
        if ts_text > str(last_ts or ""):
            last_ts = ts_text
        if ts_text.startswith(day) and str(row.get("event") or "").lower() == "fill_recorded":
            rows.append(row)
    by_symbol: dict[str, dict[str, float]] = defaultdict(lambda: {"fills": 0.0, "qty": 0.0, "edge_bps": 0.0})
    sides: Counter[str] = Counter()
    for row in rows:
        symbol = str(row.get("symbol") or "UNKNOWN")
        by_symbol[symbol]["fills"] += 1
        by_symbol[symbol]["qty"] += float(row.get("fill_qty") or row.get("qty") or 0.0)
        by_symbol[symbol]["edge_bps"] += float(row.get("realized_net_edge_bps") or 0.0)
        sides[str(row.get("side") or "unknown").lower()] += 1
    return {
        "available": True,
        "day": day,
        "fills": len(rows),
        "qty": sum(float(row.get("fill_qty") or row.get("qty") or 0.0) for row in rows),
        "edge_bps": sum(float(row.get("realized_net_edge_bps") or 0.0) for row in rows),
        "symbols": dict(sorted(by_symbol.items())),
        "sides": dict(sorted(sides.items())),
        "last_ts": last_ts,
    }


def _trading_day_summary(day: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    path = RUNTIME_DIR / "reports" / "trading_day_latest.json"
    report = _read_json(path)
    if report is None:
        return "trading-day report missing or unreadable", ["trading_day_latest.json missing/unreadable"]
    generated_at = report.get("generated_at")
    report_date = str(report.get("report_date") or "")
    if report_date != day:
        warnings.append(f"trading-day report stale: report_date={report_date or 'unknown'} generated_at={generated_at}")
    summary = report.get("openclaw_summary") if isinstance(report.get("openclaw_summary"), dict) else {}
    details = summary.get("details") if isinstance(summary.get("details"), dict) else {}
    text = str(summary.get("summary") or "no trading-day summary")
    if details:
        text += (
            f"; desired={details.get('desired')} submitted={details.get('submitted')} "
            f"fills={details.get('fills')} rejected={details.get('rejected')} "
            f"symbols={','.join(details.get('symbols_with_activity') or [])}"
        )
    return text, warnings


def _operator_issues() -> list[str]:
    issues: list[str] = []
    for rel, label in (
        (Path("operator_control_plane_latest.json"), "operator control plane"),
        (Path("openclaw_incident_state.json"), "OpenClaw incident state"),
        (Path("slack_incident_state.json"), "Slack incident state"),
    ):
        path = RUNTIME_DIR / rel
        payload = _read_json(path)
        if payload is None:
            continue
        status = payload.get("status") or payload.get("severity") or payload.get("last_status")
        generated = payload.get("generated_at") or payload.get("updated_at") or payload.get("ts")
        if status:
            issues.append(f"{label}: {status} at {generated or 'unknown time'}")
    return issues


def build_recap() -> str:
    day = _utc_today()
    service = _service_summary()
    health_text, health = _health_summary()
    fills = _fill_summary(day)
    trading_text, warnings = _trading_day_summary(day)
    journal = _journal_summary(day)
    operator_issues = _operator_issues()

    broker_flat = False
    if isinstance(health, dict):
        broker = health.get("broker") if isinstance(health.get("broker"), dict) else {}
        broker_flat = broker.get("open_orders_count") == 0 and broker.get("positions_count") == 0

    if isinstance(health, dict) and health.get("ok") is True and broker_flat:
        verdict = "Healthy close: service is up, broker is connected, and exposure is flat."
    elif isinstance(health, dict):
        verdict = f"Close needs review: health ok={health.get('ok')} status={health.get('status')}."
    else:
        verdict = f"Close needs review: {health_text}."

    fill_text = "fills unavailable"
    if fills.get("available"):
        fill_text = (
            f"{fills['fills']} fills, qty {fills['qty']:g}, realized edge sum "
            f"{fills['edge_bps']:.2f} bps, symbols "
            f"{', '.join(fills['symbols']) or 'none'}, last fill {fills.get('last_ts') or 'n/a'}"
        )

    issues = [journal, *warnings, *operator_issues]
    if not issues:
        issues = ["No operational issues found in available checks."]

    lines = [
        "**Close verdict**",
        verdict,
        "",
        "**Trading snapshot**",
        f"{fill_text}. {trading_text}.",
        "",
        "**Operational issues**",
        " ".join(issues),
        "",
        "**Next action**",
        "Leave the service running. Review any blocked research/model-readiness sections before the next open; no restart is indicated by this recap.",
    ]
    return "\n".join(lines)


def main() -> int:
    try:
        sys.stdout.write(build_recap() + "\n")
    except Exception as exc:  # pragma: no cover - final cron safety net
        sys.stdout.write(
            "**Close verdict**\n"
            "Close recap degraded: the recap script hit an unexpected error.\n\n"
            "**Trading snapshot**\n"
            "Unavailable from recap script.\n\n"
            "**Operational issues**\n"
            f"recap_exception={type(exc).__name__}: {exc}\n\n"
            "**Next action**\n"
            "Check ai-trading.service health and the daily research artifacts manually.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
