#!/usr/bin/env python3
"""Generate a robust weekly runtime/repo drift audit for OpenClaw cron.

The audit is read-only and best-effort. It intentionally converts unavailable
commands or missing OpenClaw surfaces into findings so the cron can still
deliver a useful weekly summary.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


ROOT_DIR = Path(os.environ.get("AI_TRADING_AUDIT_ROOT", "/home/aiuser/ai-trading-bot"))
OPENCLAW_DIR = Path(os.environ.get("AI_TRADING_AUDIT_OPENCLAW_DIR", "/home/aiuser/.openclaw"))
HEALTH_URL = os.environ.get("AI_TRADING_AUDIT_HEALTH_URL", "http://127.0.0.1:9001/healthz")


def _run_command(args: list[str], *, cwd: Path | None = None, timeout: float = 12.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return int(proc.returncode), (proc.stdout or proc.stderr or "").strip()


def _read_health() -> tuple[str, dict[str, Any] | None]:
    if os.environ.get("AI_TRADING_AUDIT_SKIP_SYSTEM") == "1":
        return "health skipped in test/dry mode", None
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
        f"health ok={payload.get('ok')} status={payload.get('status')} reason={payload.get('reason')} "
        f"broker_connected={broker.get('connected')} open_orders={broker.get('open_orders_count')} "
        f"positions={broker.get('positions_count')} provider={provider.get('active')} "
        f"provider_status={provider.get('status')}",
        payload,
    )


def _service_summary() -> str:
    if os.environ.get("AI_TRADING_AUDIT_SKIP_SYSTEM") == "1":
        return "service skipped in test/dry mode"
    code, output = _run_command(["systemctl", "is-active", "ai-trading.service"], timeout=5)
    if code == 0:
        return "ai-trading.service active"
    return f"ai-trading.service status check failed exit={code}: {output or 'no output'}"


def _git_summary() -> tuple[str, int]:
    code, output = _run_command(["git", "status", "--short"], cwd=ROOT_DIR, timeout=8)
    if code != 0:
        return f"git status unavailable exit={code}: {output[:200] or 'no output'}", 0
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return "repo clean", 0
    preview = "; ".join(lines[:8])
    suffix = "" if len(lines) <= 8 else f"; +{len(lines) - 8} more"
    return f"repo dirty: {len(lines)} changed/untracked entries ({preview}{suffix})", len(lines)


def _cron_summary() -> tuple[str, int]:
    if os.environ.get("AI_TRADING_AUDIT_SKIP_OPENCLAW") == "1":
        return "OpenClaw cron skipped in test/dry mode", 0
    code, output = _run_command(["openclaw", "cron", "list"], cwd=ROOT_DIR, timeout=20)
    if code != 0:
        return f"OpenClaw cron list failed exit={code}: {output[:240] or 'no output'}", 1
    job_lines = [line for line in output.splitlines() if line.strip().startswith(tuple("0123456789abcdef"))]
    error_lines = [line for line in job_lines if " error " in f" {line} "]
    ok_lines = [line for line in job_lines if " ok " in f" {line} "]
    warnings = output.count("plugin not installed")
    return (
        f"OpenClaw cron jobs={len(job_lines)} ok={len(ok_lines)} errors={len(error_lines)} "
        f"config_plugin_warnings={warnings}",
        len(error_lines),
    )


def _openclaw_workspace_summary(now: datetime) -> str:
    memory_file = OPENCLAW_DIR / "workspace" / "MEMORY.md"
    if not memory_file.exists():
        memory = "workspace memory missing"
    else:
        mtime = datetime.fromtimestamp(memory_file.stat().st_mtime, tz=UTC)
        age_days = (now - mtime).total_seconds() / 86400.0
        memory = f"workspace memory age={age_days:.1f}d path={memory_file}"
    hook_dir = OPENCLAW_DIR / "hooks"
    hook_count = sum(1 for path in hook_dir.rglob("*") if path.is_file()) if hook_dir.exists() else 0
    return f"{memory}; hook files={hook_count}"


def _artifact_summary(now: datetime) -> str:
    candidates = [
        ROOT_DIR / "scripts" / "openclaw_market_close_recap.py",
        Path("/var/lib/ai-trading-bot/runtime/reports/trading_day_latest.json"),
        Path("/var/lib/ai-trading-bot/runtime/research_reports/latest/daily_research_automation_latest.json"),
    ]
    parts: list[str] = []
    for path in candidates:
        if not path.exists():
            parts.append(f"{path.name}=missing")
            continue
        age_hours = (now - datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)).total_seconds() / 3600.0
        parts.append(f"{path.name}_age={age_hours:.1f}h")
    return ", ".join(parts)


def build_audit() -> str:
    now = datetime.now(UTC)
    service = _service_summary()
    health_text, health = _read_health()
    git_text, dirty_count = _git_summary()
    cron_text, cron_errors = _cron_summary()
    workspace_text = _openclaw_workspace_summary(now)
    artifact_text = _artifact_summary(now)

    runtime_ok = isinstance(health, dict) and health.get("ok") is True
    if runtime_ok and cron_errors == 0:
        verdict = "Runtime is healthy and OpenClaw cron is green."
    elif runtime_ok:
        verdict = "Runtime is healthy, but OpenClaw automation still has drift."
    else:
        verdict = "Runtime or health reporting needs review."

    drift: list[str] = []
    if dirty_count:
        drift.append(git_text)
    drift.append(cron_text)
    drift.append(workspace_text)
    drift.append(artifact_text)
    if not drift:
        drift.append("No high-signal drift found.")

    if cron_errors:
        step = "Keep OpenClaw recurring jobs command-based and smoke-run any cron after changing it."
    elif dirty_count:
        step = "Review and commit or intentionally leave the current repo patchset before the next large change."
    elif not runtime_ok:
        step = "Inspect /healthz and ai-trading.service before the next market session."
    else:
        step = "Keep the current automation shape; next hardening focus is reducing stale/noisy OpenClaw plugin warnings."

    return "\n".join(
        [
            f"Weekly audit complete for `{now.strftime('%Y-%m-%d %H:%M UTC')}`.",
            "",
            f"Runtime: {service}; {health_text}.",
            "",
            "Highest-signal drift:",
            *[f"- {item}" for item in drift],
            "",
            f"Verdict: {verdict}",
            "",
            f"Single best hardening step for next week: {step}",
        ]
    )


def main() -> int:
    try:
        sys.stdout.write(build_audit() + "\n")
    except Exception as exc:  # pragma: no cover - final cron safety net
        sys.stdout.write(
            "Weekly audit degraded.\n\n"
            f"Unexpected audit exception: {type(exc).__name__}: {exc}\n\n"
            "Single best hardening step for next week: inspect ai-trading.service, /healthz, and OpenClaw cron manually.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
