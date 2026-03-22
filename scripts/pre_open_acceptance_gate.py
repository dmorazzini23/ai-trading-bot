#!/usr/bin/env python3
"""Pre-open acceptance gate for the next trading session.

Workflow:
1. Sync ``.env`` -> ``.env.runtime``.
2. Refresh runtime reports.
3. Enforce runtime go/no-go (including reconciliation consistency checks).
4. Verify health endpoint is reachable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DEPRECATED_ENV_KEYS = {
    "AI_TRADING_EXEC_ALLOW_FALLBACK_WITHOUT_NBBO",
}


@dataclass
class Step:
    name: str
    status: str  # pass | warn | fail
    summary: str
    details: dict[str, Any]


def _run_command(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        text=True,
        capture_output=True,
    )


def _canonical_env_lines(path: Path) -> list[str]:
    lines: list[str] = []
    if not path.exists():
        return lines
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if not key or key in DEPRECATED_ENV_KEYS:
            continue
        lines.append(line)
    return lines


def _check_env_sync(repo_dir: Path, *, sync: bool) -> Step:
    sync_script = repo_dir / "scripts" / "sync_env_runtime.sh"
    if sync:
        if not sync_script.exists():
            return Step(
                name="env_sync",
                status="fail",
                summary="sync_env_runtime.sh is missing",
                details={"path": str(sync_script)},
            )
        proc = _run_command(["bash", str(sync_script)], cwd=repo_dir)
        if proc.returncode != 0:
            return Step(
                name="env_sync",
                status="fail",
                summary="failed to sync .env.runtime",
                details={
                    "command": f"bash {sync_script}",
                    "returncode": proc.returncode,
                    "stderr_tail": proc.stderr[-500:],
                    "stdout_tail": proc.stdout[-500:],
                },
            )
    env_lines = _canonical_env_lines(repo_dir / ".env")
    runtime_lines = _canonical_env_lines(repo_dir / ".env.runtime")
    missing = sorted(set(env_lines) - set(runtime_lines))
    extra = sorted(set(runtime_lines) - set(env_lines))
    if missing or extra:
        return Step(
            name="env_sync",
            status="fail",
            summary=".env.runtime does not match .env",
            details={
                "missing_from_runtime": missing[:50],
                "extra_in_runtime": extra[:50],
                "env_count": len(env_lines),
                "runtime_count": len(runtime_lines),
            },
        )
    return Step(
        name="env_sync",
        status="pass",
        summary=".env.runtime matches .env",
        details={"line_count": len(env_lines)},
    )


def _refresh_runtime_reports(repo_dir: Path, *, refresh: bool) -> Step:
    if not refresh:
        return Step(
            name="refresh_runtime_reports",
            status="warn",
            summary="refresh skipped by flag",
            details={},
        )
    script = repo_dir / "scripts" / "refresh_runtime_reports.sh"
    proc = _run_command(["bash", str(script)], cwd=repo_dir)
    if proc.returncode != 0:
        return Step(
            name="refresh_runtime_reports",
            status="fail",
            summary="failed to refresh runtime reports",
            details={
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-500:],
                "stdout_tail": proc.stdout[-500:],
            },
        )
    return Step(
        name="refresh_runtime_reports",
        status="pass",
        summary="runtime reports refreshed",
        details={},
    )


def _evaluate_runtime_gonogo() -> Step:
    from ai_trading.tools import runtime_performance_report as runtime_perf_report

    paths = runtime_perf_report.resolve_runtime_report_paths()
    trade_history_path = paths.get("trade_history")
    gate_summary_path = paths.get("gate_summary")
    gate_log_path = paths.get("gate_log")
    tca_path = paths.get("tca")
    if not isinstance(trade_history_path, Path) or not isinstance(gate_summary_path, Path):
        return Step(
            name="runtime_gonogo",
            status="fail",
            summary="unable to resolve runtime report paths",
            details={"paths": {key: str(value) for key, value in paths.items()}},
        )
    report = runtime_perf_report.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
        gate_log_path=gate_log_path if isinstance(gate_log_path, Path) else None,
        tca_path=tca_path if isinstance(tca_path, Path) else None,
    )
    thresholds = runtime_perf_report.resolve_runtime_gonogo_thresholds()
    decision = runtime_perf_report.evaluate_go_no_go(report, thresholds=thresholds)
    checks_raw = decision.get("checks")
    checks = dict(checks_raw) if isinstance(checks_raw, dict) else {}
    failed = [str(item) for item in decision.get("failed_checks", [])]
    gate_passed = bool(decision.get("gate_passed"))
    recon_available = bool(checks.get("open_position_reconciliation_available", False))
    recon_consistent = bool(checks.get("open_position_reconciliation_consistent", False))
    if not gate_passed:
        return Step(
            name="runtime_gonogo",
            status="fail",
            summary="runtime go/no-go failed",
            details={
                "failed_checks": failed,
                "checks": checks,
                "observed": decision.get("observed", {}),
                "thresholds": decision.get("thresholds", {}),
            },
        )
    if not recon_available or not recon_consistent:
        return Step(
            name="runtime_gonogo",
            status="fail",
            summary="reconciliation consistency gate failed",
            details={
                "failed_checks": failed,
                "checks": checks,
                "observed": decision.get("observed", {}),
            },
        )
    return Step(
        name="runtime_gonogo",
        status="pass",
        summary="runtime go/no-go and reconciliation checks passed",
        details={
            "failed_checks": failed,
            "checks": checks,
            "observed": decision.get("observed", {}),
        },
    )


def _health_port_from_env(repo_dir: Path) -> int:
    raw = os.environ.get("HEALTHCHECK_PORT")
    if raw and raw.strip():
        try:
            return int(raw.strip())
        except ValueError:
            pass
    runtime_env_path = repo_dir / ".env.runtime"
    for line in _canonical_env_lines(runtime_env_path):
        key, value = line.split("=", 1)
        if key.strip() == "HEALTHCHECK_PORT":
            try:
                return int(value.strip())
            except ValueError:
                break
    return 8081


def _check_health(repo_dir: Path, *, port: int | None, timeout_seconds: int, skip: bool) -> Step:
    if skip:
        return Step(
            name="healthz",
            status="warn",
            summary="health check skipped by flag",
            details={},
        )
    resolved_port = int(port or _health_port_from_env(repo_dir))
    url = f"http://127.0.0.1:{resolved_port}/healthz"
    deadline = time.monotonic() + max(1, timeout_seconds)
    last_error = ""
    payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2.0) as response:  # noqa: S310 - localhost health probe
                body = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                payload = parsed
                break
            last_error = "healthz payload is not a JSON object"
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            last_error = str(exc)
        time.sleep(1.0)
    if payload is None:
        return Step(
            name="healthz",
            status="fail",
            summary="health endpoint unreachable",
            details={"url": url, "error": last_error},
        )
    status_text = str(payload.get("status", "") or "").strip().lower()
    if status_text == "error":
        return Step(
            name="healthz",
            status="fail",
            summary="health endpoint reports error status",
            details={"url": url, "payload": payload},
        )
    if not bool(payload.get("ok", False)):
        return Step(
            name="healthz",
            status="warn",
            summary="health endpoint reachable but not yet healthy",
            details={"url": url, "payload": payload},
        )
    return Step(
        name="healthz",
        status="pass",
        summary="health endpoint is healthy",
        details={"url": url, "payload": payload},
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", default=".", help="Repository root (default: current directory).")
    parser.add_argument(
        "--health-port",
        type=int,
        default=None,
        help="Override HEALTHCHECK_PORT for /healthz probe.",
    )
    parser.add_argument(
        "--health-timeout-sec",
        type=int,
        default=60,
        help="Health probe timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip /healthz probe.",
    )
    parser.add_argument(
        "--no-sync-env",
        action="store_true",
        help="Skip .env -> .env.runtime sync step.",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip scripts/refresh_runtime_reports.sh.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    steps = [
        _check_env_sync(repo_dir, sync=not bool(args.no_sync_env)),
        _refresh_runtime_reports(repo_dir, refresh=not bool(args.no_refresh)),
        _evaluate_runtime_gonogo(),
        _check_health(
            repo_dir,
            port=args.health_port,
            timeout_seconds=int(args.health_timeout_sec),
            skip=bool(args.skip_health),
        ),
    ]
    fail_count = sum(1 for step in steps if step.status == "fail")
    warn_count = sum(1 for step in steps if step.status == "warn")
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_dir": str(repo_dir),
        "summary": {
            "pass": sum(1 for step in steps if step.status == "pass"),
            "warn": warn_count,
            "fail": fail_count,
        },
        "steps": [asdict(step) for step in steps],
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for step in steps:
            print(f"{step.status.upper():>4} {step.name}: {step.summary}")
        print(
            "SUMMARY pass={pass_count} warn={warn_count} fail={fail_count}".format(
                pass_count=report["summary"]["pass"],
                warn_count=warn_count,
                fail_count=fail_count,
            )
        )
    if fail_count > 0:
        return 1
    if warn_count > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
