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
import re
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

PREOPEN_REQUIRE_FLAT_START_ENV = "AI_TRADING_EXECUTION_PREOPEN_REQUIRE_FLAT_START"
PREOPEN_EXPECTED_SWING_SYMBOLS_ENV = "AI_TRADING_EXECUTION_PREOPEN_EXPECTED_SWING_SYMBOLS"
HEALTH_REQUIRE_OMS_INVARIANTS_ENV = "AI_TRADING_HEALTH_REQUIRE_OMS_INVARIANTS"
HEALTH_REQUIRE_OMS_LIFECYCLE_PARITY_ENV = "AI_TRADING_HEALTH_REQUIRE_OMS_LIFECYCLE_PARITY"

_SECRET_KEY_HINT_RE = re.compile(r"(SECRET|TOKEN|PASSWORD|WEBHOOK_URL$|API_KEY$)")
_SECRETS_BACKEND_NONE = {"", "none", "off", "disabled"}


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


def _canonical_env_map(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key or key in DEPRECATED_ENV_KEYS:
            continue
        values[key] = value
    return values


def _parse_bool(value: str, *, default: bool = False) -> bool:
    text = value.strip().strip('"').strip("'").lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _env_value(repo_dir: Path, key: str) -> tuple[str | None, str | None]:
    raw = os.environ.get(key)
    if raw is not None:
        return raw, "process"
    runtime_map = _canonical_env_map(repo_dir / ".env.runtime")
    if key in runtime_map:
        return runtime_map[key], ".env.runtime"
    env_map = _canonical_env_map(repo_dir / ".env")
    if key in env_map:
        return env_map[key], ".env"
    return None, None


def _env_bool_detail(repo_dir: Path, key: str, *, default: bool = False) -> dict[str, Any]:
    raw, source = _env_value(repo_dir, key)
    enabled = _parse_bool(str(raw), default=default) if raw is not None else default
    return {
        "enabled": enabled,
        "raw": raw,
        "source": source,
        "default": default,
    }


def _parse_managed_secret_keys(raw: str) -> set[str]:
    keys: set[str] = set()
    for part in raw.split(","):
        key = part.strip().upper()
        if key:
            keys.add(key)
    return keys


def _infer_secret_key_names(keys: set[str]) -> set[str]:
    inferred: set[str] = set()
    for key in keys:
        if key == "ALPACA_API_KEY" or _SECRET_KEY_HINT_RE.search(key):
            inferred.add(key)
    return inferred


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
    env_map = _canonical_env_map(repo_dir / ".env")
    runtime_map = _canonical_env_map(repo_dir / ".env.runtime")
    backend = str(env_map.get("AI_TRADING_SECRETS_BACKEND", "none") or "none").strip().lower()
    if backend in _SECRETS_BACKEND_NONE:
        missing = sorted(set(env_map) - set(runtime_map))
        extra = sorted(set(runtime_map) - set(env_map))
        mismatch = sorted(
            key
            for key in set(env_map).intersection(runtime_map)
            if env_map.get(key) != runtime_map.get(key)
        )
        if missing or extra or mismatch:
            return Step(
                name="env_sync",
                status="fail",
                summary=".env.runtime does not match .env",
                details={
                    "missing_from_runtime": missing[:50],
                    "extra_in_runtime": extra[:50],
                    "value_mismatch_keys": mismatch[:50],
                    "env_count": len(env_map),
                    "runtime_count": len(runtime_map),
                },
            )
        return Step(
            name="env_sync",
            status="pass",
            summary=".env.runtime matches .env",
            details={"line_count": len(env_map)},
        )

    managed_keys = _parse_managed_secret_keys(env_map.get("AI_TRADING_MANAGED_SECRET_KEYS", ""))
    managed_keys.update(_infer_secret_key_names(set(env_map) | set(runtime_map)))
    compare_keys = sorted((set(env_map) | set(runtime_map)) - managed_keys)

    missing = sorted(key for key in compare_keys if key in env_map and key not in runtime_map)
    extra = sorted(key for key in compare_keys if key in runtime_map and key not in env_map)
    mismatch = sorted(
        key
        for key in compare_keys
        if key in env_map and key in runtime_map and env_map.get(key) != runtime_map.get(key)
    )

    require_managed = _parse_bool(
        str(env_map.get("AI_TRADING_REQUIRE_MANAGED_SECRETS", "0")),
        default=False,
    )
    missing_managed = sorted(
        key for key in managed_keys if require_managed and key in env_map and key not in runtime_map
    )

    if missing or extra or mismatch or missing_managed:
        return Step(
            name="env_sync",
            status="fail",
            summary=".env.runtime does not match .env (non-secret keys)",
            details={
                "secrets_backend": backend,
                "managed_secret_key_count": len(managed_keys),
                "missing_from_runtime": missing[:50],
                "extra_in_runtime": extra[:50],
                "value_mismatch_keys": mismatch[:50],
                "missing_required_managed_keys": missing_managed[:50],
                "env_count": len(env_map),
                "runtime_count": len(runtime_map),
            },
        )
    return Step(
        name="env_sync",
        status="pass",
        summary=".env.runtime matches .env for non-secret keys",
        details={
            "secrets_backend": backend,
            "managed_secret_key_count": len(managed_keys),
            "line_count": len(compare_keys),
        },
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
    from ai_trading.env import ensure_dotenv_loaded
    from ai_trading.tools import runtime_performance_report as runtime_perf_report

    ensure_dotenv_loaded()
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
    runtime_api_port: int | None = None
    for line in _canonical_env_lines(runtime_env_path):
        key, value = line.split("=", 1)
        if key.strip() == "HEALTHCHECK_PORT":
            try:
                return int(value.strip())
            except ValueError:
                break
        if key.strip() == "API_PORT":
            try:
                runtime_api_port = int(value.strip())
            except ValueError:
                runtime_api_port = None
    raw_api_port = os.environ.get("API_PORT")
    if raw_api_port and raw_api_port.strip():
        try:
            return int(raw_api_port.strip())
        except ValueError:
            pass
    if runtime_api_port is not None:
        return runtime_api_port
    return 9001


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


def _safe_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _health_payload_from_step(health_step: Step) -> dict[str, Any] | None:
    payload = health_step.details.get("payload")
    return dict(payload) if isinstance(payload, dict) else None


def _runtime_flat_start_context(payload: dict[str, Any]) -> dict[str, Any]:
    preopen_raw = payload.get("preopen_readiness")
    if not isinstance(preopen_raw, dict):
        service_state_raw = payload.get("service_state")
        service_state = dict(service_state_raw) if isinstance(service_state_raw, dict) else {}
        preopen_raw = service_state.get("preopen_readiness")
    preopen = dict(preopen_raw) if isinstance(preopen_raw, dict) else {}
    flat_start_raw = preopen.get("flat_start")
    if not isinstance(flat_start_raw, dict):
        flat_start_raw = payload.get("flat_start")
    return dict(flat_start_raw) if isinstance(flat_start_raw, dict) else {}


def _check_preopen_operator_drill(repo_dir: Path, *, health_step: Step) -> Step:
    """Summarize pre-open flat-start and OMS strictness signals for operators."""

    flat_start_required = _env_bool_detail(
        repo_dir,
        PREOPEN_REQUIRE_FLAT_START_ENV,
        default=False,
    )
    oms_invariants_required = _env_bool_detail(
        repo_dir,
        HEALTH_REQUIRE_OMS_INVARIANTS_ENV,
        default=False,
    )
    oms_lifecycle_required = _env_bool_detail(
        repo_dir,
        HEALTH_REQUIRE_OMS_LIFECYCLE_PARITY_ENV,
        default=False,
    )
    expected_swing_symbols, expected_swing_source = _env_value(
        repo_dir,
        PREOPEN_EXPECTED_SWING_SYMBOLS_ENV,
    )
    config = {
        PREOPEN_REQUIRE_FLAT_START_ENV: flat_start_required,
        HEALTH_REQUIRE_OMS_INVARIANTS_ENV: oms_invariants_required,
        HEALTH_REQUIRE_OMS_LIFECYCLE_PARITY_ENV: oms_lifecycle_required,
        PREOPEN_EXPECTED_SWING_SYMBOLS_ENV: {
            "raw": expected_swing_symbols,
            "source": expected_swing_source,
        },
    }

    payload = _health_payload_from_step(health_step)
    if payload is None:
        return Step(
            name="preopen_operator_drill",
            status="warn",
            summary="health payload unavailable; cannot verify flat-start or OMS drill state",
            details={
                "config": config,
                "health_step_status": health_step.status,
                "health_step_summary": health_step.summary,
            },
        )

    broker = payload.get("broker")
    broker_payload = dict(broker) if isinstance(broker, dict) else {}
    open_orders_count = _safe_nonnegative_int(broker_payload.get("open_orders_count"))
    positions_count = _safe_nonnegative_int(broker_payload.get("positions_count"))
    runtime_flat_start = _runtime_flat_start_context(payload)
    runtime_open_orders_count = _safe_nonnegative_int(
        runtime_flat_start.get("open_orders_count")
    )
    runtime_unexpected_positions_count = _safe_nonnegative_int(
        runtime_flat_start.get("unexpected_positions_count")
    )
    runtime_flat_start_has_counts = (
        runtime_open_orders_count is not None
        or runtime_unexpected_positions_count is not None
    )
    trust_runtime_flat_start = bool(
        runtime_flat_start
        and runtime_flat_start.get("enabled", True)
        and runtime_flat_start_has_counts
    )
    attention_flags_raw = payload.get("attention_flags")
    attention_flags = (
        [str(flag) for flag in attention_flags_raw]
        if isinstance(attention_flags_raw, list)
        else []
    )
    readiness_failures_raw = payload.get("readiness_failures")
    readiness_failures = (
        [str(failure) for failure in readiness_failures_raw]
        if isinstance(readiness_failures_raw, list)
        else []
    )
    readiness_gates_raw = payload.get("readiness_gates")
    readiness_gates = dict(readiness_gates_raw) if isinstance(readiness_gates_raw, dict) else {}
    oms_invariants_gate_raw = readiness_gates.get("oms_invariants")
    oms_invariants_gate = (
        dict(oms_invariants_gate_raw) if isinstance(oms_invariants_gate_raw, dict) else {}
    )
    oms_lifecycle_gate_raw = readiness_gates.get("oms_lifecycle_parity")
    oms_lifecycle_gate = (
        dict(oms_lifecycle_gate_raw) if isinstance(oms_lifecycle_gate_raw, dict) else {}
    )
    oms_invariants_required_effective = bool(
        oms_invariants_required["enabled"] or oms_invariants_gate.get("required", False)
    )
    oms_lifecycle_required_effective = bool(
        oms_lifecycle_required["enabled"] or oms_lifecycle_gate.get("required", False)
    )

    flat_start_blockers: list[str] = []
    if trust_runtime_flat_start:
        if runtime_open_orders_count is not None and runtime_open_orders_count > 0:
            flat_start_blockers.append("preopen_open_orders")
        if (
            runtime_unexpected_positions_count is not None
            and runtime_unexpected_positions_count > 0
        ):
            flat_start_blockers.append("preopen_non_flat_positions")
    else:
        if open_orders_count is not None and open_orders_count > 0:
            flat_start_blockers.append("preopen_open_orders")
        if positions_count is not None and positions_count > 0:
            flat_start_blockers.append("preopen_non_flat_positions")
        if (
            "market_closed_open_orders" in attention_flags
            and "preopen_open_orders" not in flat_start_blockers
        ):
            flat_start_blockers.append("preopen_open_orders")
        if (
            "market_closed_non_flat_positions" in attention_flags
            and "preopen_non_flat_positions" not in flat_start_blockers
        ):
            flat_start_blockers.append("preopen_non_flat_positions")

    oms_blockers = [
        failure
        for failure in readiness_failures
        if failure in {"oms_invariants_failed", "oms_lifecycle_parity_failed"}
    ]
    observed_oms_warnings: list[str] = []
    for gate_name in ("oms_invariants", "oms_lifecycle_parity"):
        gate_raw = readiness_gates.get(gate_name)
        gate = dict(gate_raw) if isinstance(gate_raw, dict) else {}
        if gate.get("status") == "observed_failure":
            observed_oms_warnings.append(f"{gate_name}_observed_failure")
        if gate.get("status") == "required_failed":
            failure = f"{gate_name}_failed"
            if failure not in oms_blockers:
                oms_blockers.append(failure)

    blockers = [*flat_start_blockers, *oms_blockers]
    warnings = list(observed_oms_warnings)
    if flat_start_blockers and not bool(flat_start_required["enabled"]):
        warnings.extend(flat_start_blockers)
    status = "pass"
    if oms_blockers or (flat_start_blockers and bool(flat_start_required["enabled"])):
        status = "fail"
    elif warnings:
        status = "warn"

    flat_state = "clean" if not flat_start_blockers else ",".join(flat_start_blockers)
    oms_state = "clean" if not oms_blockers and not observed_oms_warnings else ",".join(
        [*oms_blockers, *observed_oms_warnings]
    )
    summary = (
        "pre-open drill "
        f"flat_start_required={str(bool(flat_start_required['enabled'])).lower()} "
        f"flat_state={flat_state} "
        f"oms_require_invariants={str(oms_invariants_required_effective).lower()} "
        f"oms_require_lifecycle={str(oms_lifecycle_required_effective).lower()} "
        f"oms_state={oms_state}"
    )

    return Step(
        name="preopen_operator_drill",
        status=status,
        summary=summary,
        details={
            "config": config,
            "broker": {
                "open_orders_count": open_orders_count,
                "positions_count": positions_count,
            },
            "flat_start": {
                "required": bool(flat_start_required["enabled"]),
                "blockers": flat_start_blockers,
                "open_orders_count": open_orders_count,
                "positions_count": positions_count,
                "runtime_context": runtime_flat_start,
            },
            "oms": {
                "require_invariants": oms_invariants_required_effective,
                "require_lifecycle_parity": oms_lifecycle_required_effective,
                "readiness_failures": readiness_failures,
                "readiness_gates": readiness_gates,
                "blockers": oms_blockers,
                "warnings": observed_oms_warnings,
            },
            "attention_flags": attention_flags,
            "blockers": blockers,
            "warnings": warnings,
        },
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
    ]
    health_step = _check_health(
        repo_dir,
        port=args.health_port,
        timeout_seconds=int(args.health_timeout_sec),
        skip=bool(args.skip_health),
    )
    steps.append(health_step)
    steps.append(_check_preopen_operator_drill(repo_dir, health_step=health_step))
    fail_count = sum(1 for step in steps if step.status == "fail")
    warn_count = sum(1 for step in steps if step.status == "warn")
    report: dict[str, Any] = {
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
