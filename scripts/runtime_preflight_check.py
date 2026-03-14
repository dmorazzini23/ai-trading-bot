#!/usr/bin/env python3
"""One-shot runtime and model preflight checks.

This script is intended for weekend/overnight operational checks before the
next trading session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class CheckResult:
    name: str
    status: str  # pass | warn | fail
    summary: str
    details: dict[str, Any]


def _status_emoji(status: str) -> str:
    if status == "pass":
        return "PASS"
    if status == "warn":
        return "WARN"
    return "FAIL"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _iter_nonempty_lines(path: Path) -> tuple[int, list[str], list[str]]:
    """Return line count, head sample, tail sample (non-empty lines only)."""
    head: list[str] = []
    tail: deque[str] = deque(maxlen=25)
    lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            lines += 1
            if len(head) < 25:
                head.append(line)
            tail.append(line)
    return lines, head, list(tail)


def _parse_single_json_object(line: str) -> tuple[bool, str | None]:
    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(line, 0)
        while end < len(line) and line[end].isspace():
            end += 1
        if end != len(line):
            return False, "extra_data_after_json_object"
    except Exception as exc:  # pragma: no cover - defensive
        return False, str(exc)
    return True, None


def check_json_files(runtime_dir: Path, models_dir: Path) -> CheckResult:
    targets = sorted(runtime_dir.glob("*.json"))
    targets.extend(sorted(models_dir.glob("*.json")))
    targets.extend(sorted((models_dir / "runtime").glob("*.json")))
    targets.extend(sorted((models_dir / "after_hours").glob("*.manifest.json")))
    bad: list[dict[str, str]] = []
    for path in targets:
        try:
            _load_json(path)
        except Exception as exc:
            bad.append({"path": str(path), "error": str(exc)})
    status = "pass" if not bad else "fail"
    summary = f"parsed {len(targets) - len(bad)}/{len(targets)} JSON files"
    return CheckResult(
        name="json_files",
        status=status,
        summary=summary,
        details={"total": len(targets), "bad": bad[:20]},
    )


def check_run_manifest_jsonl(path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult(
            name="run_manifest_jsonl",
            status="warn",
            summary="run manifest jsonl not found",
            details={"path": str(path)},
        )
    problems: list[dict[str, Any]] = []
    objects = 0
    lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, 1):
            lines += 1
            line = raw.strip()
            if not line:
                continue
            ok, err = _parse_single_json_object(line)
            if not ok:
                problems.append({"line": line_no, "error": err, "preview": line[:140]})
            else:
                objects += 1
    status = "pass" if not problems else "fail"
    summary = f"{objects} JSON objects across {lines} lines"
    return CheckResult(
        name="run_manifest_jsonl",
        status=status,
        summary=summary,
        details={"path": str(path), "problems": problems[:20]},
    )


def check_key_jsonl(runtime_dir: Path, sample_size: int) -> CheckResult:
    paths = [
        runtime_dir / "decision_records.jsonl",
        runtime_dir / "config_snapshots.jsonl",
        runtime_dir / "gate_effectiveness.jsonl",
        runtime_dir / "ml_shadow_predictions.jsonl",
        runtime_dir / "oms_ledger.jsonl",
        runtime_dir / "order_events.jsonl",
        runtime_dir / "fill_events.jsonl",
        runtime_dir / "tca_records.jsonl",
    ]
    failures: list[dict[str, Any]] = []
    info: dict[str, Any] = {}
    for path in paths:
        entry: dict[str, Any] = {"exists": path.exists(), "path": str(path)}
        if not path.exists():
            info[path.name] = entry
            continue
        size = path.stat().st_size
        entry["size"] = size
        if size == 0:
            entry["lines"] = 0
            info[path.name] = entry
            continue
        lines, head, tail = _iter_nonempty_lines(path)
        entry["lines"] = lines
        head_bad = 0
        tail_bad = 0
        for line in head[:sample_size]:
            try:
                json.loads(line)
            except Exception:
                head_bad += 1
        for line in tail[-sample_size:]:
            try:
                json.loads(line)
            except Exception:
                tail_bad += 1
        entry["head_bad"] = head_bad
        entry["tail_bad"] = tail_bad
        if head_bad or tail_bad:
            failures.append(
                {
                    "file": path.name,
                    "head_bad": head_bad,
                    "tail_bad": tail_bad,
                }
            )
        info[path.name] = entry
    status = "pass" if not failures else "fail"
    summary = "head/tail JSONL samples parsed cleanly" if status == "pass" else "JSONL parse issues found"
    return CheckResult(
        name="key_jsonl",
        status=status,
        summary=summary,
        details={"files": info, "failures": failures},
    )


def check_model_artifacts(models_dir: Path) -> CheckResult:
    try:
        import joblib  # type: ignore
    except Exception as exc:
        return CheckResult(
            name="model_artifacts",
            status="fail",
            summary="joblib unavailable",
            details={"error": str(exc)},
        )

    pkl_files = sorted(models_dir.glob("*.pkl"))
    joblib_files = sorted((models_dir / "runtime").glob("*.joblib"))
    joblib_files.extend(sorted((models_dir / "after_hours").glob("*.joblib")))
    bad: list[dict[str, str]] = []
    loaded = 0
    for path in pkl_files + joblib_files:
        try:
            _ = joblib.load(path)
            loaded += 1
        except Exception as exc:
            bad.append({"path": str(path), "error": str(exc)})
    status = "pass" if not bad else "fail"
    summary = f"loaded {loaded}/{len(pkl_files) + len(joblib_files)} model artifacts"
    return CheckResult(
        name="model_artifacts",
        status=status,
        summary=summary,
        details={
            "pkl_count": len(pkl_files),
            "joblib_count": len(joblib_files),
            "bad": bad[:20],
        },
    )


def check_model_checksums(models_dir: Path) -> CheckResult:
    pairs = [
        (
            models_dir / "runtime" / "ml_latest.joblib",
            models_dir / "runtime" / "ml_latest.joblib.manifest.json",
        ),
        (
            models_dir / "trained_model.pkl",
            models_dir / "trained_model.pkl.manifest.json",
        ),
    ]
    failures: list[dict[str, Any]] = []
    checked = 0
    for model_path, manifest_path in pairs:
        if not model_path.exists() or not manifest_path.exists():
            continue
        checked += 1
        try:
            manifest = _load_json(manifest_path)
            expected = str(manifest.get("checksum_sha256") or "").strip()
            actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if not expected or expected != actual:
                failures.append(
                    {
                        "model": str(model_path),
                        "manifest": str(manifest_path),
                        "expected": expected,
                        "actual": actual,
                    }
                )
        except Exception as exc:
            failures.append({"model": str(model_path), "error": str(exc)})
    if checked == 0:
        return CheckResult(
            name="model_checksums",
            status="warn",
            summary="no model checksum pairs found",
            details={},
        )
    status = "pass" if not failures else "fail"
    summary = f"validated {checked} checksum pair(s)"
    return CheckResult(
        name="model_checksums",
        status=status,
        summary=summary,
        details={"failures": failures},
    )


def check_report_freshness(runtime_dir: Path) -> CheckResult:
    perf = runtime_dir / "runtime_performance_report_latest.json"
    gate = runtime_dir / "gate_effectiveness_summary.json"
    if not perf.exists() or not gate.exists():
        return CheckResult(
            name="report_freshness",
            status="warn",
            summary="cannot compare report freshness (missing file)",
            details={"perf_exists": perf.exists(), "gate_exists": gate.exists()},
        )
    perf_mtime = datetime.fromtimestamp(perf.stat().st_mtime, tz=UTC)
    gate_mtime = datetime.fromtimestamp(gate.stat().st_mtime, tz=UTC)
    lag = gate_mtime - perf_mtime
    if lag > timedelta(hours=1):
        return CheckResult(
            name="report_freshness",
            status="warn",
            summary="runtime performance report lags gate summary",
            details={
                "perf_mtime": perf_mtime.isoformat(),
                "gate_mtime": gate_mtime.isoformat(),
                "lag_seconds": int(lag.total_seconds()),
            },
        )
    return CheckResult(
        name="report_freshness",
        status="pass",
        summary="runtime performance report is fresh",
        details={
            "perf_mtime": perf_mtime.isoformat(),
            "gate_mtime": gate_mtime.isoformat(),
            "lag_seconds": int(lag.total_seconds()),
        },
    )


def _extract_last_json_object(text: str) -> dict[str, Any] | None:
    last_obj: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            last_obj = obj
    return last_obj


def check_gonogo(repo_dir: Path) -> CheckResult:
    cmd = [sys.executable, "-m", "ai_trading.tools.runtime_gonogo_status", "--json"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            check=False,
            text=True,
            capture_output=True,
            timeout=60,
        )
    except Exception as exc:
        return CheckResult(
            name="gonogo_status",
            status="warn",
            summary="unable to run runtime_gonogo_status",
            details={"error": str(exc)},
        )
    payload = _extract_last_json_object(proc.stdout)
    if not payload:
        return CheckResult(
            name="gonogo_status",
            status="warn",
            summary="no JSON payload parsed from runtime_gonogo_status output",
            details={"exit_code": proc.returncode, "stdout_tail": proc.stdout[-500:]},
        )
    gate_passed = bool(payload.get("gate_passed"))
    status = "pass" if gate_passed else "warn"
    summary = "go/no-go PASS" if gate_passed else "go/no-go FAIL"
    details = {
        "exit_code": proc.returncode,
        "failed_checks": payload.get("failed_checks", []),
        "observed": payload.get("observed", {}),
    }
    return CheckResult(name="gonogo_status", status=status, summary=summary, details=details)


def check_event_stream_health(runtime_dir: Path) -> CheckResult:
    order_events = runtime_dir / "order_events.jsonl"
    fill_events = runtime_dir / "fill_events.jsonl"
    tca_records = runtime_dir / "tca_records.jsonl"
    decision_records = runtime_dir / "decision_records.jsonl"
    run_manifest = runtime_dir / "run_manifest.json"

    def _size(path: Path) -> int:
        return path.stat().st_size if path.exists() else -1

    mode = None
    if run_manifest.exists():
        try:
            mode = str(_load_json(run_manifest).get("mode"))
        except Exception:
            mode = None

    def _parse_decision_ts(row: dict[str, Any]) -> datetime | None:
        for key in ("ts", "timestamp", "bar_ts", "decision_ts"):
            value = row.get(key)
            if not isinstance(value, str):
                continue
            text = value.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(text)
            except Exception:
                continue
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        return None

    decision_total = 0
    decision_order_nonnull = 0
    decision_recent_total = 0
    decision_order_nonnull_recent = 0
    recent_window_hours = 24
    recent_cutoff = datetime.now(tz=UTC) - timedelta(hours=recent_window_hours)
    if decision_records.exists() and decision_records.stat().st_size > 0:
        with decision_records.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                decision_total += 1
                has_order = row.get("order") not in (None, {}, [])
                if has_order:
                    decision_order_nonnull += 1
                decision_ts = _parse_decision_ts(row)
                if decision_ts and decision_ts >= recent_cutoff:
                    decision_recent_total += 1
                    if has_order:
                        decision_order_nonnull_recent += 1

    event_sizes = {
        "order_events_size": _size(order_events),
        "fill_events_size": _size(fill_events),
        "tca_records_size": _size(tca_records),
    }
    all_empty = (
        event_sizes["order_events_size"] == 0
        and event_sizes["fill_events_size"] == 0
        and event_sizes["tca_records_size"] == 0
    )
    if all_empty and decision_order_nonnull_recent > 0:
        return CheckResult(
            name="event_stream_health",
            status="fail",
            summary="event streams empty despite recent non-null orders in decision records",
            details={
                **event_sizes,
                "mode": mode,
                "decision_total": decision_total,
                "decision_order_nonnull": decision_order_nonnull,
                "decision_recent_total": decision_recent_total,
                "decision_order_nonnull_recent": decision_order_nonnull_recent,
                "recent_window_hours": recent_window_hours,
            },
        )
    if all_empty and decision_order_nonnull > 0:
        return CheckResult(
            name="event_stream_health",
            status="warn",
            summary="event streams empty; non-null orders are outside recent window",
            details={
                **event_sizes,
                "mode": mode,
                "decision_total": decision_total,
                "decision_order_nonnull": decision_order_nonnull,
                "decision_recent_total": decision_recent_total,
                "decision_order_nonnull_recent": decision_order_nonnull_recent,
                "recent_window_hours": recent_window_hours,
            },
        )
    if all_empty:
        return CheckResult(
            name="event_stream_health",
            status="warn",
            summary="order/fill/tca streams are empty (no executed orders observed)",
            details={
                **event_sizes,
                "mode": mode,
                "decision_total": decision_total,
                "decision_order_nonnull": decision_order_nonnull,
                "decision_recent_total": decision_recent_total,
                "decision_order_nonnull_recent": decision_order_nonnull_recent,
                "recent_window_hours": recent_window_hours,
            },
        )
    return CheckResult(
        name="event_stream_health",
        status="pass",
        summary="event streams populated",
        details={
            **event_sizes,
            "mode": mode,
            "decision_total": decision_total,
            "decision_order_nonnull": decision_order_nonnull,
            "decision_recent_total": decision_recent_total,
            "decision_order_nonnull_recent": decision_order_nonnull_recent,
            "recent_window_hours": recent_window_hours,
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default="/var/lib/ai-trading-bot",
        help="Runtime base directory (default: /var/lib/ai-trading-bot).",
    )
    parser.add_argument(
        "--repo-dir",
        default=".",
        help="Repository root for module-based checks (default: current directory).",
    )
    parser.add_argument(
        "--sample-lines",
        type=int,
        default=20,
        help="Head/tail sample size for JSONL checks (default: 20).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON report.",
    )
    parser.add_argument(
        "--strict-warn",
        action="store_true",
        help="Exit non-zero when warnings exist.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    base_dir = Path(args.base_dir).resolve()
    runtime_dir = base_dir / "runtime"
    models_dir = base_dir / "models"
    repo_dir = Path(args.repo_dir).resolve()

    checks = [
        check_json_files(runtime_dir, models_dir),
        check_run_manifest_jsonl(runtime_dir / "run_manifest.jsonl"),
        check_key_jsonl(runtime_dir, sample_size=max(1, int(args.sample_lines))),
        check_model_artifacts(models_dir),
        check_model_checksums(models_dir),
        check_report_freshness(runtime_dir),
        check_gonogo(repo_dir),
        check_event_stream_health(runtime_dir),
    ]

    pass_count = sum(1 for c in checks if c.status == "pass")
    warn_count = sum(1 for c in checks if c.status == "warn")
    fail_count = sum(1 for c in checks if c.status == "fail")

    report = {
        "ts": datetime.now(tz=UTC).isoformat(),
        "base_dir": str(base_dir),
        "repo_dir": str(repo_dir),
        "summary": {
            "pass": pass_count,
            "warn": warn_count,
            "fail": fail_count,
        },
        "checks": [asdict(c) for c in checks],
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"RUNTIME_PREFLIGHT ts={report['ts']} base_dir={base_dir}")
        for check in checks:
            print(f"{_status_emoji(check.status):>4} {check.name}: {check.summary}")
        print(f"SUMMARY pass={pass_count} warn={warn_count} fail={fail_count}")

    if fail_count > 0:
        return 1
    if args.strict_warn and warn_count > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
