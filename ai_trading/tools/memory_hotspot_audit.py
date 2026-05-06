"""Audit runtime memory pressure and likely large-file hotspots."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path
from ai_trading.utils.memory_optimizer import report_memory_use

_DEFAULT_OUTPUT = "runtime/memory_hotspot_audit_latest.json"
_DEFAULT_SAMPLES = "runtime/memory_samples.jsonl"
_HOTSPOT_PATTERNS = (
    ".read_text(",
    ".read_bytes(",
    "json.loads(",
    "_read_jsonl(",
    "read_jsonl(",
    "pd.read_csv(",
    "pd.read_parquet(",
    "read_parquet(",
)
_SCAN_SUFFIXES = {".py", ".sh"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _bytes_to_mb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024.0 * 1024.0), 3)


def _parse_systemctl_bytes(raw: str | None) -> int | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value or value.lower() in {"infinity", "max"}:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _service_memory(service: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "service": service,
        "available": False,
        "memory_current_mb": None,
        "memory_peak_mb": None,
        "memory_max_mb": None,
        "tasks_current": None,
        "tasks_max": None,
    }
    try:
        completed = subprocess.run(
            [
                "systemctl",
                "show",
                service,
                "-p",
                "MemoryCurrent",
                "-p",
                "MemoryPeak",
                "-p",
                "MemoryMax",
                "-p",
                "TasksCurrent",
                "-p",
                "TasksMax",
                "--no-pager",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError):
        return result
    if completed.returncode != 0:
        result["error"] = completed.stderr.strip()[:300]
        return result
    fields: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key] = value
    result.update(
        {
            "available": True,
            "memory_current_mb": _bytes_to_mb(_parse_systemctl_bytes(fields.get("MemoryCurrent"))),
            "memory_peak_mb": _bytes_to_mb(_parse_systemctl_bytes(fields.get("MemoryPeak"))),
            "memory_max_mb": _bytes_to_mb(_parse_systemctl_bytes(fields.get("MemoryMax"))),
            "tasks_current": _parse_systemctl_bytes(fields.get("TasksCurrent")),
            "tasks_max": _parse_systemctl_bytes(fields.get("TasksMax")),
        }
    )
    current = result.get("memory_current_mb")
    maximum = result.get("memory_max_mb")
    if isinstance(current, float) and isinstance(maximum, float) and maximum > 0:
        result["memory_current_pct_of_max"] = round((current / maximum) * 100.0, 2)
    return result


def _largest_files(root: Path, *, limit: int) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        try:
            if not path.is_file():
                continue
            stat = path.stat()
        except OSError:
            continue
        rows.append(
            {
                "path": str(path),
                "size_mb": _bytes_to_mb(stat.st_size),
                "suffix": path.suffix.lower(),
            }
        )
    rows.sort(key=lambda row: float(row.get("size_mb") or 0.0), reverse=True)
    return rows[:limit]


def _tail_lines(path: Path, *, max_lines: int, max_bytes: int = 256_000) -> list[str]:
    if max_lines <= 0 or not path.exists():
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - max_bytes))
            data = handle.read(max_bytes)
    except OSError:
        return []
    text = data.decode("utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-max_lines:]


def _recent_samples(sample_path: Path, *, max_lines: int) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for line in _tail_lines(sample_path, max_lines=max_lines):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            samples.append(parsed)
    rss_values = [
        float(sample["rss_mb"])
        for sample in samples
        if isinstance(sample.get("rss_mb"), int | float)
    ]
    return {
        "path": str(sample_path),
        "rows": len(samples),
        "latest": samples[-1] if samples else {},
        "rss_min_mb": min(rss_values) if rss_values else None,
        "rss_max_mb": max(rss_values) if rss_values else None,
        "rss_delta_mb": round(rss_values[-1] - rss_values[0], 3) if len(rss_values) >= 2 else None,
    }


def _iter_scan_files(repo_root: Path) -> Iterable[Path]:
    for directory_name in ("ai_trading", "scripts", "tools"):
        directory = repo_root / directory_name
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.suffix in _SCAN_SUFFIXES and path.is_file():
                yield path


def _scan_code_hotspots(repo_root: Path, *, limit: int) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in _iter_scan_files(repo_root):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            matched = [pattern for pattern in _HOTSPOT_PATTERNS if pattern in line]
            if not matched:
                continue
            findings.append(
                {
                    "path": str(path),
                    "line": line_number,
                    "patterns": matched,
                    "snippet": line.strip()[:180],
                }
            )
            if len(findings) >= limit:
                return findings
    return findings


def _status_for_report(service_memory: dict[str, Any], sample_summary: dict[str, Any]) -> str:
    pct = service_memory.get("memory_current_pct_of_max")
    if isinstance(pct, int | float) and pct >= 85.0:
        return "critical"
    if isinstance(pct, int | float) and pct >= 70.0:
        return "watch"
    latest = sample_summary.get("latest")
    if isinstance(latest, dict) and str(latest.get("level") or "") in {"warning", "critical"}:
        return str(latest.get("level"))
    return "ok"


def build_memory_hotspot_audit(
    *,
    runtime_dir: Path,
    repo_root: Path,
    sample_path: Path,
    service: str,
    max_files: int = 20,
    max_code_findings: int = 80,
    sample_lines: int = 100,
) -> dict[str, Any]:
    service_memory = _service_memory(service)
    sample_summary = _recent_samples(sample_path, max_lines=sample_lines)
    runtime_files = _largest_files(runtime_dir, limit=max_files)
    code_hotspots = _scan_code_hotspots(repo_root, limit=max_code_findings)
    current_process = report_memory_use(write_sample=False)
    largest_runtime_file = runtime_files[0] if runtime_files else {}
    observations: list[str] = []
    if largest_runtime_file:
        observations.append("large_runtime_artifacts_present")
    if code_hotspots:
        observations.append("whole_file_reader_patterns_present")
    if sample_summary.get("rss_delta_mb") is not None:
        observations.append("memory_samples_available")
    status = _status_for_report(service_memory, sample_summary)
    return {
        "schema_version": "1.0.0",
        "artifact_type": "memory_hotspot_audit",
        "generated_at": _utc_now(),
        "status": status,
        "service_memory": service_memory,
        "current_audit_process": current_process,
        "runtime_dir": str(runtime_dir),
        "runtime_artifacts": {
            "largest_files": runtime_files,
            "largest_file": largest_runtime_file,
        },
        "recent_memory_samples": sample_summary,
        "code_hotspots": code_hotspots,
        "observations": observations,
        "recommended_actions": _recommended_actions(status, runtime_files, code_hotspots),
    }


def _recommended_actions(
    status: str,
    runtime_files: list[dict[str, Any]],
    code_hotspots: list[dict[str, Any]],
) -> list[str]:
    actions: list[str] = []
    if status in {"watch", "warning", "critical"}:
        actions.append("review service memory trend before next market session")
    if runtime_files:
        actions.append("archive or compact the largest runtime JSONL/report artifacts if they are no longer authoritative")
    if code_hotspots:
        actions.append("replace whole-file reads in high-volume runtime reports with streaming or bounded tail readers")
    if not actions:
        actions.append("continue collecting memory samples")
    return actions


def _resolve_path(value: str | None, default_relative: str, *, for_write: bool) -> Path:
    return resolve_runtime_artifact_path(value, default_relative=default_relative, for_write=for_write)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", default=None)
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    parser.add_argument("--sample-jsonl", default=None)
    parser.add_argument("--service", default="ai-trading.service")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--max-code-findings", type=int, default=80)
    parser.add_argument("--sample-lines", type=int, default=100)
    args = parser.parse_args(argv)

    runtime_dir = _resolve_path(args.runtime_dir, "runtime", for_write=False)
    sample_path = _resolve_path(args.sample_jsonl, _DEFAULT_SAMPLES, for_write=False)
    output_path = _resolve_path(args.output_json, _DEFAULT_OUTPUT, for_write=True)
    report = build_memory_hotspot_audit(
        runtime_dir=runtime_dir,
        repo_root=Path(args.repo_root).resolve(),
        sample_path=sample_path,
        service=str(args.service),
        max_files=max(1, int(args.max_files)),
        max_code_findings=max(1, int(args.max_code_findings)),
        sample_lines=max(0, int(args.sample_lines)),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    sys.stdout.write(
        json.dumps(
            {
                "status": report.get("status"),
                "output_json": str(output_path),
                "largest_runtime_file": report.get("runtime_artifacts", {}).get("largest_file", {}),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
