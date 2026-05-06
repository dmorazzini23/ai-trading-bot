"""Plan or apply bounded retention for high-volume runtime artifacts."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_trading.runtime.artifacts import resolve_runtime_artifact_path

_DEFAULT_OUTPUT = "runtime/runtime_artifact_retention_latest.json"


@dataclass(frozen=True)
class RetentionRule:
    filename: str
    max_bytes: int
    keep_lines: int


_DEFAULT_RULES: tuple[RetentionRule, ...] = (
    RetentionRule("decision_records.jsonl", 268_435_456, 50_000),
    RetentionRule("config_snapshots.jsonl", 268_435_456, 30_000),
    RetentionRule("gate_effectiveness.jsonl", 268_435_456, 300_000),
    RetentionRule("ml_shadow_predictions.jsonl", 268_435_456, 100_000),
    RetentionRule("order_events.jsonl", 134_217_728, 150_000),
    RetentionRule("fill_events.jsonl", 134_217_728, 150_000),
    RetentionRule("tca_records.jsonl", 134_217_728, 150_000),
    RetentionRule("oms_events.jsonl", 67_108_864, 100_000),
    RetentionRule("memory_samples.jsonl", 5_000_000, 20_000),
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _bytes_to_mb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024.0 * 1024.0), 3)


def _tail_lines_to_file(src: Path, dst: Path, *, keep_lines: int) -> int:
    if keep_lines <= 0:
        dst.write_text("", encoding="utf-8")
        return 0
    # Read from the end in bounded chunks so large JSONL files are not loaded all at once.
    chunks: list[bytes] = []
    newline_count = 0
    block_size = 1024 * 1024
    with src.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        while position > 0 and newline_count <= keep_lines:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
    data = b"".join(reversed(chunks))
    lines = data.splitlines()
    kept = lines[-keep_lines:]
    with dst.open("wb") as handle:
        for line in kept:
            handle.write(line)
            handle.write(b"\n")
    return len(kept)


def _gzip_backup(path: Path) -> Path | None:
    gz_path = path.with_name(f"{path.name}.gz")
    try:
        with path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        path.unlink()
    except OSError:
        return None
    return gz_path


def _apply_rule(path: Path, rule: RetentionRule) -> dict[str, Any]:
    before_size = path.stat().st_size
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.bak.{timestamp}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        kept_lines = _tail_lines_to_file(path, tmp_path, keep_lines=rule.keep_lines)
        os.replace(path, backup)
        os.replace(tmp_path, path)
        try:
            shutil.copymode(backup, path)
        except OSError:
            pass
        gz_backup = _gzip_backup(backup)
        after_size = path.stat().st_size
        return {
            "status": "compacted",
            "before_size_mb": _bytes_to_mb(before_size),
            "after_size_mb": _bytes_to_mb(after_size),
            "bytes_reclaimed": max(0, before_size - after_size),
            "kept_lines": kept_lines,
            "backup": str(gz_backup or backup),
        }
    except OSError as exc:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        return {
            "status": "error",
            "error": str(exc),
            "before_size_mb": _bytes_to_mb(before_size),
        }


def evaluate_runtime_artifact_retention(
    *,
    runtime_dir: Path,
    apply: bool = False,
    rules: tuple[RetentionRule, ...] = _DEFAULT_RULES,
) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    total_reclaimable = 0
    for rule in rules:
        path = runtime_dir / rule.filename
        action: dict[str, Any] = {
            "path": str(path),
            "filename": rule.filename,
            "max_size_mb": _bytes_to_mb(rule.max_bytes),
            "keep_lines": rule.keep_lines,
            "exists": path.exists(),
            "status": "missing",
        }
        if not path.exists():
            actions.append(action)
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            action.update({"status": "error", "error": str(exc)})
            actions.append(action)
            continue
        action["size_mb"] = _bytes_to_mb(size)
        if size <= rule.max_bytes:
            action["status"] = "within_limit"
            actions.append(action)
            continue
        action["status"] = "would_compact"
        action["bytes_over_limit"] = size - rule.max_bytes
        total_reclaimable += size - rule.max_bytes
        if apply:
            action.update(_apply_rule(path, rule))
        actions.append(action)
    status = "applied" if apply else "planned"
    if not any(action.get("status") in {"would_compact", "compacted", "error"} for action in actions):
        status = "ok"
    return {
        "schema_version": "1.0.0",
        "artifact_type": "runtime_artifact_retention",
        "generated_at": _utc_now(),
        "runtime_dir": str(runtime_dir),
        "apply": bool(apply),
        "status": status,
        "total_reclaimable_mb": _bytes_to_mb(total_reclaimable),
        "actions": actions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    runtime_dir = resolve_runtime_artifact_path(
        args.runtime_dir,
        default_relative="runtime",
        for_write=False,
    )
    output_path = resolve_runtime_artifact_path(
        args.output_json,
        default_relative=_DEFAULT_OUTPUT,
        for_write=True,
    )
    report = evaluate_runtime_artifact_retention(
        runtime_dir=runtime_dir,
        apply=bool(args.apply),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"status": report["status"], "output_json": str(output_path)}) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
