"""Persist a durable, non-authoritative record of a completed full validation."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ai_trading.runtime.atomic_io import atomic_write_text
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


def build_full_validation_artifact(*, commit: str, generated_at: datetime | None = None) -> dict[str, object]:
    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    return {
        "schema_version": "1.0.0",
        "artifact_type": "full_validation_result",
        "generated_at": generated.isoformat().replace("+00:00", "Z"),
        "full_validation_green": True,
        "commit": str(commit or "").strip() or None,
        "commands": [
            "pytest -q",
            "ruff check",
            "mypy",
            "scripts/typecheck_strict.sh",
            "python -m py_compile",
        ],
        "promotion_authority": False,
        "live_money_authority": False,
    }


def _default_output() -> Path:
    return resolve_runtime_artifact_path(
        "runtime/full_validation_green_latest.json",
        default_relative="runtime/full_validation_green_latest.json",
        for_write=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = ""
    output = args.output_json or _default_output()
    payload = build_full_validation_artifact(commit=commit)
    atomic_write_text(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
