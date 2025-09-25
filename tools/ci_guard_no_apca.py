#!/usr/bin/env python3
"""Fail if new APCA_* environment literals are introduced outside allowed files."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SKIP_PREFIXES = (
    "tests/",
    "artifacts/",
    "build/",
    ".git/",
)

ALLOW_FILES = {
    "README.md",
    "tools/env_doctor.py",
    "tools/ci_guard_no_apca.py",
}


def should_skip(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT).as_posix()
    if path.is_dir():
        return True
    if any(rel.startswith(prefix) for prefix in SKIP_PREFIXES):
        return True
    if rel in ALLOW_FILES:
        return True
    if rel.endswith(('.png', '.jpg', '.jpeg', '.gif', '.zip', '.whl', '.pyc')):
        return True
    return False


def main() -> int:
    failed = False
    for path in REPO_ROOT.rglob("*"):
        if should_skip(path):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "APCA_" in text:
            print(f"Forbidden 'APCA_' literal detected in {rel}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

