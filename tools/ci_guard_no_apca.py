#!/usr/bin/env python3
"""Fail if new legacy Alpaca environment literals are introduced outside allowed files."""

from __future__ import annotations

import codecs
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

LEGACY_PREFIX = codecs.decode("415043415f", "hex").decode("ascii")

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
        if LEGACY_PREFIX in text:
            print(f"Forbidden legacy prefix literal detected in {rel}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

