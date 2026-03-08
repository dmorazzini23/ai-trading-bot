#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATTERN = re.compile(r"\bos\.(?:getenv|environ)\b")

# Tiny allowlist for config/bootstrap and operator tooling.
ALLOWLIST_PREFIXES = (
    "ai_trading/config/",
    "ai_trading/scripts/",
    "ai_trading/tools/",
    "ai_trading/validation/",
)
ALLOWLIST_FILES: set[str] = set()


def _is_allowlisted(rel_path: str) -> bool:
    if rel_path in ALLOWLIST_FILES:
        return True
    return rel_path.startswith(ALLOWLIST_PREFIXES)


def main() -> int:
    failures: list[str] = []
    for path in sorted((ROOT / "ai_trading").rglob("*.py")):
        rel_path = path.relative_to(ROOT).as_posix()
        if _is_allowlisted(rel_path):
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if PATTERN.search(line):
                failures.append(f"{rel_path}:{line_number}: {line.strip()}")

    if failures:
        print("RUNTIME_ENV_ACCESS_GUARD_FAILED")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("RUNTIME_ENV_ACCESS_GUARD_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
