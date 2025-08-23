from __future__ import annotations
import re
import sys
from pathlib import Path

BLOCKLIST = [
    r"\bResourceMonitor\b",
    r"\bperformance_monitor\b",
    r"\bposition\.core\b",
    r"\bhttp_wrapped\b",
]
PAT = re.compile("|".join(f"(?:{p})" for p in BLOCKLIST))


def main() -> int:
    roots = [Path("ai_trading"), Path("tests")]
    bad: list[tuple[str, int, str]] = []
    for root in roots:
        for p in root.rglob("*.py"):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if PAT.search(line):
                    bad.append((str(p), i, line.strip()))
    if not bad:
        print("OK: no legacy shim symbols found")
        return 0
    print("ERROR: legacy shim symbols detected:")
    for f, ln, line in bad:
        print(f"  {f}:{ln}: {line}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
