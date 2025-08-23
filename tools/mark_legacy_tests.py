from __future__ import annotations
import argparse
import re
from pathlib import Path

LEGACY_PATTERNS = [
    r"\bai_trading\.monitoring\.performance_monitor\b",
    r"\bResourceMonitor\b",
    r"\bai_trading\.position\.core\b",
    r"\bai_trading\.runtime\.http_wrapped\b",
]

HEADER = "import pytest\npytestmark = pytest.mark.legacy  # added by mark_legacy_tests\n"

def has_marker(text: str) -> bool:
    return "pytestmark = pytest.mark.legacy" in text

def insert_header(text: str) -> str:
    # Insert after shebang or encoding or initial docstring if present, else at top.
    lines = text.splitlines(True)
    idx = 0
    while idx < len(lines) and (lines[idx].startswith("#!") or "coding" in lines[idx]):
        idx += 1
    if idx < len(lines) and lines[idx].lstrip().startswith(('"""', "'''")):
        quote = lines[idx].strip()[:3]
        idx += 1
        while idx < len(lines) and quote not in lines[idx]:
            idx += 1
        if idx < len(lines):
            idx += 1
    return "".join(lines[:idx] + [HEADER] + lines[idx:])

def matches_legacy(text: str) -> bool:
    pat = re.compile("|".join(f"(?:{p})" for p in LEGACY_PATTERNS))
    return bool(pat.search(text))

def unmark(text: str) -> str:
    return text.replace(HEADER, "")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="tests", help="directory to scan")
    ap.add_argument("--apply", action="store_true", help="write markers to files")
    ap.add_argument("--undo", action="store_true", help="remove previously added markers")
    args = ap.parse_args()

    root = Path(args.root)
    changed = 0
    for p in root.rglob("test_*.py"):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if args.undo:
            new = unmark(txt)
            if new != txt:
                p.write_text(new, encoding="utf-8")
                changed += 1
            continue

        if not matches_legacy(txt):
            continue
        if has_marker(txt):
            continue
        if args.apply:
            p.write_text(insert_header(txt), encoding="utf-8")
        changed += 1

    print(("Marked" if args.apply else "Would mark") if not args.undo else "Unmarked", changed, "files.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
