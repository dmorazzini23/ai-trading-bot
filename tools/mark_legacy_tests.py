#!/usr/bin/env python
from __future__ import annotations
import argparse, pathlib, re

LEGACY_PATTERNS = (
    r"ai_trading\.monitoring\.performance_monitor",
    r"ai_trading\.position\.core",
    r"ai_trading\.runtime\.http_wrapped",
)

DECORATOR = "@pytest.mark.legacy\n"
IMPORT = "import pytest\n"

def process(path: pathlib.Path) -> bool:
    txt = path.read_text(encoding="utf-8")
    if not re.search("|".join(LEGACY_PATTERNS), txt):
        return False
    if "pytest.mark.legacy" in txt:
        return False
    if "import pytest" not in txt:
        txt = IMPORT + txt
    txt = re.sub(r"(?m)^(def |class )", DECORATOR + r"\1", txt, count=1)
    path.write_text(txt, encoding="utf-8")
    return True

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("root", nargs="?", default="tests")
    args = ap.parse_args()
    changed = 0
    for p in pathlib.Path(args.root).rglob("test_*.py"):
        if args.apply:
            changed += bool(process(p))
    print(f"legacy-marked: {changed}")
