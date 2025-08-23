#!/usr/bin/env python3
import sys
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLOCKED = (
    r"ai_trading\.monitoring\.performance_monitor",
    r"\bResourceMonitor\b",
    r"\bperformance_monitor\b",
    r"ai_trading\.position\.core",
    r"ai_trading\.runtime\.http_wrapped",
)
rx = re.compile("|".join(BLOCKED))
violations = []
for p in ROOT.rglob("*.py"):
    if p.parts[0] == ".venv" or "/venv/" in str(p):
        continue
    if p.name in {"mark_legacy_tests.py", "repair_test_imports.py", "check_no_legacy_symbols.py"} and p.parent.name == "tools":
        continue
    text = p.read_text(encoding="utf-8", errors="ignore")
    if rx.search(text):
        violations.append(str(p))
if violations:
    print("Legacy symbols detected:\n" + "\n".join(sorted(violations)))
    sys.exit(2)
