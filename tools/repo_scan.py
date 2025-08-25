#!/usr/bin/env python3
"""Lightweight repository scan for risky patterns (non-blocking)."""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
self_path = Path(__file__).resolve()
pyfiles = [p for p in ROOT.rglob("*.py") if p != self_path]

issues: list[tuple[Path, str]] = []
for path in pyfiles:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue

    # Literal ... (excluding ok comments or explicit Ellipsis)
    for match in re.finditer(r"^\s*\.\.\.\s*$", text, flags=re.M):
        line = text[: match.start()].splitlines()[-1] if match.start() else ""
        if "# ok: ellipsis" in line or "Ellipsis" in line:
            continue
        issues.append((path, "literal '...' in code"))

    # requests.* calls without timeout
    for m in re.finditer(r"requests\.(get|post|put|delete|patch)\(", text):
        end = text.find(")", m.start())
        if end != -1 and "timeout=" not in text[m.start() : end]:
            issues.append((path, f"requests.{m.group(1)} without timeout"))

    # bare except
    if re.search(r"\n\s*except\s*:\s*\n", text):
        issues.append((path, "bare except"))

    # pickle load/loads usage
    if re.search(r"\bpickle\.load(s)?\(", text):
        issues.append((path, "pickle load/loads present"))

if issues:
    print("Repo scan warnings:")
    for p, why in issues:
        try:
            rel = p.relative_to(ROOT)
        except Exception:
            rel = p
        print(f" - {rel} -> {why}")
else:
    print("Repo scan: clean ✅")

# Non-blocking: always exit success
sys.exit(0)

