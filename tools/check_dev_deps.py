"""Tiny sanity check for dev ML dependencies used by tests.
Run: python tools/check_dev_deps.py
Exits 0 if all imports succeed; prints a minimal status line.
"""
from __future__ import annotations

import sys

def main() -> int:
    missing: list[str] = []
    for name in ("stable_baselines3", "gymnasium", "cloudpickle", "tqdm"):
        try:
            __import__(name)
        except Exception as e:  # precise intent: import-only verification
            missing.append(f"{name}: {e}")
    if missing:
        print("MISSING:")
        for m in missing:
            print(" -", m)
        return 1
    print("dev-ml-deps: OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
