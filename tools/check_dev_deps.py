"""Verify dev-time imports for packages used in tests.

Run:
    python tools/check_dev_deps.py

Exits with status 0 if all imports succeed; otherwise prints missing deps.
"""

from __future__ import annotations

REQUIRED = [
    ("cachetools", "cachetools"),
    ("torch", "torch"),
    ("stable_baselines3", "stable_baselines3"),
    ("gymnasium", "gymnasium"),
    # Both SDK lines should import cleanly during collection
    ("alpaca-py", "alpaca"),  # AI-AGENT-REF: modern SDK top-level name is `alpaca`
]


def main() -> None:
    missing = []
    for pkg_name, import_name in REQUIRED:
        try:
            __import__(import_name)
        except Exception as e:  # import-time errors only
            missing.append((pkg_name, import_name, str(e)))
    if missing:
        print("Missing dev deps (package -> import):")
        for pkg_name, import_name, err in missing:
            print(f" - {pkg_name} -> {import_name}: {err}")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()

