#!/usr/bin/env python3
"""Inspect the environment and common config files for legacy APCA_* entries."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

SEARCH_PATHS = [
    Path(".env"),
    Path("/etc/systemd/system/ai-trading.service"),
    *(Path("/etc/systemd/system").glob("ai-trading.service.d/*.conf")),
    Path.home() / ".bashrc",
    Path.home() / ".profile",
    Path("/etc/environment"),
]


def find_apca_in_file(path: Path) -> list[tuple[int, str]]:
    """Return (line_number, line) pairs for each occurrence of ``APCA_`` in ``path``."""

    hits: list[tuple[int, str]] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for lineno, line in enumerate(handle, 1):
                if "APCA_" in line:
                    hits.append((lineno, line.rstrip()))
    except FileNotFoundError:
        return []
    except PermissionError:
        return [(0, "<permission denied>")]
    return hits


def _print_section(title: str) -> None:
    print(title)
    print("=" * len(title))


def _print_env_hits(keys: Iterable[str]) -> None:
    entries = list(keys)
    _print_section("Environment variables")
    if not entries:
        print("(none)")
        return
    for key in entries:
        print(f"- {key}")


def _print_file_hits(results: Iterable[tuple[Path, list[tuple[int, str]]]]) -> None:
    _print_section("Filesystem scan")
    any_hits = False
    for path, hits in results:
        if not hits:
            continue
        any_hits = True
        print(f"\n{path}:")
        for lineno, line in hits:
            if lineno == 0:
                print("  [permission denied]")
            else:
                print(f"  {lineno:4d}: {line}")
    if not any_hits:
        print("No APCA_* entries found in scanned files.")


def main() -> int:
    apca_keys = sorted(key for key in os.environ if key.startswith("APCA_"))
    file_results = [(path, find_apca_in_file(path)) for path in SEARCH_PATHS]

    has_hits = bool(apca_keys) or any(hits for _, hits in file_results)

    _print_env_hits(apca_keys)
    print()
    _print_file_hits(file_results)

    if has_hits:
        print(
            "\nRemediation: Replace all APCA_* entries with their ALPACA_* equivalents. "
            "After editing, run 'make doctor' again to confirm the environment is clean."
        )
        return 2

    print("\nNo APCA_* variables detected. Environment looks clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

