#!/usr/bin/env python3
"""Fail CI when tracked files include likely live credentials."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SENSITIVE_KEYS = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "WEBHOOK_SECRET",
    "NEWS_API_KEY",
    "SENTIMENT_API_KEY",
    "FINNHUB_API_KEY",
    "IEX_API_TOKEN",
    "API_SECRET_KEY",
)

ASSIGNMENT_RE = re.compile(
    r"^\s*(?:export\s+)?(?P<key>[A-Z0-9_]+)\s*[:=]\s*(?P<value>.+?)\s*$"
)

SKIP_PREFIXES = (
    ".git/",
    ".venv/",
    "venv/",
    "build/",
    "dist/",
    "__pycache__/",
    "artifacts/",
    "tests/",
)

SKIP_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".whl",
    ".zip",
    ".pyc",
)

SCAN_SUFFIXES = (
    ".env",
    ".txt",
    ".md",
    ".rst",
    ".ini",
    ".cfg",
    ".conf",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".sh",
)

SCAN_FILENAMES = (
    ".env",
    ".env.example",
    ".env.local",
    ".env.production",
    ".env.prod",
    ".env.dev",
)

PLACEHOLDER_TOKENS = (
    "dummy",
    "test",
    "fake",
    "example",
    "sample",
    "changeme",
    "replace",
    "your_",
    "your-",
    "<",
    ">",
    "token_here",
    "secret_here",
    "not_real",
    "redacted",
    "masked",
)


def _git_tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _should_skip(rel_path: str) -> bool:
    if any(rel_path.startswith(prefix) for prefix in SKIP_PREFIXES):
        return True
    if any(rel_path.endswith(suffix) for suffix in SKIP_SUFFIXES):
        return True
    return False


def _is_scan_candidate(rel_path: str) -> bool:
    path = Path(rel_path)
    if path.name in SCAN_FILENAMES:
        return True
    return path.suffix.lower() in SCAN_SUFFIXES


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if char == "#" and not in_single and not in_double:
            return value[:idx].strip()
    return value.strip()


def _normalize_value(raw: str) -> str:
    value = _strip_inline_comment(raw).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value


def _looks_placeholder(value: str) -> bool:
    lowered = value.lower()
    if not lowered:
        return True
    if lowered.startswith("$"):
        return True
    if "${{ secrets." in lowered:
        return True
    if all(char in {"*", "x", "X", "-", "_"} for char in value):
        return True
    return any(token in lowered for token in PLACEHOLDER_TOKENS)


def _scan_file(path: Path, rel_path: str) -> list[tuple[str, int, str, str]]:
    findings: list[tuple[str, int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return findings

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ASSIGNMENT_RE.match(raw_line)
        if not match:
            continue
        key = match.group("key")
        if key not in SENSITIVE_KEYS:
            continue
        value = _normalize_value(match.group("value"))
        if _looks_placeholder(value):
            continue
        findings.append((rel_path, line_no, key, value))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail if tracked files contain likely live secrets.")
    parser.add_argument("--root", default=".", help="Repository root (default: current directory).")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    all_findings: list[tuple[str, int, str, str]] = []

    try:
        rel_paths = _git_tracked_files(root)
    except subprocess.CalledProcessError as exc:
        print(f"Unable to enumerate tracked files: {exc}", file=sys.stderr)
        return 2

    for rel_path in rel_paths:
        if _should_skip(rel_path):
            continue
        if not _is_scan_candidate(rel_path):
            continue
        file_path = root / rel_path
        if not file_path.is_file():
            continue
        all_findings.extend(_scan_file(file_path, rel_path))

    if not all_findings:
        print("OK: no likely live secrets detected in tracked files")
        return 0

    print("Potential live secrets detected:", file=sys.stderr)
    for rel_path, line_no, key, value in all_findings:
        masked = "***" if len(value) >= 8 else value
        print(f"  - {rel_path}:{line_no} {key}={masked}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
