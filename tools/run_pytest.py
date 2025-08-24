#!/usr/bin/env python
"""Deterministic pytest runner used by CI smoke tests."""

from __future__ import annotations

import argparse
import importlib.util as iu
import logging
import os
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Create argument parser for the runner."""
    p = argparse.ArgumentParser(description="Deterministic pytest runner")
    p.add_argument(
        "--disable-warnings",
        action="store_true",
        help="Silence warnings via -W ignore",
    )
    p.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect tests",
    )
    p.add_argument(
        "-k",
        dest="keyword",
        default=None,
        help="pytest -k expression (optional)",
    )
    p.add_argument(
        "-q",
        dest="quiet",
        action="store_true",
        help="Quiet mode (pytest -q)",
    )
    # AI-AGENT-REF: allow explicit test targets to avoid global collection
    p.add_argument(
        "targets",
        nargs="*",
        help="Explicit test file/dir/node-id targets",
    )
    return p


def build_pytest_cmd(args: argparse.Namespace) -> list[str]:
    """Construct the pytest command based on parsed arguments."""
    cmd = [sys.executable, "-m", "pytest"]
    if args.quiet:
        cmd.append("-q")
    if args.disable_warnings:
        # AI-AGENT-REF: suppress warnings for stable smoke output
        cmd += ["-W", "ignore"]
    if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1":
        # Under autoload-off, inject xdist only if available and not already present via addopts
        addopts = os.environ.get("PYTEST_ADDOPTS", "")
        if ("-p xdist.plugin" not in addopts) and (iu.find_spec("xdist") is not None):
            cmd += ["-p", "xdist.plugin", "-n", os.environ.get("PYTEST_XDIST_N", "auto")]
    if args.collect_only:
        cmd += ["--collect-only"]
    if args.targets:
        # AI-AGENT-REF: append explicit targets last so pytest limits collection
        cmd.extend(args.targets)
    elif args.keyword:
        cmd += ["-k", args.keyword]
    return cmd


def _ensure_repo_on_path() -> None:
    """Prepend repository root to sys.path and PYTHONPATH for deterministic imports."""
    # AI-AGENT-REF: prepend repo root so smoke tests import workspace modules
    repo_root = Path(__file__).resolve().parent.parent
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = repo_str if not existing else f"{repo_str}{os.pathsep}{existing}"
    try:
        import ai_trading  # type: ignore
        print(f"[run_pytest] ai_trading path -> {Path(ai_trading.__file__).resolve()}")
    except Exception as e:  # pragma: no cover - diagnostic only
        print(f"[run_pytest] ai_trading not importable yet ({e}); pytest will handle it", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("run_pytest")
    _ensure_repo_on_path()
    # AI-AGENT-REF: avoid third-party plugins influencing tests
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = build_pytest_cmd(args)
    # AI-AGENT-REF: echo exact command for smoke test assertions
    logger.info("[run_pytest] %s", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

