#!/usr/bin/env python
"""Deterministic pytest runner used by CI smoke tests."""

from __future__ import annotations

import argparse
import importlib.util as iu
import logging
import os
import shlex
import subprocess
import sys


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
    cmd = [sys.executable, "-m", "pytest", "-q"]
    if args.disable_warnings:
        # AI-AGENT-REF: map --disable-warnings to interpreter flag
        cmd += ["-W", "ignore"]
    if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1":
        addopts = os.environ.get("PYTEST_ADDOPTS", "")
        xdist_requested = "-p xdist.plugin" in addopts or "xdist.plugin" in addopts
        if iu.find_spec("xdist") is not None and not xdist_requested:
            # AI-AGENT-REF: load xdist only when autoload is off and not already requested
            cmd += ["-p", "xdist.plugin", "-n", os.environ.get("PYTEST_XDIST_WORKERS", "auto")]
    if args.collect_only:
        cmd += ["--collect-only"]
    if args.keyword:
        cmd += ["-k", args.keyword]
    if args.targets:
        # AI-AGENT-REF: append explicit targets last so pytest limits collection
        cmd.extend(args.targets)
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = build_pytest_cmd(args)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("run_pytest")
    logger.info("[run_pytest] %s", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

