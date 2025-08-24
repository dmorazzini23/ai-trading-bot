#!/usr/bin/env python
"""Deterministic pytest runner used by CI smoke tests."""

from __future__ import annotations

import argparse
import importlib.util as iu
import logging
import os
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = build_pytest_cmd(args)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("run_pytest")
    # AI-AGENT-REF: echo exact command for smoke test assertions
    logger.info("[run_pytest] %s", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

