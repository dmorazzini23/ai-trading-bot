#!/usr/bin/env python
"""Deterministic pytest runner used by CI smoke tests."""

from __future__ import annotations

import argparse
import importlib.util as iu
import logging
import os
import re
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
    p.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Explicit test file/dir/node-id targets (alias)",
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
        addopts = os.environ.get("PYTEST_ADDOPTS", "")
        if ("-p pytest_asyncio" not in addopts) and (iu.find_spec("pytest_asyncio") is not None):
            cmd += ["-p", "pytest_asyncio.plugin"]
        # Under autoload-off, inject xdist only if available and not already present via addopts
        no_xdist = os.environ.get("NO_XDIST") == "1"
        if ("-p xdist.plugin" not in addopts) and (iu.find_spec("xdist") is not None) and not no_xdist:
            cmd += ["-p", "xdist.plugin", "-n", os.environ.get("PYTEST_XDIST_N", "auto")]
        # Explicitly load plugins otherwise skipped by autoload
        cmd += ["-p", "pytest_timeout", "-p", "pytest_asyncio"]
    if args.collect_only:
        cmd += ["--collect-only"]

    targets: list[str] = []
    if args.files:
        targets.extend(args.files)
    if args.targets:
        targets.extend(args.targets)
    if targets:
        # AI-AGENT-REF: append explicit targets last so pytest limits collection
        cmd.extend(targets)
        if args.keyword:
            cmd += ["-k", args.keyword]
        return cmd

    if args.keyword:
        # AI-AGENT-REF: limit collection to matching test files when only -k is given
        keywords = {
            tok.strip()
            for tok in re.split(r"\bor\b|\band\b|\bnot\b|[()]", args.keyword)
            if tok.strip()
        }
        tests_dir = Path("tests")
        matches: list[str] = []
        for word in keywords:
            matches.extend(str(p) for p in tests_dir.glob(f"test*{word}*.py"))
        if matches:
            cmd.extend(sorted(set(matches)))
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

        logger = logging.getLogger("run_pytest")
        # AI-AGENT-REF: route import diagnostics through logger without polluting smoke output
        logger.debug("[run_pytest] ai_trading path -> %s", Path(ai_trading.__file__).resolve())
    except Exception as e:  # pragma: no cover - diagnostic only
        logger = logging.getLogger("run_pytest")
        logger.warning("[run_pytest] ai_trading not importable yet (%s); pytest will handle it", e)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("run_pytest")
    _ensure_repo_on_path()
    # AI-AGENT-REF: disable implicit plugin autoload for deterministic xdist startup
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    parser = build_parser()
    args = parser.parse_args(argv)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    cmd = build_pytest_cmd(args)
    # AI-AGENT-REF: echo exact command for smoke test assertions
    logger.info("[run_pytest] %s", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0 and "-n" in cmd and os.environ.get("NO_XDIST") != "1":
        logger.info("[run_pytest] xdist run failed; retrying without xdist…")
        xdist_pairs = [
            ("-p", "xdist.plugin"),
            ("-n", os.environ.get("PYTEST_XDIST_N", "auto")),
        ]
        cmd_wo: list[str] = []
        skip = False
        for i, c in enumerate(cmd):
            if skip:
                skip = False
                continue
            if any(i + 1 < len(cmd) and c == a and cmd[i + 1] == b for a, b in xdist_pairs):
                skip = True
                continue
            cmd_wo.append(c)
        logger.info("[run_pytest] %s", " ".join(cmd_wo))
        rc = subprocess.call(cmd_wo)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
