#!/usr/bin/env python
from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys


def has_xdist() -> bool:
    try:
        import xdist  # noqa: F401

        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])

    # AI-AGENT-REF: map --disable-warnings to interpreter flag
    if "--disable-warnings" in args:
        args.remove("--disable-warnings")
        args[:0] = ["-W", "ignore"]

    base = ["pytest", "-q"]

    autoload_disabled = os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"

    # Add -n auto only when xdist is available
    if has_xdist():
        # AI-AGENT-REF: ensure xdist plugin is loaded when autoload is disabled
        if autoload_disabled:
            base.extend(["-p", "xdist.plugin"])
        base.extend(["-n", os.environ.get("PYTEST_XDIST_WORKERS", "auto")])

    cmd = base + args

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("run_pytest")
    logger.info("[run_pytest] %s", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

