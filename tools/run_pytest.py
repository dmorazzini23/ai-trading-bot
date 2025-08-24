#!/usr/bin/env python
from __future__ import annotations

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


def main() -> int:
    args = sys.argv[1:] or []
    base = ["pytest", "-q"]
    # Respect plugin autoload disabling if the caller set it
    if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1":
        pass
    # Add -n auto only when xdist is available
    if has_xdist():
        base.extend(["-n", "auto"])
    cmd = base + args
    print("[run_pytest]", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

