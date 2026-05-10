"""Shared guard for archival demo and one-off validation scripts."""

from __future__ import annotations

import os
import sys

LEGACY_DEMO_ENV = "AI_TRADING_ENABLE_LEGACY_DEMO"
LEGACY_DEMO_ARG = "--allow-legacy-demo"


def require_legacy_demo_flag(script_name: str) -> None:
    """Require an explicit operator acknowledgement before dummy env seeding."""

    if LEGACY_DEMO_ARG in sys.argv:
        sys.argv.remove(LEGACY_DEMO_ARG)
        return
    enabled = os.environ.get(LEGACY_DEMO_ENV, "").strip().lower()
    if enabled in {"1", "true", "yes", "on"}:
        return
    sys.stderr.write(
        f"{script_name} is an archival legacy/demo validation script that seeds "
        "dummy credentials or deprecated env names. Re-run with "
        f"{LEGACY_DEMO_ARG} or set {LEGACY_DEMO_ENV}=1.\n"
    )
    raise SystemExit(2)
