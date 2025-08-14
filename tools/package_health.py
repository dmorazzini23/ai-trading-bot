from __future__ import annotations

import argparse
import importlib
import logging
import os
import pathlib
import pkgutil
import sys

# AI-AGENT-REF: repo package health validation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = "ai_trading"
CRITICAL_EXPORTS = [
    ("ai_trading.rl_trading", "RLTrader"),
    ("ai_trading.core.bot_engine", "run_all_trades_worker"),
]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# AI-AGENT-REF: requests health probe
def _probe_requests() -> bool:
    try:
        import requests
        from requests import Response  # noqa: F401
        ok = hasattr(requests, "Session") and hasattr(requests, "get")
        print("[health] requests:", "ok" if ok else "broken (missing API)")
        return ok
    except Exception as e:  # pragma: no cover - surface import issue
        print("[health] requests: import failed ->", e)
        return False


# AI-AGENT-REF: tzlocal health probe
def _probe_tzlocal() -> bool:
    try:
        import tzlocal  # noqa: F401
        return True
    except Exception:
        return False


def _run_probes() -> None:
    _probe_requests()
    print("[health] tzlocal:", "ok" if _probe_tzlocal() else "missing (optional)")


def find_dirs_missing_init(pkg_dir: pathlib.Path) -> list[str]:
    missing: list[str] = []
    for dirpath, dirnames, filenames in os.walk(pkg_dir):
        if "__pycache__" in dirpath:
            continue
        if any(fn.endswith(".py") for fn in filenames) and "__init__.py" not in filenames:
            missing.append(dirpath)
    return missing


def import_all(prefix: str) -> tuple[str, str] | None:
    first_error: tuple[str, str] | None = None
    pkg = importlib.import_module(prefix)
    for modinfo in pkgutil.walk_packages(pkg.__path__, prefix + "."):
        name = modinfo.name
        try:
            importlib.import_module(name)
        except Exception as e:  # pragma: no cover - surface first import error
            first_error = (name, repr(e))
            break
    return first_error


def check_exports() -> list[str]:
    problems: list[str] = []
    for mod, attr in CRITICAL_EXPORTS:
        try:
            m = importlib.import_module(mod)
            if not hasattr(m, attr):
                problems.append(f"{mod} missing expected export '{attr}'")
        except Exception as e:  # pragma: no cover - surface import issue
            problems.append(f"Failed to import {mod}: {e!r}")
    return problems


def main() -> int:
    pkg_dir = ROOT / PKG
    missing_init = find_dirs_missing_init(pkg_dir)
    if missing_init:
        logger.error("MISSING __init__.py in directories:")
        for d in missing_init[:25]:
            logger.error(" - %s", d)
        if len(missing_init) > 25:
            logger.error("... and %d more", len(missing_init) - 25)
        return 2

    failed = import_all(PKG)
    export_issues = check_exports()
    if failed:
        name, err = failed
        logger.error("IMPORT FAILURE: %s: %s", name, err)
        return 3
    if export_issues:
        for p in export_issues:
            logger.error("EXPORT ISSUE: %s", p)
        return 4

    logger.info("PACKAGE HEALTH: OK")
    return 0


if __name__ == "__main__":
    _run_probes()
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.parse_args()
    sys.exit(main())
