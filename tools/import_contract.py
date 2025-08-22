"""Bounded import contract checker for CI.

Runs import(s) in a short-lived subprocess with a wall-clock timeout so CI
can never hang. Designed to be called from `make test-all` before pytest.

Usage:
  python tools/import_contract.py --ci --timeout 20 --modules ai_trading,trade_execution
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _run_import_in_subprocess(module: str, timeout: float) -> subprocess.CompletedProcess:
    # Inherit env; set guard to avoid heavy init if package honors it.
    env = os.environ.copy()
    env.setdefault("AI_TRADING_IMPORT_CONTRACT", "1")

    # Allow tests to simulate a hang without touching real modules

    # Build a tiny Python snippet that imports the module; optionally sleeps to simulate hang
    code = (
        "import os, time, importlib;"
        "\nif os.getenv('IMPORT_CONTRACT_SIMULATE_HANG') == '1': time.sleep(60)"
        f"\nimportlib.import_module('{module}')\nprint('IMPORTED:{module}')"
    )

    args = [sys.executable, "-X", "utf8", "-c", code]
    return subprocess.run(
        args,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--modules", default="ai_trading,trade_execution", help="Comma-separated module list to import")
    p.add_argument("--timeout", type=float, default=20.0, help="Per-module timeout in seconds")
    p.add_argument("--ci", action="store_true", help="CI mode: concise logs, non-zero exit on failures")
    args = p.parse_args(argv)

    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    overall_rc = 0

    for mod in modules:
        try:
            cp = _run_import_in_subprocess(mod, args.timeout)
        except subprocess.TimeoutExpired:
            if args.ci:
                return 124
            overall_rc = overall_rc or 124
            continue

        if cp.returncode != 0:
            # Surface stderr in CI for debugging
            if cp.stdout:
                pass
            if cp.stderr:
                pass
            if args.ci:
                return cp.returncode
            overall_rc = overall_rc or cp.returncode
        elif not args.ci:
            pass

    return overall_rc


if __name__ == "__main__":
    sys.exit(main())

