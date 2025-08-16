from __future__ import annotations

import argparse
import os
import sys


def _validate() -> int:
    # Minimal checks used by tests; expand if you want
    required = (
        "WEBHOOK_SECRET",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "ALPACA_BASE_URL",
    )
    missing = [k for k in required if not os.getenv(k)]
    return 0 if not missing else 1


def _main() -> None:
    parser = argparse.ArgumentParser("validate-env", add_help=False)
    parser.add_argument("--quiet", action="store_true")
    # Ignore pytest's argv noise:
    _, _ = parser.parse_known_args([])
    rc = _validate()
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    _main()
