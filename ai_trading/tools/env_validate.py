
from __future__ import annotations

import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

# Test suite expects ALPACA_BASE_URL; keep CLI simple and consistent
REQUIRED = ("ALPACA_BASE_URL",)


def validate_env(env: dict[str, str] | None = None) -> list[str]:
    """Return list of missing required environment keys."""  # AI-AGENT-REF
    env = env or os.environ
    return [k for k in REQUIRED if not env.get(k)]


def _main(argv: list[str] | None = None) -> int:
    """CLI entry point returning process exit code with JSON on stdout.

    Contract: print {"ok": bool, "missing": [...]} and exit 0/1.
    """  # AI-AGENT-REF
    _ = argv or sys.argv[1:]
    missing = validate_env()
    ok = not bool(missing)
    try:
        print(json.dumps({"ok": ok, "missing": missing}))
    except Exception:  # noqa: BLE001
        print('{"ok": false, "missing": []}')
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["validate_env", "_main"]
