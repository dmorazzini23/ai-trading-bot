from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

REQUIRED = ("ALPACA_API_BASE_URL",)


def validate_env(env: dict[str, str] | None = None) -> list[str]:
    """Return list of missing required environment keys."""  # AI-AGENT-REF
    env = env or os.environ
    return [k for k in REQUIRED if not env.get(k)]


def _main(argv: list[str] | None = None) -> int:
    """CLI entry point returning process exit code."""  # AI-AGENT-REF
    _ = argv or sys.argv[1:]
    missing = validate_env()
    if missing:
        logger.error("Missing env: %s", ",".join(missing))
        return 1
    logger.info("Environment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["validate_env", "_main"]

