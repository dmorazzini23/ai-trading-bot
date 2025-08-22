
from __future__ import annotations

import os
import sys
from typing import Iterable, Mapping

from ai_trading.logging import get_logger

logger = get_logger(__name__)

REQUIRED_KEYS: tuple[str, ...] = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_BASE_URL",
)


def validate_env(env: Mapping[str, str] | None = None) -> list[str]:
    """Return list of missing required environment keys."""  # AI-AGENT-REF
    env = env or os.environ
    return [k for k in REQUIRED_KEYS if not env.get(k)]


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point returning process exit code."""  # AI-AGENT-REF
    _ = list(argv or sys.argv[1:])
    missing = validate_env()
    if not missing:
        logger.info("ENV_VALIDATE_OK", extra={"missing": 0})
        return 0
    logger.error(
        "ENV_VALIDATE_MISSING", extra={"missing_keys": missing, "count": len(missing)}
    )
    return 1


# Back-compat entrypoint
def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin wrapper
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["validate_env", "main", "_main"]
