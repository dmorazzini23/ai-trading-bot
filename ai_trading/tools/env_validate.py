from __future__ import annotations
import os
import sys
import importlib.util
from collections.abc import Iterable, Mapping
from ai_trading.logging import get_logger
logger = get_logger(__name__)
REQUIRED_KEYS: tuple[str, ...] = (
    'ALPACA_API_KEY',
    'ALPACA_SECRET_KEY',
    'ALPACA_BASE_URL',
)
REQUIRED_PACKAGES: tuple[str, ...] = ('hmmlearn',)

def validate_env(env: Mapping[str, str] | None=None) -> list[str]:
    """Return list of missing required environment keys or packages."""
    env = env or os.environ
    missing = [k for k in REQUIRED_KEYS if not env.get(k)]
    for pkg in REQUIRED_PACKAGES:
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    return missing

def main(argv: Iterable[str] | None=None) -> int:
    """CLI entry point returning process exit code."""
    _ = list(argv or sys.argv[1:])
    missing = validate_env()
    if not missing:
        logger.info('ENV_VALIDATE_OK', extra={'missing': 0})
        return 0
    logger.error('ENV_VALIDATE_MISSING', extra={'missing_keys': missing, 'count': len(missing)})
    return 1

def _main(argv: list[str] | None=None) -> int:
    return main(argv)
if __name__ == '__main__':
    raise SystemExit(main())
__all__ = ['validate_env', 'main', '_main']
