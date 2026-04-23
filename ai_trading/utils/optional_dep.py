from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=None)
def missing(pkg: str, feature: str) -> bool:
    try:
        __import__(pkg)
        return False
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        logger.warning("Optional feature '%s' disabled: missing dependency '%s'", feature, pkg)
        return True
