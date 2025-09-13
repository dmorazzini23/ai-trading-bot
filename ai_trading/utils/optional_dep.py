from __future__ import annotations
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=None)
def missing(pkg: str, feature: str) -> bool:
    try:
        __import__(pkg)
        return False
    except Exception:
        logger.warning("Optional feature '%s' disabled: missing dependency '%s'", feature, pkg)
        return True
