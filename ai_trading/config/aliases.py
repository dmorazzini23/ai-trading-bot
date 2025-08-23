from __future__ import annotations
import os
from ai_trading.logging import get_logger, logger_once
_log = get_logger(__name__)
_ALIASES = ['TRADING_MODE', 'BOT_MODE', 'bot_mode']
_CANON = 'TRADING_MODE'

def resolve_trading_mode(default: str) -> str:
    """Resolve trading mode across aliases with precedence and deprecation logs.

    Precedence: TRADING_MODE > BOT_MODE > bot_mode > default.
    If conflicting values are present, prefer TRADING_MODE and emit a once log.
    """
    values = {k: os.getenv(k) for k in _ALIASES}
    chosen: tuple[str, str] | None = None
    for key in (_CANON, 'BOT_MODE', 'bot_mode'):
        v = values.get(key)
        if v:
            chosen = (key, v)
            break
    if not chosen:
        return default
    key, val = chosen
    if key != _CANON:
        logger_once.warning('DEPRECATED_CONFIG_ALIAS', key=f'deprec:{key}', extra={'alias': key, 'use': _CANON, 'value': val})
    canon_val = values.get(_CANON)
    if canon_val and canon_val != val and (key != _CANON):
        logger_once.error('CONFIG_CONFLICT', key='conflict:TRADING_MODE', extra={'TRADING_MODE': canon_val, key: val})
        return canon_val
    return val