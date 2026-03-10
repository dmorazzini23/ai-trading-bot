from __future__ import annotations

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger, logger_once

_log = get_logger(__name__)
_CANON = "AI_TRADING_TRADING_MODE"
_ALIASES = [_CANON, "TRADING_MODE", "bot_mode"]

def resolve_trading_mode(default: str, *, skip_env: bool = False) -> str:
    """Resolve trading mode across aliases with precedence and deprecation logs.

    Precedence: AI_TRADING_TRADING_MODE > TRADING_MODE > bot_mode > default.
    If conflicting values are present, prefer AI_TRADING_TRADING_MODE and emit a once log.
    """
    if skip_env:
        return default
    values = {
        k: get_env(k, None, cast=str, resolve_aliases=False)
        for k in _ALIASES
    }
    chosen: tuple[str, str] | None = None
    for key in _ALIASES:
        v = values.get(key)
        if v:
            chosen = (key, v)
            break
    if not chosen:
        return default
    key, val = chosen
    if key != _CANON:
        logger_once.warning(
            "DEPRECATED_CONFIG_ALIAS",
            key=f"deprec:{key}",
            extra={"alias": key, "use": _CANON, "value": val},
        )
    canon_val = values.get(_CANON)
    if canon_val and canon_val != val and (key != _CANON):
        logger_once.error(
            "CONFIG_CONFLICT",
            key=f"conflict:{_CANON}",
            extra={_CANON: canon_val, key: val},
        )
        return canon_val
    return val
