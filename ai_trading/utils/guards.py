from __future__ import annotations
import contextlib
from collections.abc import Iterable
from ai_trading.logging import get_logger
_log = get_logger(__name__)

class GuardError(AssertionError):
    pass

@contextlib.contextmanager
def catch(exc_types: Iterable[type[BaseException]], *, label: str, re_raise: bool=False):
    """Context manager enforcing explicit exception types and structured logging.

    Usage:
        with catch((ValueError, KeyError), label="parse-config"):
            ... # code that may raise ValueError/KeyError
    """
    exc_tuple: tuple[type[BaseException], ...] = tuple(exc_types)
    if not exc_tuple:
        raise GuardError('guards.catch() requires at least one exception type')
    if any((t is Exception or t is BaseException for t in exc_tuple)):
        raise GuardError('guards.catch() disallows Exception/BaseException; be specific')
    try:
        yield
    except exc_tuple as e:
        _log.warning('GUARD_CAUGHT', extra={'label': label, 'exc': type(e).__name__, 'msg': str(e)})
        if re_raise:
            raise