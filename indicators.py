import warnings as _w
from ai_trading.indicators import *  # re-export
from ai_trading import indicators as _pkg
_w.warn(
    "Importing 'indicators' from repo root is deprecated; use 'from ai_trading.indicators import ...'.",
    DeprecationWarning,
    stacklevel=2,
)
try:
    __all__ = _pkg.__all__  # type: ignore[attr-defined]
except Exception:
    pass