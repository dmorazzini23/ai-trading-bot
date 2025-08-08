import warnings as _w
from ai_trading.portfolio import *  # re-export
from ai_trading import portfolio as _pkg
_w.warn(
    "Importing 'portfolio' from repo root is deprecated; use 'from ai_trading.portfolio import ...'.",
    DeprecationWarning,
    stacklevel=2,
)
try:
    __all__ = _pkg.__all__  # type: ignore[attr-defined]
except Exception:
    pass