import warnings as _w
from ai_trading.pipeline import *  # re-export
from ai_trading import pipeline as _pkg
_w.warn(
    "Importing 'pipeline' from repo root is deprecated; use 'from ai_trading.pipeline import ...'.",
    DeprecationWarning,
    stacklevel=2,
)
try:
    __all__ = _pkg.__all__  # type: ignore[attr-defined]
except Exception:
    pass