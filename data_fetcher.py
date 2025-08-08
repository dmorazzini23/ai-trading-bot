import warnings as _w
from ai_trading.data_fetcher import *  # re-export
from ai_trading import data_fetcher as _pkg
_w.warn(
    "Importing 'data_fetcher' from repo root is deprecated; use 'from ai_trading.data_fetcher import ...'.",
    DeprecationWarning,
    stacklevel=2,
)
try:
    __all__ = _pkg.__all__  # type: ignore[attr-defined]
except Exception:
    pass