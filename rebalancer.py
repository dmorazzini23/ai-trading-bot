import warnings as _w
from ai_trading.rebalancer import *  # re-export
from ai_trading import rebalancer as _pkg
_w.warn(
    "Importing 'rebalancer' from repo root is deprecated; use 'from ai_trading.rebalancer import ...'.",
    DeprecationWarning,
    stacklevel=2,
)
try:
    __all__ = _pkg.__all__  # type: ignore[attr-defined]
except Exception:
    pass