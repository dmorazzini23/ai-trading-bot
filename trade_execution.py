import warnings as _w
from ai_trading.trade_execution import *  # re-export
from ai_trading import trade_execution as _pkg
_w.warn(
    "Importing 'trade_execution' from repo root is deprecated; use 'from ai_trading.trade_execution import ...'.",
    DeprecationWarning,
    stacklevel=2,
)
try:
    __all__ = _pkg.__all__  # type: ignore[attr-defined]
except Exception:
    pass