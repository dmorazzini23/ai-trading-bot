import warnings as _w
from ai_trading.signals import *  # re-export
from ai_trading import signals as _pkg

_w.warn(
    "Importing 'signals' from repo root is deprecated; use 'from ai_trading import signals' "
    "or 'from ai_trading.signals import ...'. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Mirror __all__ if present for clean 'from signals import *'
try:
    __all__ = _pkg.__all__  # type: ignore[attr-defined]
except Exception:
    pass