"""Test bootstrap helpers for CI compatibility."""

import os
import sys

try:
    import numpy as _np  # noqa: WPS433 - test-only import aliasing
    if not hasattr(_np, "NaN"):
        # Provide the legacy alias expected by some transitive deps
        _np.NaN = _np.nan  # type: ignore[attr-defined]
except Exception as _e:  # very local/test-only guard
    # Do not make tests crash because of the shim itself
    print(f"[tests/_compat/sitecustomize] Shim init failed: {_e}")

# AI-AGENT-REF: provide BaseSettings for pydantic v2
try:
    import pydantic as _pd
    from pydantic_settings import BaseSettings as _BS

    _pd.BaseSettings = _BS  # type: ignore[attr-defined]
except Exception:
    pass

# AI-AGENT-REF: stub yfinance when unavailable or disabled
if os.getenv("YFINANCE_STUB", "1") == "1":
    from tests._compat import yfinance_stub as _stub
    sys.modules["yfinance"] = _stub
else:
    try:
        import yfinance  # noqa: F401
    except Exception:
        from tests._compat import yfinance_stub as _stub
        sys.modules.setdefault("yfinance", _stub)
