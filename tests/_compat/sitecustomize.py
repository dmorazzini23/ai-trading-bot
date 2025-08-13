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

# AI-AGENT-REF: register slippage shim for tests when real package missing
try:
    import slippage  # noqa: F401
except Exception:
    shim_path = os.path.dirname(__file__)
    if shim_path not in sys.path:
        sys.path.insert(0, shim_path)
    import importlib.util
    spec = importlib.util.spec_from_file_location("slippage", os.path.join(shim_path, "slippage.py"))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[assignment]
    sys.modules["slippage"] = module
