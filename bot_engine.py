# Deprecated shim: forward to package module
import warnings
warnings.warn(
    "Importing from root bot_engine.py is deprecated. Use 'from ai_trading.core import bot_engine' instead.",
    DeprecationWarning,
    stacklevel=2
)

from ai_trading.core.bot_engine import *  # noqa: F401,F403

# --- Test compatibility: AST-extracted function expected by tests ---
def prepare_indicators(*args, **kwargs):  
    """
    Back-compat wrapper. Delegates to the new side-effect-free shim.
    Kept at module scope so AST-based tests can find it.
    """
    # Delegate to the new shim which has proper input validation
    from ai_trading.bot_engine import prepare_indicators as shim_impl
    return shim_impl(*args, **kwargs)

# --- Test compatibility: AST-extracted functions expected by tests ---
def _load_ml_model(symbol: str):
    """Thin wrapper so tests that parse this file via AST can find the symbol."""
    from ai_trading.core.bot_engine import _load_ml_model as _impl  # type: ignore
    return _impl(symbol)

def _cleanup_ml_model_cache(max_age_minutes: int = 60):
    """Thin wrapper for test discovery; delegates to package implementation."""
    from ai_trading.core.bot_engine import _cleanup_ml_model_cache as _impl  # type: ignore
    return _impl(max_age_minutes)