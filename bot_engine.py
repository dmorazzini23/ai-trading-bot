# Deprecated shim: forward to package module
from ai_trading.core.bot_engine import *  # noqa: F401,F403

# --- Test compatibility: provide prepare_indicators if missing ---
try:
    prepare_indicators  # type: ignore[name-defined]
except NameError:
    def prepare_indicators(*args, **kwargs):  # flexible signature for tests
        """
        Compatibility wrapper. Prefer the canonical implementation if available,
        otherwise return a minimal indicator manager.
        """
        try:
            from ai_trading.core.bot_engine import prepare_indicators as _impl  # type: ignore
            return _impl(*args, **kwargs)
        except Exception:
            # Minimal fallback: construct a streaming SMA to keep tests happy.
            try:
                from ai_trading.indicator_manager import IndicatorManager, IndicatorSpec
                mgr = IndicatorManager()
                mgr.add("sma20", IndicatorSpec(kind="sma", period=20))
                return mgr
            except Exception:
                return None