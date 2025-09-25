"""
Runtime context for trading bot with standardized parameters.

This module provides a standardized runtime context that ensures consistent
access to trading parameters and configuration across the system.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from .protocols import AllocatorProtocol
from ai_trading.position_sizing import (
    resolve_max_position_size,
    _get_equity_from_alpaca,
)
from ai_trading.logging import get_logger
logger = get_logger(__name__)
if TYPE_CHECKING:
    from ai_trading.config.management import TradingConfig
REQUIRED_PARAM_DEFAULTS = {
    'CAPITAL_CAP': 0.25,
    'DOLLAR_RISK_LIMIT': 0.05,
    'MAX_POSITION_SIZE': 8000.0,
    'KELLY_FRACTION': 0.6,
    'BUY_THRESHOLD': 0.2,
    'CONF_THRESHOLD': 0.75,
}

def _cfg_coalesce(cfg, key, default):
    """
    Support both UPPER_SNAKE in params and lower_snake on cfg.
    Prefer explicit cfg attributes if present.
    """
    lower = key.lower()
    if hasattr(cfg, lower):
        val = getattr(cfg, lower)
        if val is not None:
            return val
    if hasattr(cfg, key):
        val = getattr(cfg, key)
        if val is not None:
            return val
    return default

class NullAlphaModel:
    """Fallback alpha model used when no model is configured."""

    def predict(self, *args: Any, **kwargs: Any) -> None:
        return None

@dataclass
class BotRuntime:
    """
    Standardized runtime context for the trading bot.
    
    Provides consistent access to configuration and runtime parameters
    required by the trading loop and related components.
    """
    cfg: TradingConfig
    params: dict[str, Any] = field(default_factory=dict)
    tickers: list[str] = field(default_factory=list)
    api: Any = None
    data_client: Any = None
    data_fetcher: Any = None
    signal_manager: Any = None
    risk_engine: Any = None
    capital_scaler: Any = None
    execution_engine: Any = None
    drawdown_circuit_breaker: Any = None
    model: Any = None
    allocator: AllocatorProtocol | None = None
    state: dict[str, Any] = field(default_factory=dict)

def build_runtime(cfg: TradingConfig, **kwargs: Any) -> BotRuntime:
    """
    Build a runtime context from trading configuration.
    
    Ensures all required parameters are populated from TradingConfig with 
    explicit defaults as specified in the problem statement.
    
    Args:
        cfg: Trading configuration object
        
    Returns:
        BotRuntime with fully populated params dict
    """
    impl_raw = os.getenv("AI_TRADING_EXECUTION_IMPL", os.getenv("EXECUTION_IMPL", ""))
    impl = (impl_raw or "").lower()
    if impl in {"live", "broker", "alpaca"}:
        missing = [
            key
            for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL")
            if not os.getenv(key)
        ]
        if missing:
            logger.critical(
                "LIVE_REQUESTED_BUT_CREDS_MISSING",
                extra={"missing": missing},
            )
            raise RuntimeError(
                f"Live trading requested but missing credentials: {', '.join(missing)}"
            )

    params: dict[str, float] = {}
    for k, dflt in REQUIRED_PARAM_DEFAULTS.items():
        # MAX_POSITION_SIZE is resolved after capital cap so the value can be
        # derived from it if not explicitly provided.
        if k == "MAX_POSITION_SIZE":
            continue
        params[k] = float(_cfg_coalesce(cfg, k, dflt))

    if getattr(cfg, "capital_cap", None) is None:
        try:
            object.__setattr__(cfg, "capital_cap", params["CAPITAL_CAP"])
        except Exception:
            pass

    # Ensure equity is populated so downstream sizing uses a cached value.
    eq = getattr(cfg, "equity", None)
    if eq in (None, 0.0):
        fetched = _get_equity_from_alpaca(cfg, force_refresh=True)
        eq = fetched if (fetched is not None and fetched > 0) else None
        try:
            object.__setattr__(cfg, "equity", eq)
        except Exception:
            try:
                setattr(cfg, "equity", eq)
            except Exception:  # pragma: no cover - defensive
                pass

    mode = str(getattr(cfg, "max_position_mode", "STATIC")).upper()
    raw_cfg_value = getattr(cfg, "max_position_size", object())
    explicit_none = raw_cfg_value is None
    val = _cfg_coalesce(cfg, "MAX_POSITION_SIZE", None)
    sizing_meta: dict[str, Any] = {}
    if mode == "AUTO":
        resolved, sizing_meta = resolve_max_position_size(cfg, cfg, force_refresh=True)
    else:
        if val is None and not explicit_none:
            try:
                from ai_trading.config.management import get_env

                env_val = get_env("MAX_POSITION_SIZE", cast=float)
            except (ImportError, RuntimeError):
                env_val = None
            if env_val is not None:
                val = env_val

        if val is None and explicit_none:
            resolved = float(REQUIRED_PARAM_DEFAULTS["MAX_POSITION_SIZE"])
            sizing_meta = {
                "mode": mode,
                "source": "required_default",
                "capital_cap": getattr(cfg, "capital_cap", 0.0),
            }
        elif val is None:
            resolved, sizing_meta = resolve_max_position_size(cfg, cfg, force_refresh=True)
        else:
            resolved = float(val)
            sizing_meta = {
                "mode": mode,
                "source": "provided",
                "capital_cap": getattr(cfg, "capital_cap", 0.0),
            }

    if not sizing_meta:
        sizing_meta = {"mode": mode, "capital_cap": getattr(cfg, "capital_cap", 0.0)}

    storage = getattr(cfg, "_values", None)
    if isinstance(storage, dict):
        storage["max_position_size"] = float(resolved)
        if sizing_meta:
            storage["max_position_size_meta"] = dict(sizing_meta)
    if getattr(cfg, "max_position_size", None) != resolved:
        try:
            object.__setattr__(cfg, "max_position_size", float(resolved))
        except Exception:
            try:
                setattr(cfg, "max_position_size", float(resolved))
            except Exception:
                pass
    if resolved <= 0:
        raise ValueError("MAX_POSITION_SIZE must be positive")
    params["MAX_POSITION_SIZE"] = float(resolved)
    try:
        import ai_trading.core.bot_engine as be

        if getattr(be, "MAX_POSITION_SIZE", None) != float(resolved):
            be.MAX_POSITION_SIZE = float(resolved)
    except Exception:
        pass
    if sizing_meta.get("source") == "fallback":
        logger.warning("POSITION_SIZING_FALLBACK", extra={**sizing_meta, "resolved": resolved})
    else:
        logger.info("POSITION_SIZING_RESOLVED", extra={**sizing_meta, "resolved": resolved})
    runtime = BotRuntime(cfg=cfg, params=params, allocator=kwargs.get('allocator'))
    runtime.model = NullAlphaModel()
    return runtime

def enhance_runtime_with_context(runtime: BotRuntime, lazy_context: Any, **kwargs: Any) -> BotRuntime:
    """
    Enhance runtime with attributes from LazyBotContext after initialization.
    
    Args:
        runtime: BotRuntime to enhance
        lazy_context: Initialized LazyBotContext
        
    Returns:
        Enhanced runtime with context attributes
    """
    runtime.api = getattr(lazy_context, 'api', None)
    runtime.data_client = getattr(lazy_context, 'data_client', None)
    runtime.data_fetcher = getattr(lazy_context, 'data_fetcher', None)
    runtime.signal_manager = getattr(lazy_context, 'signal_manager', None)
    runtime.risk_engine = getattr(lazy_context, 'risk_engine', None)
    runtime.capital_scaler = getattr(lazy_context, 'capital_scaler', None)
    runtime.execution_engine = getattr(lazy_context, 'execution_engine', None)
    runtime.drawdown_circuit_breaker = getattr(lazy_context, 'drawdown_circuit_breaker', None)
    runtime.model = getattr(lazy_context, 'model', getattr(lazy_context, 'alpha_model', runtime.model))
    if 'allocator' in kwargs:
        runtime.allocator = kwargs['allocator']
    return runtime
