"""
Runtime context for trading bot with standardized parameters.

This module provides a standardized runtime context that ensures consistent
access to trading parameters and configuration across the system.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from .protocols import AllocatorProtocol
from ai_trading.position_sizing import get_max_position_size
if TYPE_CHECKING:
    from ai_trading.config.management import TradingConfig
REQUIRED_PARAM_DEFAULTS = {
    'CAPITAL_CAP': 0.04,
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

    # Resolve max_position_size consistently with position_sizing helper.
    val = _cfg_coalesce(cfg, "MAX_POSITION_SIZE", None)
    if val is None:
        try:
            from ai_trading.config.management import get_env

            env_val = get_env("MAX_POSITION_SIZE", cast=float)
        except (ImportError, RuntimeError):
            env_val = None
        if env_val is not None:
            val = env_val
    if val is not None:
        # TradingConfig is frozen; mutate via object.__setattr__
        try:
            object.__setattr__(cfg, "max_position_size", float(val))
        except Exception:
            pass
    resolved = get_max_position_size(cfg, cfg, force_refresh=True)
    if getattr(cfg, "max_position_size", None) != resolved:
        try:
            object.__setattr__(cfg, "max_position_size", float(resolved))
        except Exception:
            pass
    if resolved <= 0:
        raise ValueError("MAX_POSITION_SIZE must be positive")
    params["MAX_POSITION_SIZE"] = float(resolved)
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
