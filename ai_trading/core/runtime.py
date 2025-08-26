"""
Runtime context for trading bot with standardized parameters.

This module provides a standardized runtime context that ensures consistent
access to trading parameters and configuration across the system.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from .protocols import AllocatorProtocol
if TYPE_CHECKING:
    from ai_trading.config.management import TradingConfig
REQUIRED_PARAM_DEFAULTS = {'CAPITAL_CAP': 0.04, 'DOLLAR_RISK_LIMIT': 0.05, 'MAX_POSITION_SIZE': 1.0, 'KELLY_FRACTION': 0.6, 'BUY_THRESHOLD': 0.2, 'CONF_THRESHOLD': 0.75}

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

    # Resolve max_position_size with precedence:
    #   1. explicit cfg attribute
    #   2. environment variable MAX_POSITION_SIZE
    #   3. derived from capital_cap via position_sizing helper
    #   4. fallback to default
    val = _cfg_coalesce(cfg, "MAX_POSITION_SIZE", None)
    if val is None:
        try:
            from ai_trading.config.management import get_env

            env_val = get_env("MAX_POSITION_SIZE", cast=float)
        except Exception:
            env_val = None
        if env_val is not None:
            val = env_val

    if val is None:
        try:
            from ai_trading.config.management import get_env

            env_override = get_env("AI_TRADING_MAX_POSITION_SIZE", cast=float)
        except Exception:
            env_override = None
        if env_override is not None:
            val = env_override

    if val is None:
        cap = params.get("CAPITAL_CAP", REQUIRED_PARAM_DEFAULTS["CAPITAL_CAP"])
        equity = getattr(cfg, "equity", None)
        basis = equity if equity and equity > 0 else 200000.0
        val = float(round(cap * basis, 2))

    if val is None or float(val) <= 0:
        raise ValueError("MAX_POSITION_SIZE must be positive")

    params["MAX_POSITION_SIZE"] = float(val)
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
