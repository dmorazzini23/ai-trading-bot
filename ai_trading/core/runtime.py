"""
Runtime context for trading bot with standardized parameters.

This module provides a standardized runtime context that ensures consistent
access to trading parameters and configuration across the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List
from pathlib import Path  # AI-AGENT-REF: path helper for future extensions

if TYPE_CHECKING:
    from ai_trading.config.management import TradingConfig


# Required parameter defaults as specified in the problem statement
REQUIRED_PARAM_DEFAULTS = {
    "CAPITAL_CAP": 0.04,          # fraction of equity cap per position
    "DOLLAR_RISK_LIMIT": 0.05,    # fraction or absolute depending on your semantics
    "MAX_POSITION_SIZE": 1.0,     # multiplier / lots
    "KELLY_FRACTION": 0.6,        # Kelly criterion fraction
    "BUY_THRESHOLD": 0.2,         # Buy signal threshold
    "CONF_THRESHOLD": 0.75,       # Confidence threshold
}


def _cfg_coalesce(cfg, key, default):
    """
    Support both UPPER_SNAKE in params and lower_snake on cfg.
    Prefer explicit cfg attributes if present.
    """
    lower = key.lower()
    # prefer explicit cfg attributes if present
    if hasattr(cfg, lower):
        return getattr(cfg, lower)
    if hasattr(cfg, key):
        return getattr(cfg, key)
    return default


class NullAlphaModel:
    """Fallback alpha model used when no model is configured."""

    def predict(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - simple no-op
        return None


@dataclass
class BotRuntime:
    """
    Standardized runtime context for the trading bot.
    
    Provides consistent access to configuration and runtime parameters
    required by the trading loop and related components.
    """
    cfg: "TradingConfig"
    params: dict[str, Any] = field(default_factory=dict)
    tickers: List[str] = field(default_factory=list)  # AI-AGENT-REF: runtime-selected tickers
    
    # Additional runtime attributes will be set by _ensure_initialized
    # These are forwarded from the underlying LazyBotContext
    api: Any = None
    data_client: Any = None
    data_fetcher: Any = None
    signal_manager: Any = None
    risk_engine: Any = None
    capital_scaler: Any = None
    execution_engine: Any = None
    drawdown_circuit_breaker: Any = None
    model: Any = None


def build_runtime(cfg: "TradingConfig") -> BotRuntime:
    """
    Build a runtime context from trading configuration.
    
    Ensures all required parameters are populated from TradingConfig with 
    explicit defaults as specified in the problem statement.
    
    Args:
        cfg: Trading configuration object
        
    Returns:
        BotRuntime with fully populated params dict
    """
    params = {}
    for k, dflt in REQUIRED_PARAM_DEFAULTS.items():
        params[k] = float(_cfg_coalesce(cfg, k, dflt))
    
    # Add any additional keys the loop expects if needed
    runtime = BotRuntime(cfg=cfg, params=params)
    runtime.model = NullAlphaModel()
    return runtime


def enhance_runtime_with_context(runtime: BotRuntime, lazy_context: Any) -> BotRuntime:
    """
    Enhance runtime with attributes from LazyBotContext after initialization.
    
    Args:
        runtime: BotRuntime to enhance
        lazy_context: Initialized LazyBotContext
        
    Returns:
        Enhanced runtime with context attributes
    """
    # Forward key attributes from the lazy context
    runtime.api = getattr(lazy_context, 'api', None)
    runtime.data_client = getattr(lazy_context, 'data_client', None) 
    runtime.data_fetcher = getattr(lazy_context, 'data_fetcher', None)
    runtime.signal_manager = getattr(lazy_context, 'signal_manager', None)
    runtime.risk_engine = getattr(lazy_context, 'risk_engine', None)
    runtime.capital_scaler = getattr(lazy_context, 'capital_scaler', None)
    runtime.execution_engine = getattr(lazy_context, 'execution_engine', None)
    runtime.drawdown_circuit_breaker = getattr(lazy_context, 'drawdown_circuit_breaker', None)
    runtime.model = getattr(
        lazy_context,
        'model',
        getattr(lazy_context, 'alpha_model', runtime.model),
    )
    return runtime

