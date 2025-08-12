"""
Runtime context for trading bot with standardized parameters.

This module provides a standardized runtime context that ensures consistent
access to trading parameters and configuration across the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_trading.config.management import TradingConfig


@dataclass
class BotRuntime:
    """
    Standardized runtime context for the trading bot.
    
    Provides consistent access to configuration and runtime parameters
    required by the trading loop and related components.
    """
    cfg: "TradingConfig"
    params: dict[str, Any] = field(default_factory=dict)
    
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


def build_runtime(cfg: "TradingConfig") -> BotRuntime:
    """
    Build a runtime context from trading configuration.
    
    Args:
        cfg: Trading configuration object
        
    Returns:
        BotRuntime with populated params dict
    """
    params = {
        "CAPITAL_CAP": float(getattr(cfg, "capital_cap", 0.04)),
        "DOLLAR_RISK_LIMIT": float(getattr(cfg, "dollar_risk_limit", 0.05)),
        "MAX_POSITION_SIZE": float(getattr(cfg, "max_position_size", 1)),
        "KELLY_FRACTION": float(getattr(cfg, "kelly_fraction", 0.6)),
        "BUY_THRESHOLD": float(getattr(cfg, "buy_threshold", 0.2)),
        "CONF_THRESHOLD": float(getattr(cfg, "conf_threshold", 0.75)),
    }
    
    return BotRuntime(cfg=cfg, params=params)


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
    
    return runtime