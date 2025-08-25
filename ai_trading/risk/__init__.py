"""Risk Management Module - Institutional Grade Risk Controls."""
from __future__ import annotations

from ai_trading.config.management import TradingConfig
from . import kelly as _kelly
from .circuit_breakers import (
    CircuitBreakerState,
    DeadMansSwitch,
    DrawdownCircuitBreaker,
    SafetyLevel,
    TradingHaltManager,
    VolatilityCircuitBreaker,
)
from .engine import RiskEngine
from .kelly import (
    InstitutionalKelly,
    KellyCalculator,
    KellyCriterion,
    KellyParams,
    institutional_kelly,
)
from .manager import PortfolioRiskAssessor, RiskManager
from .metrics import DrawdownAnalyzer, RiskMetricsCalculator
from .position_sizing import (
    ATRPositionSizer,
    DynamicPositionSizer,
    PortfolioPositionManager,
    VolatilityPositionSizer,
)

# AI-AGENT-REF: initialize Kelly defaults without importing settings
_kelly._DEFAULT_CONFIG = TradingConfig.from_env()

__all__ = [
    "RiskEngine",
    "KellyCriterion",
    "KellyCalculator",
    "institutional_kelly",
    "InstitutionalKelly",
    "KellyParams",
    "RiskManager",
    "PortfolioRiskAssessor",
    "ATRPositionSizer",
    "VolatilityPositionSizer",
    "DynamicPositionSizer",
    "PortfolioPositionManager",
    "DrawdownCircuitBreaker",
    "VolatilityCircuitBreaker",
    "TradingHaltManager",
    "DeadMansSwitch",
    "CircuitBreakerState",
    "SafetyLevel",
    "RiskMetricsCalculator",
    "DrawdownAnalyzer",
]
