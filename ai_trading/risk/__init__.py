"""Risk Management Module - Institutional Grade Risk Controls."""
from __future__ import annotations

from ai_trading.config.management import TradingConfig, from_env_relaxed
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


def _load_default_kelly_config() -> TradingConfig:
    """Build the Kelly default config lazily with relaxed fallback."""

    try:
        return TradingConfig.from_env()
    except RuntimeError as exc:
        message = str(exc)
        if "MAX_DRAWDOWN_THRESHOLD" not in message:
            raise
        return from_env_relaxed()


_kelly.configure_default_config(_load_default_kelly_config)

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
