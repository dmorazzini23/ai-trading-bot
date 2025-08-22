"""
Institutional-grade testing framework for live trading bot.

This package provides comprehensive testing capabilities for validating
live trading functionality, risk management, and compliance requirements.
"""

from .framework import (
    ComplianceTestSuite as ComplianceTestSuite,
)
from .framework import (
    MockMarketDataProvider as MockMarketDataProvider,
)
from .framework import (
    TradingScenarioRunner as TradingScenarioRunner,
)

__all__ = [
    "MockMarketDataProvider",
    "TradingScenarioRunner",
    "ComplianceTestSuite"
]
