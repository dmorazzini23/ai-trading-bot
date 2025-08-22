"""
Institutional-grade testing framework for live trading bot.

This package provides comprehensive testing capabilities for validating
live trading functionality, risk management, and compliance requirements.
"""

from .framework import (
    ComplianceTestSuite,
    MockMarketDataProvider,
    TradingScenarioRunner,
)

__all__ = [
    "MockMarketDataProvider",
    "TradingScenarioRunner",
    "ComplianceTestSuite"
]
