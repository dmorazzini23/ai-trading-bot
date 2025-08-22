# AI-AGENT-REF: Compatibility shim - module moved to ai_trading.execution.transaction_costs
"""
Transaction Cost Calculator Compatibility Shim

This module provides backward compatibility for code that still imports
from scripts.transaction_cost_calculator. The actual implementation has been
moved to ai_trading.execution.transaction_costs.
"""

from ai_trading.execution.transaction_costs import (
    LiquidityTier,
    ProfitabilityAnalysis,
    TradeType,
    TransactionCostBreakdown,
    TransactionCostCalculator,
    create_transaction_cost_calculator,
)

# Re-export all classes for backward compatibility
__all__ = [
    "TradeType",
    "LiquidityTier",
    "TransactionCostBreakdown",
    "ProfitabilityAnalysis",
    "TransactionCostCalculator",
    "create_transaction_cost_calculator"
]
