"""
AI Trading Bot Module - Institutional Grade Trading Platform

This module contains the core trading bot functionality and institutional-grade
components for professional trading operations including:

- Core trading enums and constants
- Database models and connection management
- Kelly Criterion risk management
- Strategy framework and execution engine
- Performance monitoring and alerting
- Institutional-grade order management

The platform is designed for institutional-scale operations with proper
risk controls, monitoring, and compliance capabilities.
"""

__version__ = "2.0.0"

# Import-light init - only expose version and basic metadata
__all__ = ["__version__"]

try:  # defensive convenience
    from . import strategy_allocator  # noqa: F401
except Exception:
    pass
