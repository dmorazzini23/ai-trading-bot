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
import sys as _sys

# AI-AGENT-REF: expose validate_env module at top-level for tests
try:
    from .tools import validate_env as _validate_env_mod

    _sys.modules.setdefault("validate_env", _validate_env_mod)
except Exception:
    pass
