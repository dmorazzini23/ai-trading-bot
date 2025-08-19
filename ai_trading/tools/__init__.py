"""Utility CLI tools for ai_trading."""

from . import env_validate as _env_validate  # AI-AGENT-REF: load validator
import sys as _sys

_sys.modules[__name__ + ".validate_env"] = _env_validate
validate_env = _env_validate

__all__ = ["validate_env"]


