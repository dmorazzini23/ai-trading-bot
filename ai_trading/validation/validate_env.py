from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator

from ai_trading.logging.redact import redact, redact_env

class Settings(BaseModel):
    ALPACA_API_KEY: str = Field(...)
    ALPACA_SECRET_KEY: str = Field(...)
    ALPACA_BASE_URL: str = Field(...)
    TRADING_MODE: str = Field('testing')
    FORCE_TRADES: bool = Field(False)

    @field_validator('ALPACA_API_KEY')
    @classmethod
    def _api_key(cls, v: str) -> str:
        if len(v) < 16:
            raise ValueError('ALPACA_API_KEY appears too short')
        return v

    @field_validator('ALPACA_SECRET_KEY')
    @classmethod
    def _secret_key(cls, v: str) -> str:
        if len(v) < 16:
            raise ValueError('ALPACA_SECRET_KEY appears too short')
        return v

    @field_validator('ALPACA_BASE_URL')
    @classmethod
    def _base_url(cls, v: str) -> str:
        if not v.startswith('https://'):
            raise ValueError('ALPACA_BASE_URL must use HTTPS')
        return v

    @field_validator('TRADING_MODE')
    @classmethod
    def _trading_mode(cls, v: str) -> str:
        if v not in {'testing', 'production'}:
            raise ValueError("TRADING_MODE must be one of ['testing', 'production']")
        return v

    @field_validator('FORCE_TRADES')
    @classmethod
    def _force_trades(cls, v: bool | str) -> bool:
        if isinstance(v, str):
            v = v.lower() in {'1', 'true', 'yes', 'on'}
        return bool(v)

def debug_environment() -> dict[str, Any]:
    """Return structured snapshot of environment configuration.

    The report masks sensitive values and provides metadata for each
    environment variable to aid debugging without leaking secrets.
    """

    raw_env = dict(os.environ)
    masked_env = redact(redact_env(raw_env))

    env_vars: Dict[str, Dict[str, Any]] = {}
    for key, original_value in raw_env.items():
        masked_value = masked_env.get(key)
        env_vars[key] = {
            "status": "set" if original_value else "missing",
            "value": masked_value if original_value else None,
            "length": len(original_value) if original_value else 0,
        }

    report: Dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "validation_status": "passed",
        "critical_issues": [],
        "warnings": [],
        "environment_vars": env_vars,
        "recommendations": [],
    }
    if report["critical_issues"]:
        report["validation_status"] = "issues"
    return report


def validate_specific_env_var(name: str) -> dict[str, Any]:
    """Validate presence of a specific environment variable.

    Parameters
    ----------
    name:
        Name of the environment variable to check.
    """

    value = os.environ.get(name)
    if value is None:
        return {
            "variable": name,
            "status": "missing",
            "value": None,
            "issues": [f"{name} is not set"],
        }

    masked_value = redact({name: value})[name]
    return {
        "variable": name,
        "status": "set",
        "value": masked_value,
        "issues": [],
    }


__all__ = ["Settings", "debug_environment", "validate_specific_env_var"]

