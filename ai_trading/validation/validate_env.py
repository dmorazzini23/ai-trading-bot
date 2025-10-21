from __future__ import annotations

import os
from datetime import datetime

from pydantic import BaseModel
from pydantic import field_validator, Field

from ai_trading.logging import get_logger
from ai_trading.logging.redact import redact_env

logger = get_logger(__name__)

class Settings(BaseModel):
    ALPACA_API_KEY: str = Field(default_factory=lambda: os.environ["ALPACA_API_KEY"])
    ALPACA_SECRET_KEY: str = Field(default_factory=lambda: os.environ["ALPACA_SECRET_KEY"])
    ALPACA_BASE_URL: str = Field(default_factory=lambda: os.environ["ALPACA_BASE_URL"])
    TRADING_MODE: str = Field(default_factory=lambda: os.environ.get("TRADING_MODE", "testing"))
    FORCE_TRADES: bool = Field(default_factory=lambda: os.environ.get("FORCE_TRADES", False))

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

def debug_environment() -> dict:
    """Return a structured dump of the current environment.

    The output is intentionally small and side-effect free so tests can
    inspect it safely.  Environment variable values are passed through
    :func:`redact_env` to avoid leaking secrets.
    """

    masked = redact_env(os.environ)
    env_vars = {}
    for name in os.environ:
        raw = os.environ.get(name)
        env_vars[name] = {
            "status": "set",
            "value": masked.get(name, "<unset>"),
            "length": len(str(raw)) if raw is not None else 0,
        }

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "validation_status": "unknown",
        "critical_issues": [],
        "warnings": [],
        "environment_vars": env_vars,
        "recommendations": [],
    }


def validate_specific_env_var(name: str, required: bool = False) -> dict:
    """Validate and report on a specific environment variable."""

    val = os.environ.get(name)
    if val is None:
        result = {
            "variable": name,
            "status": "missing",
            "value": None,
            "issues": [f"{name} is not set"],
        }
        if required:
            raise RuntimeError(f"Missing required env var: {name}")
        return result

    return {
        "variable": name,
        "status": "set",
        "value": val,
        "issues": [],
    }


def main() -> int:
    """Validate critical environment variables.

    Missing credentials are tolerated for dry-run scenarios.
    """
    try:
        Settings()
    except KeyError as exc:
        logger.warning("Missing credential: %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    'Settings',
    'debug_environment',
    'validate_specific_env_var',
    'main',
]
