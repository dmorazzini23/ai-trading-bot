from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel
from pydantic import field_validator, Field

from ai_trading.config.management import (
    get_env,
    merged_env_snapshot,
    validate_no_deprecated_env,
)
from ai_trading.logging import get_logger
from ai_trading.logging.redact import redact_env

logger = get_logger(__name__)


def _required_env(name: str) -> str:
    value = get_env(name, None, cast=str, resolve_aliases=False)
    if value in (None, ""):
        raise KeyError(name)
    return str(value)


def _optional_env(name: str, default: str) -> str:
    value = get_env(name, default, cast=str, resolve_aliases=False)
    return str(value or default)


class Settings(BaseModel):
    ALPACA_API_KEY: str = Field(default_factory=lambda: _required_env("ALPACA_API_KEY"))
    ALPACA_SECRET_KEY: str = Field(default_factory=lambda: _required_env("ALPACA_SECRET_KEY"))
    ALPACA_TRADING_BASE_URL: str = Field(
        default_factory=lambda: _required_env("ALPACA_TRADING_BASE_URL")
    )
    ALPACA_DATA_BASE_URL: str = Field(
        default_factory=lambda: _optional_env("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets")
    )
    AI_TRADING_TRADING_MODE: str = Field(
        default_factory=lambda: _optional_env("AI_TRADING_TRADING_MODE", "balanced")
    )
    FORCE_TRADES: bool = Field(
        default_factory=lambda: bool(get_env("FORCE_TRADES", False, cast=bool, resolve_aliases=False))
    )

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

    @field_validator('ALPACA_TRADING_BASE_URL')
    @classmethod
    def _base_url(cls, v: str) -> str:
        if not v.startswith('https://'):
            raise ValueError('ALPACA_TRADING_BASE_URL must use HTTPS')
        return v

    @field_validator('ALPACA_DATA_BASE_URL')
    @classmethod
    def _data_base_url(cls, v: str) -> str:
        if not v.startswith('https://'):
            raise ValueError('ALPACA_DATA_BASE_URL must use HTTPS')
        return v

    @field_validator('AI_TRADING_TRADING_MODE')
    @classmethod
    def _trading_mode(cls, v: str) -> str:
        if v not in {"conservative", "balanced", "aggressive"}:
            raise ValueError(
                "AI_TRADING_TRADING_MODE must be one of ['conservative', 'balanced', 'aggressive']"
            )
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

    env_snapshot = merged_env_snapshot()
    masked = redact_env(env_snapshot)
    env_vars = {
        name: {
            "status": "set",
            "value": masked.get(name, "<redacted>"),
            "length": len(str(env_snapshot[name])),
        }
        for name in env_snapshot
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

    val = get_env(name, None, cast=str, resolve_aliases=False)
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
        validate_no_deprecated_env()
    except RuntimeError as exc:
        logger.error("Deprecated environment keys are not supported: %s", exc)
        return 1
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
