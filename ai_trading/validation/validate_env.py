from __future__ import annotations

import os

from pydantic import BaseModel, Field, field_validator

# AI-AGENT-REF: facade for environment validation


class Settings(BaseModel):
    ALPACA_API_KEY: str = Field(...)
    ALPACA_SECRET_KEY: str = Field(...)
    ALPACA_BASE_URL: str = Field(...)
    BOT_MODE: str = Field("testing")
    TRADING_MODE: str = Field("paper")
    FORCE_TRADES: bool = Field(False)

    @field_validator("ALPACA_API_KEY")
    @classmethod
    def _api_key(cls, v: str) -> str:
        if len(v) < 16:
            raise ValueError("ALPACA_API_KEY appears too short")
        return v

    @field_validator("ALPACA_SECRET_KEY")
    @classmethod
    def _secret_key(cls, v: str) -> str:
        if len(v) < 16:
            raise ValueError("ALPACA_SECRET_KEY appears too short")
        return v

    @field_validator("ALPACA_BASE_URL")
    @classmethod
    def _base_url(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError("ALPACA_BASE_URL must use HTTPS")
        return v

    @field_validator("BOT_MODE")
    @classmethod
    def _bot_mode(cls, v: str) -> str:
        if v not in {"testing", "production"}:
            raise ValueError("BOT_MODE must be one of ['testing', 'production']")
        return v

    @field_validator("TRADING_MODE")
    @classmethod
    def _trading_mode(cls, v: str) -> str:
        if v not in {"paper", "live"}:
            raise ValueError("Invalid TRADING_MODE")
        return v

    @field_validator("FORCE_TRADES")
    @classmethod
    def _force_trades(cls, v: bool | str) -> bool:
        if isinstance(v, str):
            v = v.lower() in {"1", "true", "yes", "on"}
        return bool(v)


def debug_environment() -> dict:
    """Return a tiny dump used by tests without side effects."""
    return {"pythonpath": os.environ.get("PYTHONPATH", ""), "env": dict(os.environ)}


def validate_specific_env_var(name: str, required: bool = False) -> str | None:
    val = os.environ.get(name)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


__all__ = [
    "Settings",
    "debug_environment",
    "validate_specific_env_var",
]
