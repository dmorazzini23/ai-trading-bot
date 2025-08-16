"""CLI helper for environment validation."""  # AI-AGENT-REF: script entrypoint

import sys
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Minimal settings used for validation in tests."""  # AI-AGENT-REF: pydantic model

    ALPACA_API_KEY: str = Field(...)
    ALPACA_SECRET_KEY: str = Field(...)
    ALPACA_BASE_URL: str = Field(...)
    BOT_MODE: str = Field(default="balanced")
    TRADING_MODE: str = Field(default="paper")
    FORCE_TRADES: bool = Field(default=False)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("ALPACA_API_KEY")
    @classmethod
    def _api_key(cls, v: str) -> str:
        if not v:
            raise ValueError("ALPACA_API_KEY missing")
        return v

    @field_validator("ALPACA_SECRET_KEY")
    @classmethod
    def _secret_key(cls, v: str) -> str:
        if len(v) < 10:
            raise ValueError("ALPACA_SECRET_KEY too short")
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
        return v

    @field_validator("TRADING_MODE")
    @classmethod
    def _trading_mode(cls, v: str) -> str:
        return v

    @field_validator("FORCE_TRADES")
    @classmethod
    def _force_trades(cls, v: bool) -> bool:
        return v


def validate_env() -> tuple[bool, List[str]]:
    """Validate required environment variables."""  # AI-AGENT-REF: validation entry
    try:
        Settings()
        return True, []
    except Exception as e:  # pragma: no cover - simple reporting
        return False, [str(e)]


try:
    settings = Settings()
except Exception:
    settings = Settings.model_construct()  # type: ignore[attr-defined]


def _main() -> int:
    ok, problems = validate_env()
    if not ok:
        for p in problems:
            print(p, file=sys.stderr)
        return 1
    print("Environment OK")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
