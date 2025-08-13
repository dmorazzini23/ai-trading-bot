from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Minimal project settings for tests and CI."""

    env: str = Field(default="test", alias="APP_ENV")
    data_provider: str = Field(default="mock", alias="DATA_PROVIDER")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    enable_memory_optimization: bool = Field(default=True)
    alpaca_api_key: str = Field(default="test_key", alias="ALPACA_API_KEY")
    alpaca_secret_key: SecretStr = Field(
        default=SecretStr("test_secret"), alias="ALPACA_SECRET_KEY"
    )
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")

    model_config = SettingsConfigDict(env_prefix="AI_TRADING_", extra="ignore")

    @property
    def alpaca_secret_key_plain(self) -> str:
        return self.alpaca_secret_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def broker_keys() -> tuple[str, str]:
    s = get_settings()
    return (s.alpaca_api_key, s.alpaca_secret_key_plain)

