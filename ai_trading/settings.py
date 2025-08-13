"""Runtime settings with env aliases and sane defaults."""

from __future__ import annotations

from functools import lru_cache
from datetime import timedelta
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, AliasChoices


class Settings(BaseSettings):
    """Env-driven runtime settings."""

    # --- Loop controls ---
    interval: int = Field(60, validation_alias=AliasChoices("AI_TRADING_INTERVAL", "INTERVAL"))
    iterations: int = Field(0, validation_alias=AliasChoices("AI_TRADING_ITERATIONS", "ITERATIONS"))
    seed: int = Field(42, validation_alias=AliasChoices("AI_TRADING_SEED", "SEED"))

    # --- Paths ---
    model_path: str = Field(
        "trained_model.pkl",
        validation_alias=AliasChoices("AI_TRADING_MODEL_PATH", "MODEL_PATH"),
    )
    halt_flag_path: str = Field(
        "halt.flag",
        validation_alias=AliasChoices("HALT_FLAG_PATH", "AI_TRADING_HALT_FLAG_PATH"),
    )

    # --- Trading / risk knobs ---
    trade_cooldown_min: int = Field(15, validation_alias=AliasChoices("TRADE_COOLDOWN_MIN",))

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # AI-AGENT-REF: defensive int coercion
    @field_validator("interval", "iterations", "seed", "trade_cooldown_min", mode="before")
    @classmethod
    def _coerce_int(cls, v):
        if v in (None, "", "None", "none", "null"):
            return None
        return int(v)

    @field_validator("model_path", "halt_flag_path", mode="before")
    @classmethod
    def _default_nonempty(cls, v):
        if v in (None, "", "None", "none", "null"):
            return None
        return str(v)

    @property
    def trade_cooldown(self) -> timedelta:
        """Return a ready-to-use cooldown timedelta."""
        return timedelta(minutes=int(self.trade_cooldown_min))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings loader."""
    return Settings()

