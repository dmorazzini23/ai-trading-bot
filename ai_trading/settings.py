"""Runtime settings with env aliases and safe defaults."""

from __future__ import annotations

from functools import lru_cache
from datetime import timedelta
from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for runtime configuration."""  # AI-AGENT-REF: runtime config model

    # loop control
    interval: int = Field(60, alias="AI_TRADING_INTERVAL")  # AI-AGENT-REF: interval alias
    iterations: int = Field(0, alias="AI_TRADING_ITERATIONS")  # AI-AGENT-REF: iterations alias
    seed: int = Field(42, alias="AI_TRADING_SEED")  # AI-AGENT-REF: seed alias

    # paths
    model_path: str = Field("trained_model.pkl", alias="AI_TRADING_MODEL_PATH")  # AI-AGENT-REF: model path
    halt_flag_path: str = Field("halt.flag", alias="HALT_FLAG_PATH")  # AI-AGENT-REF: halt flag path
    rl_model_path: str = Field("rl_agent.zip", alias="AI_TRADING_RL_MODEL_PATH")  # AI-AGENT-REF: RL model path

    # RL switch
    use_rl_agent: bool = Field(False, alias="USE_RL_AGENT")  # AI-AGENT-REF: RL toggle

    # cooling
    trade_cooldown_min: int = Field(15, alias="TRADE_COOLDOWN_MIN")  # AI-AGENT-REF: cooldown minutes

    @field_validator("model_path", "halt_flag_path", "rl_model_path", mode="before")
    @classmethod
    def _empty_to_default(cls, v, info):  # AI-AGENT-REF: blank string to default
        if v in (None, "", "None"):
            return info.field_info.default
        return v

    # AI-AGENT-REF: derive timedelta from minutes
    @computed_field
    @property
    def trade_cooldown(self) -> timedelta:
        return timedelta(minutes=self.trade_cooldown_min)

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")  # AI-AGENT-REF: allow AI_TRADING_* and bare names


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor."""  # AI-AGENT-REF: cache settings
    return Settings()  # pydantic-settings auto-loads .env when present
