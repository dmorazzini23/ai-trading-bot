"""Runtime settings with env aliases and safe defaults."""

from __future__ import annotations

from functools import lru_cache
from datetime import timedelta
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# AI-AGENT-REF: base directory for resolving relative paths
BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Single source of truth for runtime configuration."""  # AI-AGENT-REF: runtime config model

    # loop control
    interval: int = Field(60, alias="AI_TRADING_INTERVAL")  # AI-AGENT-REF: interval alias
    iterations: int = Field(0, alias="AI_TRADING_ITERATIONS")  # AI-AGENT-REF: iterations alias
    seed: int = Field(42, alias="AI_TRADING_SEED")  # AI-AGENT-REF: seed alias

    # paths
    model_path: str | None = Field("trained_model.pkl", alias="AI_TRADING_MODEL_PATH")  # AI-AGENT-REF: model path
    halt_flag_path: str | None = Field("halt.flag", alias="HALT_FLAG_PATH")  # AI-AGENT-REF: halt flag path
    rl_model_path: str | None = Field("rl_agent.zip", alias="RL_MODEL_PATH")  # AI-AGENT-REF: RL model path

    # RL switch
    use_rl_agent: bool = Field(False, alias="USE_RL_AGENT")  # AI-AGENT-REF: RL toggle

    # cooling
    trade_cooldown_min: int = Field(15, alias="TRADE_COOLDOWN_MIN")  # AI-AGENT-REF: cooldown minutes

    # derived fields
    trade_cooldown: timedelta | None = None  # AI-AGENT-REF: composed timedelta

    @field_validator("model_path", "halt_flag_path", "rl_model_path", mode="before")
    @classmethod
    def _empty_to_default(cls, v, info):  # AI-AGENT-REF: blank string to default
        if v in (None, "", "None"):
            return info.field_info.default
        return v

    @field_validator("trade_cooldown", mode="after")
    def _compose_td(cls, v, info):  # AI-AGENT-REF: build timedelta from minutes
        minutes = info.data.get("trade_cooldown_min", 15)
        return timedelta(minutes=int(minutes))

    def abspath(self, fname: str | None) -> Path:  # AI-AGENT-REF: resolve repo-relative paths
        p = Path(fname or "")
        return p if p.is_absolute() else (BASE_DIR / p)

    @property
    def model_path_abs(self) -> Path:  # AI-AGENT-REF: absolute model path
        return self.abspath(self.model_path)

    @property
    def rl_model_path_abs(self) -> Path:  # AI-AGENT-REF: absolute RL model path
        return self.abspath(self.rl_model_path)

    @property
    def halt_flag_path_abs(self) -> Path:  # AI-AGENT-REF: absolute halt flag path
        return self.abspath(self.halt_flag_path)

    class Config:
        env_prefix = ""  # AI-AGENT-REF: allow AI_TRADING_* and bare names
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor."""  # AI-AGENT-REF: cache settings
    return Settings()  # pydantic-settings auto-loads .env when present
