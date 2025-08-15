"""Runtime settings with env aliases and safe defaults."""

from __future__ import annotations

from functools import lru_cache
from datetime import timedelta
from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # Min confidence threshold
    conf_threshold: float = Field(default=0.8, env="AI_TRADER_CONF_THRESHOLD")
    # Min model buy score
    buy_threshold: float = Field(default=0.4, env="AI_TRADER_BUY_THRESHOLD")
    # Max daily loss fraction before halt
    daily_loss_limit: float = Field(default=0.03, env="AI_TRADER_DAILY_LOSS_LIMIT")
    # Max running drawdown before action
    max_drawdown_threshold: float = Field(default=0.08, env="AI_TRADER_MAX_DRAWDOWN_THRESHOLD")
    # Rebalance drift threshold
    portfolio_drift_threshold: float = Field(default=0.15, env="AI_TRADER_PORTFOLIO_DRIFT_THRESHOLD")
    # Max fraction of equity at risk per trade
    dollar_risk_limit: float = Field(default=0.05, env="AI_TRADER_DOLLAR_RISK_LIMIT")
    # Max fraction of equity per new position
    capital_cap: float = Field(default=0.04, env="AI_TRADER_CAPITAL_CAP")
    # Max fraction of equity in one sector
    sector_exposure_cap: float = Field(default=0.33, env="AI_TRADER_SECTOR_EXPOSURE_CAP")
    # Upper bound on concurrent open positions
    max_portfolio_positions: int = Field(default=10, env="AI_TRADER_MAX_PORTFOLIO_POSITIONS")
    disaster_dd_limit: float = Field(default=0.25, env="AI_TRADER_DISASTER_DD_LIMIT")
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

    # External APIs
    news_api_key: Optional[str] = None  # AI-AGENT-REF: optional News API key
    rebalance_interval_min: int = Field(
        60,
        ge=1,
        description="Minutes between portfolio rebalances",
    )  # AI-AGENT-REF: rebalance interval

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

    model_config = SettingsConfigDict(
        env_prefix="AI_TRADER_",
        extra="ignore",
        case_sensitive=False,
    )  # AI-AGENT-REF: AI_TRADER_ env prefix


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor."""  # AI-AGENT-REF: cache settings
    return Settings()  # pydantic-settings auto-loads .env when present


def get_news_api_key() -> Optional[str]:
    """Lazy accessor for optional News API key."""  # AI-AGENT-REF: runtime News API key
    return get_settings().news_api_key


def get_rebalance_interval_min() -> int:
    """Lazy accessor for rebalance interval."""  # AI-AGENT-REF: runtime rebalance interval
    return int(get_settings().rebalance_interval_min)

# ---- Lazy getters (access only at runtime; never at module import) ----
def get_disaster_dd_limit() -> float:
    return float(get_settings().disaster_dd_limit)

# ---- Lazy getters (access only at runtime; never at module import) ----
def get_max_portfolio_positions() -> int:
    return int(get_settings().max_portfolio_positions)



def get_sector_exposure_cap() -> float:
    return float(get_settings().sector_exposure_cap)


def get_capital_cap() -> float:
    return float(get_settings().capital_cap)


def get_dollar_risk_limit() -> float:
    return float(get_settings().dollar_risk_limit)


def get_portfolio_drift_threshold() -> float:
    return float(get_settings().portfolio_drift_threshold)


def get_max_drawdown_threshold() -> float:
    return float(get_settings().max_drawdown_threshold)


def get_daily_loss_limit() -> float:
    return float(get_settings().daily_loss_limit)


def get_buy_threshold() -> float:
    return float(get_settings().buy_threshold)


def get_conf_threshold() -> float:
    return float(get_settings().conf_threshold)


def get_trade_cooldown_min() -> int:
    return int(get_settings().trade_cooldown_min)
