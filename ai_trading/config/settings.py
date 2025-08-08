import functools
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator, AliasChoices

class Settings(BaseSettings):
    # Runtime toggles
    trading_mode: str = Field("balanced", env="TRADING_MODE")
    shadow_mode: bool = Field(False, env="SHADOW_MODE")
    disable_daily_retrain: bool = Field(False, env="DISABLE_DAILY_RETRAIN")

    # Worker sizing (None/0 => auto)
    executor_workers: Optional[int] = Field(None, env="EXECUTOR_WORKERS")
    prediction_workers: Optional[int] = Field(None, env="PREDICTION_WORKERS")

    # Alpaca credentials (dual-schema)
    alpaca_api_key: Optional[str] = Field(None, validation_alias=AliasChoices("ALPACA_API_KEY", "APCA_API_KEY_ID"))
    alpaca_secret_key: Optional[str] = Field(None, validation_alias=AliasChoices("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY"))
    alpaca_base_url: Optional[str] = Field(None, validation_alias=AliasChoices("ALPACA_BASE_URL", "APCA_API_BASE_URL"))

    # Risk knobs (align defaults with current behavior)
    capital_cap: float = Field(0.04, env="CAPITAL_CAP")
    dollar_risk_limit: float = Field(0.05, env="DOLLAR_RISK_LIMIT")
    max_position_size: int = Field(8000, env="MAX_POSITION_SIZE")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }

    @model_validator(mode='after')
    def _normalize_alpaca(self):
        def clean(v: Optional[str]) -> Optional[str]:
            if not v:
                return None
            return v.strip().strip('"').strip("'") or None
        self.alpaca_api_key = clean(self.alpaca_api_key)
        self.alpaca_secret_key = clean(self.alpaca_secret_key)
        self.alpaca_base_url = clean(self.alpaca_base_url) or "https://paper-api.alpaca.markets"
        return self

    # Enforcement
    def require_alpaca_or_raise(self) -> None:
        if self.shadow_mode:
            return
        if not (self.alpaca_api_key and self.alpaca_secret_key):
            raise RuntimeError("Missing Alpaca API credentials")

    # Policy: worker sizing
    def effective_executor_workers(self, cpu_count: int) -> int:
        return (self.executor_workers or 0) or max(2, min(4, cpu_count or 2))

    def effective_prediction_workers(self, cpu_count: int) -> int:
        return (self.prediction_workers or 0) or max(2, min(4, cpu_count or 2))

@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()