from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import os

from ai_trading.utils.capital_scaling import derive_cap_from_settings as _derive_cap
from ai_trading.config import get_env as _get_env, reload_env as _reload_env

SEED = 42


def _coerce_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _first(env: Dict[str, str], *keys: str, cast=str):
    for k in keys:
        val = env.get(k)
        if val not in (None, ""):
            return cast(val)
    return None


@dataclass(frozen=True)
class TradingConfig:
    capital_cap: float = 0.04
    dollar_risk_limit: float = 0.05
    max_position_mode: str = "STATIC"
    max_position_size: Optional[float] = None
    max_position_size_pct: float = 0.0

    kelly_fraction_max: float = 0.25
    min_sample_size: int = 30
    confidence_level: float = 0.95
    max_var_95: Optional[float] = None
    min_profit_factor: Optional[float] = None
    min_sharpe_ratio: Optional[float] = None
    min_win_rate: Optional[float] = None

    data_feed: str = "iex"
    data_adjustment: str = "all"
    data_timeframe_day: str = "1Day"
    data_timeframe_min: str = "1Min"
    provider: str = "alpaca"
    paper: bool = True

    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None, **_ignore: Any) -> "TradingConfig":
        env = env or os.environ

        fb = lambda *keys, default=None: _first(env, *keys, cast=float) if _first(env, *keys, cast=float) is not None else default
        fi = lambda *keys, default=None: _first(env, *keys, cast=int) if _first(env, *keys, cast=int) is not None else default
        fs = lambda *keys, default=None: _first(env, *keys, cast=str) or default
        fb_bool = lambda *keys, default=None: _coerce_bool(fs(*keys, default=str(default) if default is not None else None)) if fs(*keys, default=None) is not None else default

        capital_cap = fb("AI_TRADER_CAPITAL_CAP", "CAPITAL_CAP", "POSITION_CAP", default=cls.capital_cap)
        dollar_risk_limit = fb("AI_TRADER_DOLLAR_RISK_LIMIT", "DAILY_LOSS_LIMIT", default=cls.dollar_risk_limit)
        max_position_mode = fs("AI_TRADER_MAX_POSITION_MODE", "MAX_POSITION_MODE", default=cls.max_position_mode)
        max_position_size = fb("AI_TRADER_MAX_POSITION_SIZE", "MAX_POSITION_SIZE", default=cls.max_position_size)
        max_position_size_pct = fb("AI_TRADER_MAX_POSITION_SIZE_PCT", "MAX_POSITION_SIZE_PCT", default=cls.max_position_size_pct)

        kelly_fraction_max = fb("AI_TRADER_KELLY_FRACTION_MAX", "KELLY_FRACTION_MAX", default=cls.kelly_fraction_max)
        min_sample_size = fi("AI_TRADER_MIN_SAMPLE_SIZE", "MIN_SAMPLE_SIZE", default=cls.min_sample_size)
        confidence_level = fb("AI_TRADER_CONFIDENCE_LEVEL", "CONFIDENCE_LEVEL", default=cls.confidence_level)

        max_var_95 = fb("AI_TRADER_MAX_VAR_95", "MAX_VAR_95", default=cls.max_var_95)
        min_profit_factor = fb("AI_TRADER_MIN_PROFIT_FACTOR", "MIN_PROFIT_FACTOR", default=cls.min_profit_factor)
        min_sharpe_ratio = fb("AI_TRADER_MIN_SHARPE_RATIO", "MIN_SHARPE_RATIO", default=cls.min_sharpe_ratio)
        min_win_rate = fb("AI_TRADER_MIN_WIN_RATE", "MIN_WIN_RATE", default=cls.min_win_rate)

        data_feed = fs("AI_TRADER_DATA_FEED", "DATA_FEED", default=cls.data_feed)
        data_adjustment = fs("AI_TRADER_DATA_ADJUSTMENT", "DATA_ADJUSTMENT", default=cls.data_adjustment)
        data_timeframe_day = fs("AI_TRADER_TIMEFRAME_DAY", "DATA_TIMEFRAME_DAY", default=cls.data_timeframe_day)
        data_timeframe_min = fs("AI_TRADER_TIMEFRAME_MIN", "DATA_TIMEFRAME_MIN", default=cls.data_timeframe_min)
        provider = fs("AI_TRADER_PROVIDER", "DATA_PROVIDER", default=cls.provider)
        paper = fb_bool("AI_TRADER_PAPER", "PAPER", default=cls.paper)

        known = {
            "AI_TRADER_CAPITAL_CAP", "CAPITAL_CAP", "POSITION_CAP",
            "AI_TRADER_DOLLAR_RISK_LIMIT", "DAILY_LOSS_LIMIT",
            "AI_TRADER_MAX_POSITION_MODE", "MAX_POSITION_MODE",
            "AI_TRADER_MAX_POSITION_SIZE", "MAX_POSITION_SIZE",
            "AI_TRADER_MAX_POSITION_SIZE_PCT", "MAX_POSITION_SIZE_PCT",
            "AI_TRADER_KELLY_FRACTION_MAX", "KELLY_FRACTION_MAX",
            "AI_TRADER_MIN_SAMPLE_SIZE", "MIN_SAMPLE_SIZE",
            "AI_TRADER_CONFIDENCE_LEVEL", "CONFIDENCE_LEVEL",
            "AI_TRADER_MAX_VAR_95", "MAX_VAR_95",
            "AI_TRADER_MIN_PROFIT_FACTOR", "MIN_PROFIT_FACTOR",
            "AI_TRADER_MIN_SHARPE_RATIO", "MIN_SHARPE_RATIO",
            "AI_TRADER_MIN_WIN_RATE", "MIN_WIN_RATE",
            "AI_TRADER_DATA_FEED", "DATA_FEED",
            "AI_TRADER_DATA_ADJUSTMENT", "DATA_ADJUSTMENT",
            "AI_TRADER_TIMEFRAME_DAY", "DATA_TIMEFRAME_DAY",
            "AI_TRADER_TIMEFRAME_MIN", "DATA_TIMEFRAME_MIN",
            "AI_TRADER_PROVIDER", "DATA_PROVIDER",
            "AI_TRADER_PAPER", "PAPER",
        }
        extras = {k: v for k, v in env.items() if k not in known}

        return cls(
            capital_cap=capital_cap,
            dollar_risk_limit=dollar_risk_limit,
            max_position_mode=max_position_mode,
            max_position_size=max_position_size,
            max_position_size_pct=max_position_size_pct,
            kelly_fraction_max=kelly_fraction_max,
            min_sample_size=min_sample_size,
            confidence_level=confidence_level,
            max_var_95=max_var_95,
            min_profit_factor=min_profit_factor,
            min_sharpe_ratio=min_sharpe_ratio,
            min_win_rate=min_win_rate,
            data_feed=data_feed,
            data_adjustment=data_adjustment,
            data_timeframe_day=data_timeframe_day,
            data_timeframe_min=data_timeframe_min,
            provider=provider,
            paper=paper,
            extras=extras,
        )

    def snapshot_sanitized(self) -> Dict[str, Any]:
        return {
            "capital_cap": self.capital_cap,
            "dollar_risk_limit": self.dollar_risk_limit,
            "max_position_mode": self.max_position_mode,
            "max_position_size": self.max_position_size,
            "max_position_size_pct": self.max_position_size_pct,
            "kelly_fraction_max": self.kelly_fraction_max,
            "min_sample_size": self.min_sample_size,
            "confidence_level": self.confidence_level,
            "data": {
                "feed": self.data_feed,
                "adjustment": self.data_adjustment,
                "timeframe_day": self.data_timeframe_day,
                "timeframe_min": self.data_timeframe_min,
                "provider": self.provider,
                "paper": self.paper,
            },
            "extras": dict(self.extras),
        }

    def derive_cap_from_settings(self, equity: Optional[float], fallback: float = 8000.0) -> float:
        return _derive_cap(equity, self.capital_cap, fallback)


# Legacy alias
Settings = TradingConfig


def derive_cap_from_settings(settings: TradingConfig, equity: Optional[float], fallback: float, capital_cap: Optional[float] = None) -> float:
    cap = capital_cap if capital_cap is not None else settings.capital_cap
    return _derive_cap(equity, cap, fallback)


def get_env(name: str, default: Any = None, *, reload: bool = False, required: bool = False) -> Any:
    return _get_env(name, default, reload=reload, required=required)


def reload_env() -> None:
    _reload_env()


__all__ = [
    "TradingConfig",
    "Settings",
    "derive_cap_from_settings",
    "get_env",
    "reload_env",
    "SEED",
]

