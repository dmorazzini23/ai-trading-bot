from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

# Authoritative runtime settings come from ai_trading.config.settings (which
# re-exports _base_get_settings from ai_trading.settings in this repo).
from ai_trading.config.settings import get_settings

# Single source of truth for capital sizing math.
from ai_trading.utils.capital_scaling import derive_cap_from_settings as _derive_cap_from_settings


__all__ = [
    "TradingConfig",
    "get_settings",
    "derive_cap_from_settings",
    "build_legacy_params_from_config",
]


@dataclass(frozen=True)
class TradingConfig:
    """
    Minimal, concrete config object expected by call-sites in core/bot_engine.py.
    We do not guess new fields. Only fields observed in code/logs are exposed.
    Values are sourced from central settings via `from_env()`.
    """
    enable_finbert: bool = False
    # Legacy sizing knobs are optional; if absent, we read them from central settings.
    capital_cap: Optional[float] = None
    dollar_risk_limit: Optional[float] = None
    max_position_size: Optional[float] = None  # explicit cap if provided

    @classmethod
    def from_env(cls) -> "TradingConfig":
        s = get_settings()
        # We only materialize what's referenced in code/logs; everything else stays in `s`.
        return cls(
            enable_finbert=bool(getattr(s, "enable_finbert", False)),
            capital_cap=(getattr(s, "capital_cap", None)),
            dollar_risk_limit=(getattr(s, "dollar_risk_limit", None)),
            max_position_size=(getattr(s, "max_position_size", None)),
        )


def derive_cap_from_settings(settings_obj, equity: Optional[float], fallback: float, capital_cap: float) -> float:
    """
    Thin, explicit re-export that delegates to utils.capital_scaling.
    `settings_obj` is accepted for API compatibility but unused.
    """
    return float(_derive_cap_from_settings(equity, capital_cap, fallback))


def build_legacy_params_from_config(cfg: "TradingConfig") -> Tuple[float, float, float]:
    """
    Produces the tuple expected by BotMode in core/bot_engine.py:
        (capital_cap, dollar_risk_limit, max_position_size)

    - capital_cap / dollar_risk_limit: prefer cfg.* if present, otherwise read from central settings.
    - max_position_size: computed by the canonical derive_cap_from_settings() using the
      central settings object. If an explicit 'given' cap is supplied (cfg.max_position_size),
      it is honored; otherwise we pass None to derive a cap from equity * capital_cap.
    """
    s = get_settings()

    # Capital cap
    capital_cap = cfg.capital_cap if cfg.capital_cap is not None else float(getattr(s, "capital_cap", 0.0))
    # Dollar risk limit
    dollar_risk_limit = (
        cfg.dollar_risk_limit if cfg.dollar_risk_limit is not None else float(getattr(s, "dollar_risk_limit", 0.0))
    )

    # Explicit cap (if any) takes precedence; otherwise compute from settings.
    explicit_cap = cfg.max_position_size
    if explicit_cap is not None and explicit_cap != 0:
        max_position_size = float(explicit_cap)
    else:
        max_position_size = derive_cap_from_settings(
            s,
            equity=None,  # the bot_engine call-sites pass None; keep behavior consistent
            fallback=8000.0,
            capital_cap=float(capital_cap),
        )

    return float(capital_cap), float(dollar_risk_limit), float(max_position_size)
