from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

# Authoritative runtime settings come from ai_trading.config.settings (which
# re-exports _base_get_settings from ai_trading.settings in this repo).
from ai_trading.config.settings import get_settings

# Single source of truth for capital sizing math.
from ai_trading.utils.capital_scaling import derive_cap_from_settings as _derive_cap_from_settings


__all__ = [
    "TradingConfig",
    "get_settings",
    "derive_cap_from_settings",
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

    def get_legacy_params(self) -> Dict[str, Any]:
        """Return minimal legacy params for runner compatibility.

        AI-AGENT-REF: expose legacy params without external helpers
        """
        from ai_trading import settings as S  # lazy import to avoid cycles

        params: Dict[str, Any] = {
            "CAPITAL_CAP": float(getattr(S, "get_capital_cap")()),
            "DOLLAR_RISK_LIMIT": float(getattr(S, "get_dollar_risk_limit")()),
        }
        try:
            from ai_trading.position_sizing import resolve_max_position_size
            mps = resolve_max_position_size(capital_cap=params["CAPITAL_CAP"])
            if mps is not None:
                params["MAX_POSITION_SIZE"] = float(mps)
        except Exception:
            pass
        return params


def derive_cap_from_settings(settings_obj, equity: Optional[float], fallback: float, capital_cap: float) -> float:
    """
    Thin, explicit re-export that delegates to utils.capital_scaling.
    `settings_obj` is accepted for API compatibility but unused.
    """
    return float(_derive_cap_from_settings(equity, capital_cap, fallback))
