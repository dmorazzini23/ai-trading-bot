from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, TypeVar

# Authoritative runtime settings come from ai_trading.config.settings (which
# Lazy accessors avoid optional dependency imports at module import time.
# re-exports _base_get_settings from ai_trading.settings in this repo.
def get_settings():  # type: ignore[override]
    from ai_trading.config.settings import get_settings as _gs

    return _gs()

# Single source of truth for capital sizing math.
from ai_trading.utils.capital_scaling import derive_cap_from_settings as _derive_cap_from_settings

T = TypeVar("T")


def _to_bool(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def get_env(
    key: str,
    default: Optional[str] = None,
    *,
    cast: Optional[Callable[[str], T]] = None,
    required: bool = False,
) -> T | str | None:
    """Fetch an environment variable with optional casting and required check.

    AI-AGENT-REF: typed env access helper
    Assumes dotenv was already loaded by the top-level startup code.
    """
    raw = os.environ.get(key, default)
    if raw is None:
        if required:
            raise RuntimeError(f"Missing required environment variable: {key}")
        return None
    if cast is None:
        return raw
    if cast is bool:
        return _to_bool(str(raw))  # type: ignore[return-value]
    try:
        return cast(raw)  # type: ignore[misc]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to cast env var {key!r}={raw!r} via {cast}: {e}"
        ) from e


def is_shadow_mode() -> bool:
    """Return True when the ``SHADOW_MODE`` env var is truthy."""
    return bool(get_env("SHADOW_MODE", "0", cast=bool))


# Canonical runtime seed used by risk/engine and anywhere else needing determinism.
SEED: int = int(os.environ.get("SEED", "42"))  # AI-AGENT-REF: expose runtime seed

# Required environment variables for a functional deployment. "MAX_POSITION_SIZE"
# is intentionally excluded; when unset the runtime derives an appropriate value
# based on capital constraints.
_MANDATORY_ENV_VARS: tuple[str, ...] = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_API_URL",
    "WEBHOOK_SECRET",
    "CAPITAL_CAP",
    "DOLLAR_RISK_LIMIT",
)


def _mask(val: str) -> str:
    return "***" if val else ""


def validate_required_env(
    keys: Iterable[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Dict[str, str]:
    """Validate presence of mandatory environment variables.

    Parameters
    ----------
    keys:
        Specific keys to validate. Defaults to the canonical mandatory set.
    env:
        Optional mapping to read from (defaults to ``os.environ``).

    Returns
    -------
    dict
        Mapping of the checked keys with their values masked for safe logging.

    Raises
    ------
    RuntimeError
        If any required key is missing or empty.
    """

    env = dict(env or os.environ)
    required = list(keys or _MANDATORY_ENV_VARS)
    missing: list[str] = []
    snapshot: Dict[str, str] = {}
    for k in required:
        if k in {"ALPACA_API_URL", "ALPACA_BASE_URL"}:
            val = env.get("ALPACA_API_URL") or env.get("ALPACA_BASE_URL", "")
            if not val.strip():
                missing.append(k)
            snapshot[k] = _mask(val)
            continue
        val = env.get(k, "")
        if not val.strip():
            missing.append(k)
        snapshot[k] = _mask(val)
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    return snapshot


__all__ = [
    "TradingConfig",
    "get_settings",
    "derive_cap_from_settings",
    "get_env",
    "is_shadow_mode",
    "SEED",
    "validate_required_env",
]

# NOTE: Keep dotenv imports inside the function to avoid import-time costs.
def reload_env(path: str | None = None, *, override: bool = True) -> str | None:
    """Re-load environment variables from `.env` (or specified path).

    Safe no-op if python-dotenv is unavailable or file is missing.
    Returns the path that was loaded (if any).

    AI-AGENT-REF: reload dotenv with override control
    """
    try:
        from dotenv import load_dotenv, find_dotenv  # type: ignore
    except Exception:
        return None

    # Explicit path first
    if path:
        p = os.fspath(path)
        if os.path.exists(p):
            load_dotenv(p, override=override)
            return p
        return None

    # Otherwise, discover `.env` from CWD upward
    p = find_dotenv(usecwd=True)
    if p:
        load_dotenv(p, override=override)
        return p or None
    return None

__all__.append("reload_env")


@dataclass(frozen=True)
class TradingConfig:
    """
    Minimal, concrete config object expected by call-sites in core/bot_engine.py.
    Values are sourced from environment variables via ``from_env``.
    """
    seed: int = SEED  # AI-AGENT-REF: propagate runtime seed
    enable_finbert: bool = False
    disable_daily_retrain: bool = False
    capital_cap: Optional[float] = None
    dollar_risk_limit: Optional[float] = None
    max_position_size: Optional[float] = None
    max_position_equity_fallback: float = 200000.0
    sector_exposure_cap: Optional[float] = None
    max_drawdown_threshold: Optional[float] = None
    trailing_factor: Optional[float] = None
    take_profit_factor: Optional[float] = None
    max_position_size_pct: Optional[float] = None
    max_var_95: Optional[float] = None
    min_profit_factor: Optional[float] = None
    min_sharpe_ratio: Optional[float] = None
    min_win_rate: Optional[float] = None
    kelly_fraction_max: float = 0.25
    min_sample_size: int = 10
    confidence_level: float = 0.90
    max_position_mode: str = "STATIC"
    paper: bool = True
    data_feed: Optional[str] = None
    data_provider: Optional[str] = None

    # --- Ergonomics: safe update & dict view ---
    def update(self, **kwargs) -> None:
        """Update known fields only; raise on unknown key."""
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"TradingConfig has no field '{k}'")
            object.__setattr__(self, k, v)

    def to_dict(self) -> dict[str, object]:
        """Return a dict of current tunables (no secrets)."""
        keys = [
            "seed",
            "enable_finbert",
            "disable_daily_retrain",
            "capital_cap",
            "dollar_risk_limit",
            "max_position_size",
            "max_position_equity_fallback",
            "sector_exposure_cap",
            "max_drawdown_threshold",
            "trailing_factor",
            "take_profit_factor",
            "max_position_size_pct",
            "max_var_95",
            "min_profit_factor",
            "min_sharpe_ratio",
            "min_win_rate",
            "kelly_fraction_max",
            "min_sample_size",
            "confidence_level",
            "max_position_mode",
            "paper",
            "data_feed",
            "data_provider",
        ]
        return {k: getattr(self, k) for k in keys}

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None
    ) -> "TradingConfig":
        env_map = {k.upper(): v for k, v in (env or os.environ).items()}

        def _get(
            key: str,
            cast: Callable[[str], Any] | type = str,
            default: Any = None,
            aliases: Iterable[str] = (),
        ):
            for k in (key, *aliases):
                if k in env_map and env_map[k] != "":
                    val = env_map[k]
                    return cast(val) if cast is not str else val
            return default

        mps = _get("MAX_POSITION_SIZE", float)
        if mps is not None and mps <= 0:
            raise ValueError("MAX_POSITION_SIZE must be positive")

        return cls(
            capital_cap=_get("CAPITAL_CAP", float),
            dollar_risk_limit=_get(
                "DOLLAR_RISK_LIMIT", float, aliases=("DAILY_LOSS_LIMIT",)
            ),
            max_position_size=mps,
            max_position_equity_fallback=_get(
                "MAX_POSITION_EQUITY_FALLBACK", float, default=200000.0
            ),
            sector_exposure_cap=_get("SECTOR_EXPOSURE_CAP", float),
            max_drawdown_threshold=_get("MAX_DRAWDOWN_THRESHOLD", float),
            trailing_factor=_get("TRAILING_FACTOR", float),
            take_profit_factor=_get("TAKE_PROFIT_FACTOR", float),
            max_position_size_pct=_get("MAX_POSITION_SIZE_PCT", float),
            max_var_95=_get("MAX_VAR_95", float),
            min_profit_factor=_get("MIN_PROFIT_FACTOR", float),
            min_sharpe_ratio=_get("MIN_SHARPE_RATIO", float),
            min_win_rate=_get("MIN_WIN_RATE", float),
            kelly_fraction_max=_get(
                "KELLY_FRACTION_MAX",
                float,
                default=0.25,
                aliases=("AI_TRADING_KELLY_FRACTION_MAX",),
            ),
            min_sample_size=_get(
                "MIN_SAMPLE_SIZE",
                int,
                default=10,
                aliases=("AI_TRADING_MIN_SAMPLE_SIZE",),
            ),
            confidence_level=_get(
                "CONFIDENCE_LEVEL",
                float,
                default=0.90,
                aliases=("AI_TRADING_CONFIDENCE_LEVEL",),
            ),
            max_position_mode=_get("MAX_POSITION_MODE", str, default="STATIC"),
            paper=_get("PAPER", _to_bool, default=True),
            disable_daily_retrain=_get(
                "DISABLE_DAILY_RETRAIN", _to_bool, default=False
            ),
            data_feed=_get("DATA_FEED", str),
            data_provider=_get("DATA_PROVIDER", str),
        )

    def snapshot_sanitized(self) -> Dict[str, Any]:
        """Return a sanitized dict suitable for logging."""
        return {
            "capital_cap": self.capital_cap,
            "dollar_risk_limit": self.dollar_risk_limit,
            "max_position_mode": self.max_position_mode,
            "data": {
                "feed": self.data_feed,
                "provider": self.data_provider,
            },
        }

    @property
    def max_correlation_exposure(self) -> Optional[float]:
        """Back-compat alias for ``sector_exposure_cap``."""
        return self.sector_exposure_cap

    @property
    def max_drawdown(self) -> Optional[float]:
        """Back-compat alias for ``max_drawdown_threshold``."""
        return self.max_drawdown_threshold

    @property
    def stop_loss_multiplier(self) -> Optional[float]:
        """Back-compat alias for ``trailing_factor``."""
        return self.trailing_factor

    @property
    def take_profit_multiplier(self) -> Optional[float]:
        """Back-compat alias for ``take_profit_factor``."""
        return self.take_profit_factor


def derive_cap_from_settings(settings_obj, equity: Optional[float], fallback: float, capital_cap: float) -> float:
    """
    Thin, explicit re-export that delegates to utils.capital_scaling.
    `settings_obj` is accepted for API compatibility but unused.
    """
    return float(_derive_cap_from_settings(equity, capital_cap, fallback))
