from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, TypeVar

# Authoritative runtime settings come from ai_trading.config.settings (which
# re-exports _base_get_settings from ai_trading.settings in this repo).
from ai_trading.config.settings import get_settings

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


# Canonical runtime seed used by risk/engine and anywhere else needing determinism.
SEED: int = int(os.environ.get("SEED", "42"))  # AI-AGENT-REF: expose runtime seed

# Required environment variables for a functional deployment
_MANDATORY_ENV_VARS: tuple[str, ...] = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_BASE_URL",
    "WEBHOOK_SECRET",
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
    missing = [k for k in required if not env.get(k, "").strip()]
    snapshot = {k: _mask(env.get(k, "")) for k in required}
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
    We do not guess new fields. Only fields observed in code/logs are exposed.
    Values are sourced from central settings via `from_env()`.
    """
    seed: int = SEED  # AI-AGENT-REF: propagate runtime seed
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
            seed=SEED,
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
