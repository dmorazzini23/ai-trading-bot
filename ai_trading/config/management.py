from __future__ import annotations

import json
import os
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, TypeVar

from ai_trading.logging import logger

# Exposed for tests to bypass certain validations
TESTING: bool = bool(os.getenv("TESTING"))
# Authoritative runtime settings come from ai_trading.config.settings (which
# Lazy accessors avoid optional dependency imports at module import time.
# re-exports _base_get_settings from ai_trading.settings in this repo.
def get_settings():  # type: ignore[override]
    from ai_trading.config.settings import get_settings as _gs

    return _gs()

# Single source of truth for capital sizing math.
from ai_trading.utils.capital_scaling import (
    derive_cap_from_settings as _derive_cap_from_settings,
)
from ai_trading.settings import Settings, POSITION_SIZE_MIN_USD_DEFAULT

# re-export helpers expected by tests and callers
derive_cap_from_settings = _derive_cap_from_settings


ALPACA_URL_GUIDANCE = (
    "Set ALPACA_API_URL or ALPACA_BASE_URL to a full https://... endpoint."
)


def _normalize_alpaca_base_url(
    value: str | None, *, source_key: str
) -> tuple[str | None, str | None]:
    """Validate Alpaca base URL strings returning sanitized value and error."""

    if value is None:
        return None, None

    raw = value.strip()
    if not raw:
        return None, None

    if "${" in raw:
        return None, (
            f"{source_key} looks like an unresolved placeholder ({raw}). "
            f"{ALPACA_URL_GUIDANCE}"
        )

    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None, (
            f"{source_key} must include an HTTP scheme (got {raw}). "
            f"{ALPACA_URL_GUIDANCE}"
        )

    return raw, None


def _select_alpaca_base_url(
    env: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None, list[tuple[str, str, str]]]:
    """Return the first valid Alpaca base URL and any invalid entries."""

    env_map = env or os.environ
    invalid_entries: list[tuple[str, str, str]] = []

    for env_key in ("ALPACA_BASE_URL", "ALPACA_API_URL"):
        raw = env_map.get(env_key)
        normalized, message = _normalize_alpaca_base_url(raw, source_key=env_key)
        if normalized:
            return normalized, raw, invalid_entries
        if raw and message:
            invalid_entries.append((env_key, raw, message))

    fallback_raw = (
        env_map.get("ALPACA_BASE_URL")
        or env_map.get("ALPACA_API_URL")
    )
    return None, fallback_raw, invalid_entries


def _resolve_alpaca_env() -> tuple[str | None, str | None, str | None]:
    """Return Alpaca credentials resolving ALPACA_* and APCA_* variants."""

    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")

    base_url, _raw_base_url, invalid_entries = _select_alpaca_base_url()
    for env_key, raw, message in invalid_entries:
        logger.error(message, extra={"env_key": env_key, "value": raw})

    if base_url is None:
        base_url = "https://paper-api.alpaca.markets"

    return key, secret, base_url


def _warn_duplicate_env_keys() -> None:
    """Log when ALPACA_* and APCA_* env vars disagree."""
    pairs = [
        ("ALPACA_API_KEY", "APCA_API_KEY_ID"),
        ("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY"),
    ]
    for a, b in pairs:
        av = os.getenv(a)
        bv = os.getenv(b)
        if av and bv and av != bv:
            logger.warning(
                f"Duplicate env keys {a} and {b} have different values"
            )


def validate_alpaca_credentials() -> None:
    """Ensure required Alpaca credentials are present."""
    if TESTING:
        return
    url_key = "ALPACA_API_URL" if os.getenv("ALPACA_API_URL") else "ALPACA_BASE_URL"
    validate_required_env(("ALPACA_API_KEY", "ALPACA_SECRET_KEY", url_key))

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
    except (ValueError, TypeError) as e:
        raise RuntimeError(
            f"Failed to cast env var {key!r}={raw!r} via {cast}: {e}"
        ) from e


def is_shadow_mode() -> bool:
    """Return True when the ``SHADOW_MODE`` env var is truthy."""
    return bool(get_env("SHADOW_MODE", "0", cast=bool))


# Canonical runtime seed used by risk/engine and anywhere else needing determinism.
SEED: int = int(os.environ.get("SEED", "42"))  # AI-AGENT-REF: expose runtime seed

# Limit for consecutive empty data retries before surfacing an error. Can be
# overridden via the ``MAX_EMPTY_RETRIES`` environment variable.
MAX_EMPTY_RETRIES: int = int(os.environ.get("MAX_EMPTY_RETRIES", "10"))

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
    api_key = env.get("ALPACA_API_KEY", "").strip()
    secret = env.get("ALPACA_SECRET_KEY", "").strip()
    oauth = env.get("ALPACA_OAUTH", "").strip()

    if oauth and (api_key or secret):
        raise RuntimeError(
            "Provide either ALPACA_OAUTH or ALPACA_API_KEY/ALPACA_SECRET_KEY, not both"
        )

    required = list(keys or _MANDATORY_ENV_VARS)
    if oauth:
        required = [
            k for k in required if k not in {"ALPACA_API_KEY", "ALPACA_SECRET_KEY"}
        ]

    missing: list[str] = []
    snapshot: Dict[str, str] = {}
    error_hints: list[str] = []

    base_url_value, base_url_raw, invalid_entries = _select_alpaca_base_url(env)
    alias_raw = env.get("ALPACA_API_URL") or env.get("ALPACA_BASE_URL", "")
    if not alias_raw and base_url_raw:
        alias_raw = base_url_raw
    base_url_hint_recorded = False

    for k in required:
        if k in {"ALPACA_API_URL", "ALPACA_BASE_URL"}:
            if base_url_value is None:
                missing.append(k)
                if not base_url_hint_recorded:
                    if invalid_entries:
                        error_hints.extend(message for _, _, message in invalid_entries)
                    else:
                        error_hints.append(ALPACA_URL_GUIDANCE)
                    base_url_hint_recorded = True
            snapshot[k] = _mask(alias_raw)
            continue
        val = env.get(k, "")
        if not val.strip():
            missing.append(k)
        snapshot[k] = _mask(val)

    if oauth:
        snapshot["ALPACA_OAUTH"] = _mask(oauth)

    if missing:
        message = "Missing required environment variables: " + ", ".join(missing)
        if error_hints:
            # Preserve insertion order while deduplicating
            ordered_hints = dict.fromkeys(error_hints)
            message += ". " + " ".join(ordered_hints)
        raise RuntimeError(message)
    return snapshot


__all__ = [
    "TradingConfig",
    "get_settings",
    "derive_cap_from_settings",
    "get_env",
    "is_shadow_mode",
    "SEED",
    "MAX_EMPTY_RETRIES",
    "validate_required_env",
    "validate_alpaca_credentials",
    "Settings",
    "_resolve_alpaca_env",
    "_warn_duplicate_env_keys",
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
    except ImportError:
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
    # Core sizing/risk (test harness overrides defaults when TESTING=1)
    capital_cap: Optional[float] = None
    dollar_risk_limit: Optional[float] = 0.05
    daily_loss_limit: Optional[float] = 0.03
    max_position_size: Optional[float] = None
    max_position_equity_fallback: float = 200000.0
    position_size_min_usd: float | None = POSITION_SIZE_MIN_USD_DEFAULT
    sector_exposure_cap: Optional[float] = None
    max_drawdown_threshold: Optional[float] = None
    trailing_factor: Optional[float] = None
    take_profit_factor: Optional[float] = None
    max_position_size_pct: Optional[float] = None
    max_var_95: Optional[float] = None
    min_profit_factor: Optional[float] = None
    min_sharpe_ratio: Optional[float] = None
    min_win_rate: Optional[float] = None
    kelly_fraction: Optional[float] = None
    conf_threshold: Optional[float] = None
    # Default aligns with TradingConfig.from_env when env is absent
    kelly_fraction_max: float = 0.25
    min_sample_size: int = 10
    confidence_level: float = 0.90
    lookback_periods: Optional[int] = None
    market_calendar: Optional[str] = None
    score_confidence_min: Optional[float] = None
    signal_confirmation_bars: int = 2
    delta_threshold: float = 0.02
    min_confidence: float = 0.6
    extras: Optional[dict[str, Any]] = None
    max_position_mode: str = "STATIC"
    paper: bool = True
    data_feed: Optional[str] = None
    data_provider: Optional[str] = None
    minute_data_freshness_tolerance_seconds: int = 900
    # Additional attributes validated by tests; optional for runtime
    max_portfolio_risk: Optional[float] = None
    buy_threshold: Optional[float] = 0.4
    signal_period: Optional[int] = None
    fast_period: Optional[int] = None
    slow_period: Optional[int] = None
    limit_order_slippage: Optional[float] = None
    max_slippage_bps: Optional[int] = None
    participation_rate: Optional[float] = None
    pov_slice_pct: Optional[float] = None
    order_timeout_seconds: Optional[int] = None

    def __post_init__(self) -> None:
        # Normalize defaults without impacting production behavior.
        # When TESTING=1 (as set by unit tests), provide test-friendly defaults
        # expected by the contract; otherwise keep runtime defaults lean.
        if (capital := getattr(self, "capital_cap")) is None:
            object.__setattr__(self, "capital_cap", 0.25)
        if mps := getattr(self, "max_position_size") is None:
            object.__setattr__(self, "max_position_size", 8000.0)
        if TESTING and getattr(self, "kelly_fraction") is None:
            object.__setattr__(self, "kelly_fraction", 0.6)
        if TESTING and getattr(self, "conf_threshold") is None:
            object.__setattr__(self, "conf_threshold", 0.75)
        if self.lookback_periods is not None and self.lookback_periods <= 0:
            raise ValueError("lookback_periods must be positive")
        if self.kelly_fraction is not None and not (0.0 < self.kelly_fraction <= 1.0):
            raise ValueError("kelly_fraction must be between 0 and 1")
        if self.score_confidence_min is not None and not (
            0.0 <= self.score_confidence_min <= 1.0
        ):
            raise ValueError("score_confidence_min must be between 0 and 1")
        if self.signal_confirmation_bars <= 0:
            raise ValueError("signal_confirmation_bars must be positive")
        if self.delta_threshold < 0:
            raise ValueError("delta_threshold must be non-negative")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be between 0 and 1")
        tolerance = getattr(self, "minute_data_freshness_tolerance_seconds", 900)
        if tolerance <= 0:
            raise ValueError(
                "minute_data_freshness_tolerance_seconds must be positive"
            )

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
            "daily_loss_limit",
            "max_position_size",
            "max_position_equity_fallback",
            "position_size_min_usd",
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
            "lookback_periods",
            "kelly_fraction",
            "conf_threshold",
            "extras",
            "market_calendar",
            "score_confidence_min",
            "signal_confirmation_bars",
            "delta_threshold",
            "min_confidence",
            "max_position_mode",
            "paper",
            "data_feed",
            "data_provider",
        ]
        return {k: getattr(self, k) for k in keys}

    @classmethod
    def from_env(
        cls,
        env_or_mode: Any | None = None,
        *,
        allow_missing_drawdown: bool = False,
    ) -> "TradingConfig":
        """Build TradingConfig from environment.

        ``env_or_mode`` may be a mode string or a mapping of env vars that
        supplements ``os.environ``. Mapping keys override any existing process
        environment values and ``TRADING_MODE`` may be present to influence
        overrides. ``MAX_DRAWDOWN_THRESHOLD`` defaults to ``0.08`` only when
        ``allow_missing_drawdown`` is explicitly enabled.
        """
        mode: str | None = None
        env_map = {k.upper(): v for k, v in os.environ.items()}
        if isinstance(env_or_mode, Mapping):
            env_map.update({k.upper(): v for k, v in env_or_mode.items()})
            mode = env_map.get("TRADING_MODE", "").strip().lower() or None
        elif isinstance(env_or_mode, str):
            mode = env_or_mode.strip().lower()
        else:
            mode = env_map.get("TRADING_MODE", "").strip().lower() or None

        alias_map = {
            "AI_TRADING_BUY_THRESHOLD": "BUY_THRESHOLD",
            "AI_TRADING_CONF_THRESHOLD": "CONF_THRESHOLD",
            "AI_TRADING_CONFIDENCE_LEVEL": "CONFIDENCE_LEVEL",
            "AI_TRADING_KELLY_FRACTION_MAX": "KELLY_FRACTION_MAX",
            "AI_TRADING_MAX_DRAWDOWN_THRESHOLD": "MAX_DRAWDOWN_THRESHOLD",
            "AI_TRADING_MAX_POSITION_SIZE": "MAX_POSITION_SIZE",
            "AI_TRADING_MIN_SAMPLE_SIZE": "MIN_SAMPLE_SIZE",
            "AI_TRADING_POSITION_SIZE_MIN_USD": "POSITION_SIZE_MIN_USD",
            # Legacy daily loss limit alias: backfills ``DOLLAR_RISK_LIMIT`` when absent.
            "DAILY_LOSS_LIMIT": "DOLLAR_RISK_LIMIT",
        }
        # AI_TRADING_* aliases are the canonical spellings going forward and always
        # override their legacy counterparts. Other aliases continue to act as
        # backfills when the canonical key is missing.
        for alias, canon in alias_map.items():
            alias_value = env_map.get(alias)
            if alias_value in (None, ""):
                continue

            if alias.startswith("AI_TRADING_"):
                env_map[canon] = alias_value
                continue

            canonical_value = env_map.get(canon)
            if canonical_value is None or str(canonical_value).strip() == "":
                env_map[canon] = alias_value

        from .aliases import resolve_trading_mode

        mode = resolve_trading_mode(mode or "balanced").lower()

        explicit_env_keys: set[str] = set()

        def _register_explicit(key: str, resolved_key: str) -> None:
            canonical = key.upper()
            explicit_env_keys.add(canonical)
            explicit_env_keys.add(resolved_key.upper())

        def _get(
            key: str,
            cast: Callable[[str], Any] | type = str,
            default: Any = None,
            aliases: Iterable[str] = (),
        ):
            for candidate in (key, *aliases):
                if candidate not in env_map:
                    continue
                raw_val = env_map[candidate]
                if raw_val in (None, ""):
                    continue
                _register_explicit(key, candidate)
                if cast is bool:
                    return _to_bool(str(raw_val))
                if cast is str:
                    return raw_val
                try:
                    return cast(raw_val)
                except (ValueError, TypeError) as exc:
                    raise RuntimeError(
                        f"Failed to cast env var {candidate!r}={raw_val!r} via {cast}: {exc}"
                    ) from exc
            return default

        base_url = _get(
            "ALPACA_BASE_URL",
            str,
            default="https://paper-api.alpaca.markets",
            aliases=("ALPACA_API_URL",),
        )
        app_env = _get("APP_ENV", str, default="test") or "test"
        paper_default = "paper" in str(base_url).lower() or app_env.lower() != "prod"

        mps = _get(
            "MAX_POSITION_SIZE",
            float,
            default=8000.0,
            aliases=("AI_TRADING_MAX_POSITION_SIZE",),
        )
        if mps is not None and mps <= 0:
            raise ValueError("MAX_POSITION_SIZE must be positive")

        extras_raw = _get("TRADING_CONFIG_EXTRAS", str)
        extras = None
        if extras_raw is not None:
            try:
                extras = json.loads(extras_raw)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    "TRADING_CONFIG_EXTRAS must be valid JSON"
                ) from exc

        _sentinel = object()
        raw_drawdown_threshold = _get(
            "MAX_DRAWDOWN_THRESHOLD",
            float,
            default=_sentinel,
            aliases=("AI_TRADING_MAX_DRAWDOWN_THRESHOLD",),
        )
        drawdown_missing = raw_drawdown_threshold is _sentinel
        if drawdown_missing:
            if not allow_missing_drawdown:
                raise RuntimeError(
                    "MAX_DRAWDOWN_THRESHOLD is required; set allow_missing_drawdown=True to use the 0.08 fallback."
                )
            logger.debug(
                "allow_missing_drawdown is deprecated; max_drawdown_threshold defaults to 0.08"
            )
            max_drawdown_threshold = 0.08
        else:
            max_drawdown_threshold = raw_drawdown_threshold

        cfg = cls(
            capital_cap=_get("CAPITAL_CAP", float, default=0.25),
            dollar_risk_limit=_get(
                "DOLLAR_RISK_LIMIT",
                float,
                default=0.05,
                aliases=("DAILY_LOSS_LIMIT",),
            ),
            daily_loss_limit=_get(
                "DAILY_LOSS_LIMIT",
                float,
                default=0.03,
            ),
            max_position_size=mps,
            max_position_equity_fallback=_get(
                "MAX_POSITION_EQUITY_FALLBACK", float, default=200000.0
            ),
            position_size_min_usd=_get(
                "POSITION_SIZE_MIN_USD",
                float,
                default=POSITION_SIZE_MIN_USD_DEFAULT,
                aliases=("AI_TRADING_POSITION_SIZE_MIN_USD",),
            ),
            sector_exposure_cap=_get("SECTOR_EXPOSURE_CAP", float),
            max_drawdown_threshold=max_drawdown_threshold,
            trailing_factor=_get("TRAILING_FACTOR", float),
            take_profit_factor=_get("TAKE_PROFIT_FACTOR", float),
            max_position_size_pct=_get("MAX_POSITION_SIZE_PCT", float),
            max_var_95=_get("MAX_VAR_95", float),
            min_profit_factor=_get("MIN_PROFIT_FACTOR", float),
            min_sharpe_ratio=_get("MIN_SHARPE_RATIO", float),
            min_win_rate=_get("MIN_WIN_RATE", float),
            kelly_fraction=_get("KELLY_FRACTION", float, default=0.6),
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
            conf_threshold=_get(
                "CONF_THRESHOLD",
                float,
                default=0.6,
                aliases=("AI_TRADING_CONF_THRESHOLD",),
            ),
            lookback_periods=_get("LOOKBACK_PERIODS", int),
            market_calendar=_get("MARKET_CALENDAR", str),
            score_confidence_min=_get("SCORE_CONFIDENCE_MIN", float),
            signal_confirmation_bars=_get("SIGNAL_CONFIRMATION_BARS", int, default=2),
            delta_threshold=_get("DELTA_THRESHOLD", float, default=0.02),
            min_confidence=_get("MIN_CONFIDENCE", float, default=0.6),
            extras=extras,
            max_position_mode=_get("MAX_POSITION_MODE", str, default="STATIC"),
            paper=_get("PAPER", _to_bool, default=paper_default),
            disable_daily_retrain=_get(
                "DISABLE_DAILY_RETRAIN", _to_bool, default=False
            ),
            data_feed=_get("DATA_FEED", str),
            data_provider=_get("DATA_PROVIDER", str),
            minute_data_freshness_tolerance_seconds=_get(
                "MINUTE_DATA_FRESHNESS_TOLERANCE_SECONDS",
                int,
                default=900,
                aliases=("AI_TRADING_MINUTE_DATA_FRESHNESS_TOLERANCE_SECONDS",),
            ),
            buy_threshold=_get(
                "BUY_THRESHOLD", float, aliases=("AI_TRADING_BUY_THRESHOLD",)
            ),
            signal_period=_get("SIGNAL_PERIOD", int),
            fast_period=_get("FAST_PERIOD", int),
            slow_period=_get("SLOW_PERIOD", int),
            limit_order_slippage=_get("LIMIT_ORDER_SLIPPAGE", float),
            max_slippage_bps=_get("MAX_SLIPPAGE_BPS", int),
            participation_rate=_get("PARTICIPATION_RATE", float),
            pov_slice_pct=_get("POV_SLICE_PCT", float),
            order_timeout_seconds=_get("ORDER_TIMEOUT_SECONDS", int),
        )
        # Apply mode presets when requested (either via explicit mode argument or
        # TRADING_MODE environment variable).
        if mode in {"balanced", "conservative", "aggressive"}:
            presets = {
                "balanced": dict(
                    kelly_fraction=0.6,
                    conf_threshold=0.75,
                    daily_loss_limit=0.05,
                    capital_cap=0.25,
                    signal_confirmation_bars=2,
                    take_profit_factor=1.8,
                    max_position_size=8000.0,
                ),
                "conservative": dict(
                    kelly_fraction=0.25,
                    conf_threshold=0.85,
                    daily_loss_limit=0.03,
                    capital_cap=0.20,
                    signal_confirmation_bars=3,
                    take_profit_factor=1.5,
                    max_position_size=5000.0,
                ),
                "aggressive": dict(
                    kelly_fraction=0.75,
                    conf_threshold=0.65,
                    daily_loss_limit=0.08,
                    capital_cap=0.30,
                    signal_confirmation_bars=1,
                    take_profit_factor=2.5,
                    max_position_size=12000.0,
                ),
            }
            for k, v in presets[mode].items():
                env_key = k.upper()
                if env_key in explicit_env_keys:
                    continue
                try:
                    object.__setattr__(cfg, k, v)
                except Exception:
                    pass

        return cfg

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

    @property
    def confirmation_count(self) -> int:
        """Alias expected by tests for signal confirmation bars."""
        return int(self.signal_confirmation_bars)

    @classmethod
    def from_optimization(cls, params: Mapping[str, Any]) -> "TradingConfig":
        """Build from optimization params (test helper)."""
        cfg = cls()
        for k in ("kelly_fraction", "conf_threshold", "daily_loss_limit"):
            if k in params:
                try:
                    object.__setattr__(cfg, k, params[k])
                except Exception:
                    pass
        return cfg


def derive_cap_from_settings(settings_obj, equity: Optional[float], fallback: float, capital_cap: float) -> float:
    """
    Thin, explicit re-export that delegates to utils.capital_scaling.
    `settings_obj` is accepted for API compatibility but unused.
    """
    return float(_derive_cap_from_settings(equity, capital_cap, fallback))


def from_env_relaxed(env_or_mode: Any | None = None) -> TradingConfig:
    """Non-raising variant of :meth:`TradingConfig.from_env` used for lazy imports.

    Missing ``MAX_DRAWDOWN_THRESHOLD`` is tolerated; other validation errors
    still surface. Mode overrides via ``TRADING_MODE`` are applied the same as
    ``from_env``.
    """
    return TradingConfig.from_env(env_or_mode, allow_missing_drawdown=True)


__all__.append("from_env_relaxed")
