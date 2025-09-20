"""Typed runtime configuration parsed from environment variables.

This module defines a declarative schema for all environment variables used by
the trading bot.  The schema powers a strongly-typed :class:`TradingConfig`
object, provides a single point of validation, and can be rendered into a
human-readable Markdown table.  No configuration is evaluated at import time;
call :func:`get_trading_config` when runtime settings are required.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _strip_inline_comment(value: str) -> str:
    """Remove trailing inline comments introduced with ``#``.

    Inline comments are considered to begin when a ``#`` follows whitespace or
    starts the string (e.g. ``"1.0  # note"``).  Content preceding the comment
    is returned with surrounding whitespace trimmed.  Hash symbols that are part
    of the actual value (e.g. ``"abc#123"``) remain untouched.
    """

    for idx, char in enumerate(value):
        if char == "#" and (idx == 0 or value[idx - 1].isspace()):
            return value[:idx].rstrip()
    return value.rstrip()


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(
        filter(None, (part.strip() for part in _strip_inline_comment(value).split(",")))
    )


def _parse_numeric_sequence(value: str) -> tuple[float, ...]:
    cleaned_value = _strip_inline_comment(value)
    cleaned = [chunk.strip() for chunk in cleaned_value.split(",") if chunk.strip()]
    return tuple(float(chunk) for chunk in cleaned)


CastFn = Callable[[str], Any]


@dataclass(frozen=True)
class ConfigSpec:
    field: str
    env: tuple[str, ...]
    cast: str | CastFn
    default: Any
    description: str
    required: bool = False
    min_value: float | None = None
    max_value: float | None = None
    choices: tuple[Any, ...] | None = None
    deprecated_env: Mapping[str, str] = field(default_factory=dict)
    mask: bool = False


def _cast_value(spec: ConfigSpec, raw: str) -> Any:
    if isinstance(spec.cast, str):
        kind = spec.cast
        if kind == "str":
            return raw
        if kind == "int":
            return int(_strip_inline_comment(raw))
        if kind == "float":
            return float(_strip_inline_comment(raw))
        if kind == "bool":
            return _to_bool(_strip_inline_comment(raw))
        if kind == "tuple[str]":
            return _split_csv(raw)
        if kind == "tuple[float]":
            return _parse_numeric_sequence(_strip_inline_comment(raw))
        if kind == "json":
            return json.loads(raw)
        raise ValueError(f"Unsupported cast kind: {kind}")
    return spec.cast(raw)


def _validate_bounds(spec: ConfigSpec, value: Any) -> Any:
    if isinstance(value, (int, float)):
        if spec.min_value is not None and value < spec.min_value:
            raise ValueError(
                f"{spec.field} must be >= {spec.min_value}, got {value}"
            )
        if spec.max_value is not None and value > spec.max_value:
            raise ValueError(
                f"{spec.field} must be <= {spec.max_value}, got {value}"
            )
    if spec.choices and value not in spec.choices:
        choices = ", ".join(map(str, spec.choices))
        raise ValueError(f"{spec.field} must be one of: {choices}; got {value}")
    return value


SPEC_BY_FIELD: dict[str, ConfigSpec] = {}
SPEC_BY_ENV: dict[str, ConfigSpec] = {}
_DEPRECATION_LOGGED: set[str] = set()


CONFIG_SPECS: tuple[ConfigSpec, ...] = (
    ConfigSpec(
        field="seed",
        env=("SEED",),
        cast="int",
        default=42,
        description="Deterministic seed applied to random number generators.",
        min_value=0,
    ),
    ConfigSpec(
        field="app_env",
        env=("APP_ENV",),
        cast="str",
        default="test",
        description="Deployment environment label (prod/stage/test).",
    ),
    ConfigSpec(
        field="testing",
        env=("TESTING",),
        cast="bool",
        default=False,
        description="Set to true when running under automated tests.",
    ),
    ConfigSpec(
        field="pytest_running",
        env=("PYTEST_RUNNING",),
        cast="bool",
        default=False,
        description="Compatibility flag used by legacy tests to disable network calls.",
    ),
    ConfigSpec(
        field="shadow_mode",
        env=("SHADOW_MODE",),
        cast="bool",
        default=False,
        description="When true the service mirrors production flow without submitting trades.",
    ),
    ConfigSpec(
        field="alpaca_api_key",
        env=("ALPACA_API_KEY",),
        cast="str",
        default=None,
        description="Alpaca API key identifier.",
        mask=True,
    ),
    ConfigSpec(
        field="alpaca_secret_key",
        env=("ALPACA_SECRET_KEY",),
        cast="str",
        default=None,
        description="Alpaca API secret.",
        mask=True,
    ),
    ConfigSpec(
        field="alpaca_oauth_token",
        env=("ALPACA_OAUTH",),
        cast="str",
        default=None,
        description="OAuth token used when authenticating via Alpaca OAuth flow.",
        mask=True,
    ),
    ConfigSpec(
        field="alpaca_base_url",
        env=("ALPACA_API_URL", "ALPACA_BASE_URL"),
        cast="str",
        default="https://paper-api.alpaca.markets",
        description="Alpaca REST endpoint base URL.",
        deprecated_env={"ALPACA_BASE_URL": "Use ALPACA_API_URL instead."},
    ),
    ConfigSpec(
        field="alpaca_allow_sip",
        env=("ALPACA_ALLOW_SIP",),
        cast="bool",
        default=False,
        description="Permit usage of Alpaca SIP data where the account has entitlements.",
    ),
    ConfigSpec(
        field="alpaca_has_sip",
        env=("ALPACA_HAS_SIP",),
        cast="bool",
        default=False,
        description="Indicates that the Alpaca account is SIP entitled.",
    ),
    ConfigSpec(
        field="allow_after_hours",
        env=("ALLOW_AFTER_HOURS",),
        cast="bool",
        default=False,
        description="If true the trading engine will consider after-hours market data.",
    ),
    ConfigSpec(
        field="api_host",
        env=("API_HOST",),
        cast="str",
        default="0.0.0.0",
        description="Bind address for the REST API.",
    ),
    ConfigSpec(
        field="api_port",
        env=("API_PORT", "AI_TRADING_API_PORT"),
        cast="int",
        default=9001,
        description="TCP port used by the REST API.",
        min_value=1,
        max_value=65535,
        deprecated_env={"AI_TRADING_API_PORT": "Use API_PORT instead."},
    ),
    ConfigSpec(
        field="webhook_secret",
        env=("WEBHOOK_SECRET", "AI_TRADING_WEBHOOK_SECRET"),
        cast="str",
        default=None,
        description="Shared secret required to authenticate webhook callbacks.",
        mask=True,
        deprecated_env={"AI_TRADING_WEBHOOK_SECRET": "Use WEBHOOK_SECRET instead."},
    ),
    ConfigSpec(
        field="capital_cap",
        env=("CAPITAL_CAP", "AI_TRADING_CAPITAL_CAP"),
        cast="float",
        default=0.25,
        description="Maximum fraction of account equity allocated to a single position.",
        min_value=0.0,
        max_value=1.0,
        deprecated_env={"AI_TRADING_CAPITAL_CAP": "Use CAPITAL_CAP instead."},
    ),
    ConfigSpec(
        field="dollar_risk_limit",
        env=("DOLLAR_RISK_LIMIT", "AI_TRADING_DOLLAR_RISK_LIMIT"),
        cast="float",
        default=0.05,
        description="Maximum fraction of account equity risked per trade.",
        min_value=0.0,
        max_value=1.0,
        deprecated_env={"AI_TRADING_DOLLAR_RISK_LIMIT": "Use DOLLAR_RISK_LIMIT instead."},
    ),
    ConfigSpec(
        field="daily_loss_limit",
        env=("DAILY_LOSS_LIMIT", "AI_TRADING_DAILY_LOSS_LIMIT"),
        cast="float",
        default=0.03,
        description="Maximum tolerated realised drawdown over a single trading day.",
        min_value=0.0,
        max_value=1.0,
        deprecated_env={"AI_TRADING_DAILY_LOSS_LIMIT": "Use DAILY_LOSS_LIMIT instead."},
    ),
    ConfigSpec(
        field="disaster_dd_limit",
        env=("AI_TRADING_DISASTER_DD_LIMIT", "DISASTER_DD_LIMIT"),
        cast="float",
        default=0.25,
        description="Emergency stop when drawdown exceeds this fraction of equity.",
        min_value=0.0,
        max_value=1.0,
        deprecated_env={"DISASTER_DD_LIMIT": "Use AI_TRADING_DISASTER_DD_LIMIT instead."},
    ),
    ConfigSpec(
        field="max_drawdown_threshold",
        env=("MAX_DRAWDOWN_THRESHOLD", "AI_TRADING_MAX_DRAWDOWN_THRESHOLD"),
        cast="float",
        default=0.08,
        description="Maximum rolling drawdown before trading halts.",
        min_value=0.0,
        max_value=1.0,
        deprecated_env={"AI_TRADING_MAX_DRAWDOWN_THRESHOLD": "Use MAX_DRAWDOWN_THRESHOLD instead."},
    ),
    ConfigSpec(
        field="max_position_size",
        env=("MAX_POSITION_SIZE", "AI_TRADING_MAX_POSITION_SIZE"),
        cast="float",
        default=8000.0,
        description="Absolute maximum position notional in USD.",
        min_value=0.0,
        deprecated_env={"AI_TRADING_MAX_POSITION_SIZE": "Use MAX_POSITION_SIZE instead."},
    ),
    ConfigSpec(
        field="max_position_equity_fallback",
        env=("MAX_POSITION_EQUITY_FALLBACK",),
        cast="float",
        default=200000.0,
        description="Equity fallback used when broker balances are unavailable.",
        min_value=0.0,
    ),
    ConfigSpec(
        field="position_size_min_usd",
        env=("AI_TRADING_POSITION_SIZE_MIN_USD", "POSITION_SIZE_MIN_USD"),
        cast="float",
        default=25.0,
        description="Minimum notional for any individual trade.",
        min_value=0.0,
        deprecated_env={"POSITION_SIZE_MIN_USD": "Use AI_TRADING_POSITION_SIZE_MIN_USD instead."},
    ),
    ConfigSpec(
        field="sector_exposure_cap",
        env=("AI_TRADING_SECTOR_EXPOSURE_CAP", "SECTOR_EXPOSURE_CAP"),
        cast="float",
        default=0.33,
        description="Maximum sector weight as fraction of portfolio equity.",
        min_value=0.0,
        max_value=1.0,
        deprecated_env={"SECTOR_EXPOSURE_CAP": "Use AI_TRADING_SECTOR_EXPOSURE_CAP instead."},
    ),
    ConfigSpec(
        field="take_profit_factor",
        env=("TAKE_PROFIT_FACTOR",),
        cast="float",
        default=None,
        description="Custom take-profit multiplier overriding mode defaults.",
    ),
    ConfigSpec(
        field="trailing_factor",
        env=("TRAILING_FACTOR",),
        cast="float",
        default=None,
        description="Custom trailing stop multiplier overriding mode defaults.",
    ),
    ConfigSpec(
        field="kelly_fraction",
        env=("KELLY_FRACTION", "AI_TRADING_KELLY_FRACTION"),
        cast="float",
        default=None,
        description="Kelly criterion fraction for position sizing (0-1).",
        min_value=0.0,
        max_value=1.0,
        deprecated_env={"AI_TRADING_KELLY_FRACTION": "Use KELLY_FRACTION instead."},
    ),
    ConfigSpec(
        field="kelly_fraction_max",
        env=("KELLY_FRACTION_MAX", "AI_TRADING_KELLY_FRACTION_MAX"),
        cast="float",
        default=0.25,
        description="Upper bound applied to Kelly-derived sizes.",
        min_value=0.0,
        max_value=1.0,
        deprecated_env={"AI_TRADING_KELLY_FRACTION_MAX": "Use KELLY_FRACTION_MAX instead."},
    ),
    ConfigSpec(
        field="min_confidence",
        env=("MIN_CONFIDENCE",),
        cast="float",
        default=0.6,
        description="Minimum model confidence required before entering trades.",
        min_value=0.0,
        max_value=1.0,
    ),
    ConfigSpec(
        field="conf_threshold",
        env=("CONF_THRESHOLD", "AI_TRADING_CONF_THRESHOLD"),
        cast="float",
        default=0.75,
        description="Primary confidence threshold used for scoring trades.",
        min_value=0.0,
        max_value=1.0,
        deprecated_env={"AI_TRADING_CONF_THRESHOLD": "Use CONF_THRESHOLD instead."},
    ),
    ConfigSpec(
        field="min_sample_size",
        env=("MIN_SAMPLE_SIZE", "AI_TRADING_MIN_SAMPLE_SIZE"),
        cast="int",
        default=10,
        description="Rolling window size required for statistical indicators.",
        min_value=1,
        deprecated_env={"AI_TRADING_MIN_SAMPLE_SIZE": "Use MIN_SAMPLE_SIZE instead."},
    ),
    ConfigSpec(
        field="confidence_level",
        env=("CONFIDENCE_LEVEL", "AI_TRADING_CONFIDENCE_LEVEL"),
        cast="float",
        default=0.90,
        description="Confidence level used when computing VAR and value-at-risk thresholds.",
        min_value=0.5,
        max_value=0.999,
        deprecated_env={"AI_TRADING_CONFIDENCE_LEVEL": "Use CONFIDENCE_LEVEL instead."},
    ),
    ConfigSpec(
        field="signal_confirmation_bars",
        env=("SIGNAL_CONFIRMATION_BARS",),
        cast="int",
        default=2,
        description="Number of confirming bars required before executing signals.",
        min_value=1,
    ),
    ConfigSpec(
        field="delta_threshold",
        env=("DELTA_THRESHOLD",),
        cast="float",
        default=0.02,
        description="Minimum delta between entry price and confirmation price to trigger trades.",
        min_value=0.0,
    ),
    ConfigSpec(
        field="allow_after_hours_volume_threshold",
        env=("VOLUME_THRESHOLD", "AI_TRADING_VOLUME_THRESHOLD"),
        cast="float",
        default=0.0,
        description="Override minimum average volume requirement for symbols.",
        min_value=0.0,
        deprecated_env={"VOLUME_THRESHOLD": "Use AI_TRADING_VOLUME_THRESHOLD instead."},
    ),
    ConfigSpec(
        field="minute_data_freshness_tolerance_seconds",
        env=(
            "MINUTE_DATA_FRESHNESS_TOLERANCE_SECONDS",
            "AI_TRADING_MINUTE_DATA_FRESHNESS_TOLERANCE_SECONDS",
        ),
        cast="int",
        default=900,
        description="Maximum age in seconds for minute bars before data is considered stale.",
        min_value=1,
        deprecated_env={
            "AI_TRADING_MINUTE_DATA_FRESHNESS_TOLERANCE_SECONDS": "Use MINUTE_DATA_FRESHNESS_TOLERANCE_SECONDS instead."
        },
    ),
    ConfigSpec(
        field="alpaca_data_feed",
        env=("ALPACA_DATA_FEED", "DATA_FEED"),
        cast="str",
        default="iex",
        description="Preferred Alpaca data feed (iex or sip).",
        choices=("iex", "sip"),
        deprecated_env={"DATA_FEED": "Use ALPACA_DATA_FEED instead."},
    ),
    ConfigSpec(
        field="alpaca_feed_failover",
        env=("ALPACA_FEED_FAILOVER",),
        cast="tuple[str]",
        default=("sip",),
        description="Order in which Alpaca feeds are attempted after the primary fails.",
    ),
    ConfigSpec(
        field="alpaca_empty_to_backup",
        env=("ALPACA_EMPTY_TO_BACKUP",),
        cast="bool",
        default=True,
        description="Route empty Alpaca payloads to the configured backup provider.",
    ),
    ConfigSpec(
        field="data_provider_priority",
        env=("DATA_PROVIDER_PRIORITY",),
        cast="tuple[str]",
        default=("alpaca_iex", "alpaca_sip", "yahoo"),
        description="Global fetch priority order for data providers.",
    ),
    ConfigSpec(
        field="max_data_fallbacks",
        env=("MAX_DATA_FALLBACKS",),
        cast="int",
        default=2,
        description="Maximum number of provider fallbacks attempted per fetch.",
        min_value=0,
    ),
    ConfigSpec(
        field="data_provider_backoff_factor",
        env=("DATA_PROVIDER_BACKOFF_FACTOR",),
        cast="float",
        default=2.0,
        description="Backoff multiplier applied when providers repeatedly fail.",
        min_value=1.0,
    ),
    ConfigSpec(
        field="data_provider_max_cooldown",
        env=("DATA_PROVIDER_MAX_COOLDOWN",),
        cast="int",
        default=3600,
        description="Upper bound (seconds) for provider disablement cooldowns.",
        min_value=60,
    ),
    ConfigSpec(
        field="http_timeout_seconds",
        env=("AI_TRADING_HTTP_TIMEOUT",),
        cast="float",
        default=5.0,
        description="Default HTTP connect/read timeout used by outbound requests.",
        min_value=0.1,
    ),
    ConfigSpec(
        field="host_concurrency_limit",
        env=("AI_TRADING_HOST_LIMIT",),
        cast="int",
        default=8,
        description="Maximum concurrent requests issued against a single host.",
        min_value=1,
    ),
    ConfigSpec(
        field="max_empty_retries",
        env=("MAX_EMPTY_RETRIES",),
        cast="int",
        default=10,
        description="Number of consecutive empty responses tolerated before aborting a fetch.",
        min_value=1,
    ),
    ConfigSpec(
        field="symbol_process_budget_seconds",
        env=("SYMBOL_PROCESS_BUDGET",),
        cast="float",
        default=300.0,
        description="Soft budget (seconds) allotted for processing the full symbol universe.",
        min_value=0.0,
    ),
    ConfigSpec(
        field="cycle_budget_fraction",
        env=("CYCLE_BUDGET_FRACTION",),
        cast="float",
        default=0.9,
        description="Fraction of the schedule interval reserved for work before yielding.",
        min_value=0.0,
        max_value=1.0,
    ),
    ConfigSpec(
        field="cycle_compute_budget_factor",
        env=("CYCLE_COMPUTE_BUDGET",),
        cast="float",
        default=None,
        description="Optional override for compute budget fraction. When unset the fraction is derived from `cycle_budget_fraction`.",
    ),
    ConfigSpec(
        field="signal_hold_epsilon",
        env=("AI_TRADING_SIGNAL_HOLD_EPS",),
        cast="float",
        default=0.01,
        description="Tolerance used when deciding whether to hold positions around target weights.",
        min_value=0.0,
    ),
    ConfigSpec(
        field="strict_model_loading",
        env=("AI_TRADING_STRICT_MODEL_LOADING",),
        cast="bool",
        default=False,
        description="When true, missing ML models raise immediately instead of falling back to neutral mode.",
    ),
    ConfigSpec(
        field="warn_if_model_missing",
        env=("AI_TRADING_WARN_IF_MODEL_MISSING",),
        cast="bool",
        default=False,
        description="Controls whether missing ML models emit warnings during startup checks.",
    ),
    ConfigSpec(
        field="default_model_url",
        env=("DEFAULT_MODEL_URL",),
        cast="str",
        default=None,
        description="URL used to download the default ML model when not present locally.",
    ),
    ConfigSpec(
        field="max_symbols_per_cycle",
        env=("MAX_SYMBOLS_PER_CYCLE",),
        cast="int",
        default=500,
        description="Hard cap on number of symbols processed during a trading cycle.",
        min_value=1,
    ),
    ConfigSpec(
        field="health_tick_seconds",
        env=("HEALTH_TICK_SECONDS",),
        cast="float",
        default=300.0,
        description="Interval between background health checks.",
        min_value=30.0,
    ),
    ConfigSpec(
        field="hard_stop_cooldown_min",
        env=("HARD_STOP_COOLDOWN_MIN",),
        cast="int",
        default=15,
        description="Cooldown window (minutes) enforced after a hard stop triggers.",
        min_value=0,
    ),
    ConfigSpec(
        field="force_continue_on_exposure",
        env=("FORCE_CONTINUE_ON_EXPOSURE",),
        cast="bool",
        default=False,
        description="If true exposure checks log and continue instead of aborting the cycle.",
    ),
    ConfigSpec(
        field="log_level_yfinance",
        env=("LOG_LEVEL_YFINANCE",),
        cast="str",
        default="WARNING",
        description="Log verbosity applied to the yfinance library.",
    ),
    ConfigSpec(
        field="log_quiet_libraries",
        env=("LOG_QUIET_LIBRARIES",),
        cast="tuple[str]",
        default=(),
        description="Comma-separated list of libraries whose logs should be silenced.",
    ),
    ConfigSpec(
        field="max_slippage_bps",
        env=("MAX_SLIPPAGE_BPS",),
        cast="int",
        default=50,
        description="Maximum tolerated slippage expressed in basis points.",
        min_value=0,
    ),
    ConfigSpec(
        field="slippage_limit_tolerance_bps",
        env=("SLIPPAGE_LIMIT_TOLERANCE_BPS",),
        cast="float",
        default=25.0,
        description="Additional slippage headroom permitted before positions are trimmed.",
        min_value=0.0,
    ),
    ConfigSpec(
        field="sentiment_api_key",
        env=("SENTIMENT_API_KEY",),
        cast="str",
        default=None,
        description="Primary sentiment provider API key.",
        mask=True,
    ),
    ConfigSpec(
        field="sentiment_api_url",
        env=("SENTIMENT_API_URL",),
        cast="str",
        default="https://newsapi.org/v2/everything",
        description="Primary sentiment provider API endpoint.",
    ),
    ConfigSpec(
        field="alternative_sentiment_api_key",
        env=("ALTERNATIVE_SENTIMENT_API_KEY",),
        cast="str",
        default=None,
        description="Fallback sentiment provider API key.",
        mask=True,
    ),
    ConfigSpec(
        field="alternative_sentiment_api_url",
        env=("ALTERNATIVE_SENTIMENT_API_URL",),
        cast="str",
        default=None,
        description="Fallback sentiment API endpoint.",
    ),
    ConfigSpec(
        field="sentiment_retry_max",
        env=("SENTIMENT_MAX_RETRIES",),
        cast="int",
        default=5,
        description="Maximum retry attempts for sentiment fetches.",
        min_value=0,
    ),
    ConfigSpec(
        field="sentiment_backoff_base",
        env=("SENTIMENT_BACKOFF_BASE",),
        cast="float",
        default=5.0,
        description="Initial delay (seconds) used when backing off sentiment retries.",
        min_value=0.0,
    ),
    ConfigSpec(
        field="sentiment_backoff_strategy",
        env=("SENTIMENT_BACKOFF_STRATEGY",),
        cast="str",
        default="exponential",
        description="Algorithm applied to sentiment retry backoff (linear/exponential).",
    ),
    ConfigSpec(
        field="enable_finnhub",
        env=("ENABLE_FINNHUB",),
        cast="bool",
        default=True,
        description="Toggle Finnhub integrations without removing API keys.",
    ),
    ConfigSpec(
        field="finnhub_api_key",
        env=("FINNHUB_API_KEY",),
        cast="str",
        default=None,
        description="Finnhub API key for supplemental market data.",
        mask=True,
    ),
    ConfigSpec(
        field="timeframe",
        env=("TIMEFRAME",),
        cast="str",
        default="1Min",
        description="Default Alpaca timeframe used when one is not specified explicitly.",
    ),
)


for spec in CONFIG_SPECS:
    SPEC_BY_FIELD[spec.field] = spec
    for key in spec.env:
        SPEC_BY_ENV[key.upper()] = spec
    for legacy in spec.deprecated_env:
        SPEC_BY_ENV[legacy.upper()] = spec


class TradingConfig:
    """Immutable container mapping config specifications to resolved values."""

    __slots__ = ("_values",)

    def __init__(self, **values: Any) -> None:
        object.__setattr__(self, "_values", values)

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - attribute passthrough
        try:
            return self._values[item]
        except KeyError as exc:  # noqa: F401
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:  # pragma: no cover - immutability
        raise AttributeError("TradingConfig is immutable")

    def to_dict(self) -> dict[str, Any]:
        return dict(self._values)

    def snapshot_sanitized(self) -> dict[str, Any]:
        data = {
            "risk": {
                "capital_cap": self.capital_cap,
                "dollar_risk_limit": self.dollar_risk_limit,
                "daily_loss_limit": self.daily_loss_limit,
                "max_drawdown_threshold": self.max_drawdown_threshold,
            },
            "data": {
                "feed": getattr(self, "alpaca_data_feed", None),
                "provider": getattr(self, "data_provider", None),
            },
            "auth": {
                "alpaca_api_key": "***" if getattr(self, "alpaca_api_key", None) else "",
                "alpaca_secret_key": "***" if getattr(self, "alpaca_secret_key", None) else "",
            },
        }
        return data

    @property
    def symbol_process_budget_ms(self) -> int:
        return int(max(0.0, self._values["symbol_process_budget_seconds"]) * 1000)

    @classmethod
    def from_env(
        cls, env_overrides: Mapping[str, Any] | None = None
    ) -> "TradingConfig":
        env_map = _env_snapshot(env_overrides)
        values: dict[str, Any] = {}
        for spec in CONFIG_SPECS:
            values[spec.field] = _build_value(spec, env_map)

        if values["cycle_compute_budget_factor"] is None:
            values["cycle_compute_budget_factor"] = values["cycle_budget_fraction"]

        # Derived convenience fields expected by legacy callers.
        values.setdefault("data_provider", values.get("data_provider_priority", (None,))[0])
        values.setdefault("paper", _infer_paper_mode(values))
        values.setdefault("max_position_mode", values.get("max_position_mode", "STATIC"))

        return cls(**values)


def _build_value(spec: ConfigSpec, env_map: Mapping[str, str]) -> Any:
    canonical_keys: Sequence[str] = spec.env
    found_key: str | None = None
    raw_value: str | None = None
    for candidate in canonical_keys:
        if candidate in env_map and env_map[candidate] not in (None, ""):
            found_key = candidate
            raw_value = env_map[candidate]
            break
    if raw_value is None:
        for alias, message in spec.deprecated_env.items():
            if alias in env_map and env_map[alias] not in (None, ""):
                found_key = alias
                raw_value = env_map[alias]
                if alias not in _DEPRECATION_LOGGED:
                    logger.warning(
                        "CONFIG_ENV_DEPRECATED",
                        extra={"deprecated": alias, "replacement": spec.env[0], "note": message},
                    )
                    _DEPRECATION_LOGGED.add(alias)
                break
    if raw_value is None:
        if spec.required:
            raise RuntimeError(
                f"Environment variable required for '{spec.field}' not set: one of {spec.env}"
            )
        return spec.default
    try:
        value = _cast_value(spec, raw_value)
    except Exception as exc:  # noqa: BLE001 - convert to user message
        raise RuntimeError(
            f"Failed to parse environment variable {found_key}={raw_value!r}: {exc}"
        ) from exc
    return _validate_bounds(spec, value)


def _env_snapshot(overrides: Mapping[str, Any] | None = None) -> dict[str, str]:
    snap = {k: v for k, v in os.environ.items() if isinstance(v, str)}
    if overrides:
        snap.update({k.upper(): str(v) for k, v in overrides.items()})
    return snap


@lru_cache(maxsize=1)
def get_trading_config() -> TradingConfig:
    return TradingConfig.from_env()


def reload_trading_config(env_overrides: Mapping[str, Any] | None = None) -> TradingConfig:
    get_trading_config.cache_clear()  # type: ignore[attr-defined]
    return TradingConfig.from_env(env_overrides)


def generate_config_schema() -> str:
    """Render a Markdown table describing all configuration fields."""

    header = "| Field | Env Vars | Type | Default | Description |\n| --- | --- | --- | --- | --- |"
    lines = [header]
    for spec in CONFIG_SPECS:
        default = "***" if spec.mask and spec.default else str(spec.default)
        env_names = ", ".join(spec.env)
        type_name = spec.cast if isinstance(spec.cast, str) else spec.cast.__name__
        description = spec.description
        if spec.choices:
            description += f" Choices: {', '.join(map(str, spec.choices))}."
        bounds: list[str] = []
        if spec.min_value is not None:
            bounds.append(f">= {spec.min_value}")
        if spec.max_value is not None:
            bounds.append(f"<= {spec.max_value}")
        if bounds:
            description += f" Bounds: {'; '.join(bounds)}."
        if spec.deprecated_env:
            deprecated_keys = ", ".join(spec.deprecated_env.keys())
            description += f" Deprecated aliases: {deprecated_keys}."
        lines.append(
            f"| {spec.field} | {env_names} | {type_name} | {default} | {description} |"
        )
    return "\n".join(lines)


__all__ = [
    "TradingConfig",
    "CONFIG_SPECS",
    "get_trading_config",
    "reload_trading_config",
    "generate_config_schema",
]
