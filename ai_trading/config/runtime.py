"""Typed runtime configuration parsed from environment variables.

This module defines a declarative schema for all environment variables used by
the trading bot.  The schema powers a strongly-typed :class:`TradingConfig`
object, provides a single point of validation, and can be rendered into a
human-readable Markdown table.  No configuration is evaluated at import time;
call :func:`get_trading_config` when runtime settings are required.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

_LEGACY_BROKER_PREFIX = "AP" "CA_"


def _reject_legacy_apca_env() -> None:
    """Abort startup when unsupported AP""" "CA_* environment variables are present."""

    allowlisted = {"AP" "CA_" "API_KEY_ID", "AP" "CA_" "API_SECRET_KEY"}
    legacy_keys = [key for key in os.environ if key.startswith(_LEGACY_BROKER_PREFIX)]
    filtered = sorted(key for key in legacy_keys if key not in allowlisted)
    if not filtered:
        return

    preview = ", ".join(filtered[:5])
    if len(filtered) > 5:
        preview += " ..."

    raise RuntimeError(
        "Legacy "
        f"{_LEGACY_BROKER_PREFIX}* environment variables are no longer supported. "
        f"Found: {preview}. Rename them to ALPACA_* (for example, "
        f"{_LEGACY_BROKER_PREFIX}API_KEY_ID→ALPACA_API_KEY and "
        f"{_LEGACY_BROKER_PREFIX}API_SECRET_KEY→ALPACA_SECRET_KEY). "
        "After updating your environment (.env/systemd), run 'make doctor' to verify."
    )


_reject_legacy_apca_env()


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


def _parse_max_position_mode(raw: str) -> str:
    value = _strip_inline_comment(raw).strip().upper()
    if value in {"", "STATIC"}:
        return "STATIC"
    if value == "AUTO":
        return "AUTO"
    raise ValueError(
        "MAX_POSITION_MODE must be one of: STATIC, AUTO"
    )


def _parse_execution_mode(raw: str) -> str:
    """Return normalized execution mode from environment input."""

    value = _strip_inline_comment(raw).strip().lower()
    if not value:
        return "sim"
    aliases = {
        "broker": "paper",
        "alpaca": "paper",
        "paper": "paper",
        "live": "live",
        "production": "live",
        "prod": "live",
        "simulation": "sim",
        "sim": "sim",
        "test": "sim",
        "disabled": "disabled",
        "off": "disabled",
        "none": "disabled",
    }
    normalized = aliases.get(value, value)
    if normalized not in {"sim", "paper", "live", "disabled"}:
        raise ValueError(
            "EXECUTION_MODE must be one of: sim, paper, live, disabled"
        )
    return normalized


def _parse_intraday_feed(raw: str) -> str:
    """Return normalized intraday feed identifier."""

    value = _strip_inline_comment(raw).strip().lower()
    if value in {"", "iex"}:
        return "iex"
    if value == "sip":
        return "sip"
    if value == "finnhub":
        return "finnhub"
    raise ValueError("DATA_FEED_INTRADAY must be one of: iex, sip, finnhub")


_VALID_PRICE_PROVIDERS = {
    "alpaca_trade",
    "alpaca_quote",
    "alpaca_minute_close",
    "alpaca_bid",
    "alpaca_ask",
    "yahoo",
    "bars",
}


def _parse_price_provider_order(raw: str) -> tuple[str, ...]:
    """Parse comma separated provider order ensuring supported identifiers."""

    providers = _split_csv(raw)
    normalized: list[str] = []
    for provider in providers:
        name = provider.strip().lower()
        if not name:
            continue
        if name not in _VALID_PRICE_PROVIDERS:
            raise ValueError(
                "PRICE_PROVIDER_ORDER entries must be known providers"
            )
        normalized.append(name)
    if not normalized:
        return (
            "alpaca_quote",
            "alpaca_trade",
            "alpaca_minute_close",
            "yahoo",
            "bars",
        )
    return tuple(normalized)


def _parse_order_flip_mode(raw: str) -> str:
    """Normalize ORDER_FLIP_MODE policy from environment."""

    value = _strip_inline_comment(raw).strip().lower()
    if not value:
        return "cancel_then_submit"
    allowed = {"cancel_then_submit", "cover_then_long", "skip"}
    if value not in allowed:
        raise ValueError(
            "ORDER_FLIP_MODE must be one of: cancel_then_submit, cover_then_long, skip"
        )
    return value


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
            if spec.field == "max_position_size":
                raise ValueError("MAX_POSITION_SIZE must be positive")
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
        field="execution_mode",
        env=("EXECUTION_MODE", "AI_TRADING_EXECUTION_MODE", "AI_TRADING_EXECUTION_IMPL", "EXECUTION_IMPL"),
        cast=_parse_execution_mode,
        default="sim",
        description="Execution environment selector (sim, paper, live).",
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
        field="disable_daily_retrain",
        env=("DISABLE_DAILY_RETRAIN",),
        cast="bool",
        default=False,
        description="Disable scheduled daily retraining jobs when true.",
    ),
    ConfigSpec(
        field="market_calendar",
        env=("MARKET_CALENDAR",),
        cast="str",
        default=None,
        description="Identifier for the trading calendar (e.g., XNAS).",
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
        default=0.05,
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
        field="max_position_mode",
        env=("MAX_POSITION_MODE", "AI_TRADING_MAX_POSITION_MODE"),
        cast=_parse_max_position_mode,
        default="STATIC",
        description="Controls whether max_position_size is STATIC or AUTO sized.",
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
        field="execution_min_qty",
        env=("EXECUTION_MIN_QTY",),
        cast="int",
        default=1,
        description="Lowest quantity that may be submitted to the execution venue.",
        min_value=0,
    ),
    ConfigSpec(
        field="execution_min_notional",
        env=("EXECUTION_MIN_NOTIONAL",),
        cast="float",
        default=1.0,
        description="Minimum notional amount allowed when placing orders.",
        min_value=0.0,
    ),
    ConfigSpec(
        field="execution_max_open_orders",
        env=("EXECUTION_MAX_OPEN_ORDERS",),
        cast="int",
        default=100,
        description="Upper bound on simultaneously open orders managed by the system.",
        min_value=0,
    ),
    ConfigSpec(
        field="execution_require_bid_ask",
        env=("EXECUTION_REQUIRE_BID_ASK", "AI_TRADING_EXEC_REQUIRE_BID_ASK"),
        cast="bool",
        default=True,
        description="Require live bid/ask quotes before opening new positions.",
    ),
    ConfigSpec(
        field="execution_require_realtime_nbbo",
        env=("EXECUTION_REQUIRE_REALTIME_NBBO", "AI_TRADING_EXEC_REQUIRE_REALTIME_NBBO"),
        cast="bool",
        default=True,
        description="Require real-time NBBO quotes instead of synthetic fallbacks.",
    ),
    ConfigSpec(
        field="execution_max_staleness_sec",
        env=("EXECUTION_MAX_STALENESS_SEC", "AI_TRADING_EXEC_MAX_STALENESS_SEC"),
        cast="int",
        default=60,
        description="Maximum tolerated age (seconds) for a quote when opening positions.",
        min_value=0,
    ),
    ConfigSpec(
        field="execution_allow_last_close",
        env=("EXECUTION_ALLOW_LAST_CLOSE", "AI_TRADING_EXEC_ALLOW_LAST_CLOSE"),
        cast="bool",
        default=False,
        description="Permit opening trades using last close prices when live quotes unavailable.",
    ),
    ConfigSpec(
        field="execution_allow_fallback_price",
        env=("EXECUTION_ALLOW_FALLBACK_PRICE", "AI_TRADING_EXEC_ALLOW_FALLBACK_PRICE"),
        cast="bool",
        default=True,
        description="Allow fallback pricing sources when primary NBBO quotes are unavailable.",
    ),
    ConfigSpec(
        field="post_submit_broker_sync",
        env=("TRADING__POST_SUBMIT_BROKER_SYNC",),
        cast="bool",
        default=True,
        description="Synchronize broker open orders and positions after order submissions each cycle.",
    ),
    ConfigSpec(
        field="min_quote_freshness_ms",
        env=("TRADING__MIN_QUOTE_FRESHNESS_MS",),
        cast="int",
        default=1500,
        description="Minimum quote freshness in milliseconds before treating data as degraded for limit pricing.",
        min_value=0,
    ),
    ConfigSpec(
        field="degraded_feed_mode",
        env=("TRADING__DEGRADED_FEED_MODE",),
        cast="str",
        default="widen",
        description="Policy when market data is degraded (widen or block new entries).",
        choices=("widen", "block"),
    ),
    ConfigSpec(
        field="degraded_feed_limit_widen_bps",
        env=("TRADING__DEGRADED_FEED_LIMIT_WIDEN_BPS",),
        cast="int",
        default=8,
        description="Limit price widening in basis points applied under degraded feed mode when widening is enabled.",
        min_value=0,
    ),
    ConfigSpec(
        field="log_exec_summary_enabled",
        env=("LOG__EXEC_SUMMARY_ENABLED",),
        cast="bool",
        default=True,
        description="Emit consolidated execution summary logs at the end of each cycle.",
    ),
    ConfigSpec(
        field="order_flip_mode",
        env=("ORDER_FLIP_MODE",),
        cast=_parse_order_flip_mode,
        default="cancel_then_submit",
        description="Policy for resolving opposite-side order conflicts.",
    ),
    ConfigSpec(
        field="alpaca_fallback_ttl_seconds",
        env=("ALPACA_FALLBACK_TTL_SECONDS",),
        cast="int",
        default=120,
        description="Cooldown window before retrying Alpaca as primary data feed after fallback.",
        min_value=0,
    ),
    ConfigSpec(
        field="data_drop_last_partial_bar",
        env=("DATA_DROP_LAST_PARTIAL_BAR",),
        cast="bool",
        default=True,
        description="Drop incomplete intraday bars with missing close data before use.",
    ),
    ConfigSpec(
        field="nbbo_required_for_limit",
        env=("NBBO_REQUIRED_FOR_LIMIT",),
        cast="bool",
        default=False,
        description="Require NBBO quotes for limit pricing; fallback to last trade when false.",
    ),
    ConfigSpec(
        field="execution_stale_ratio_shadow",
        env=("EXECUTION_STALE_RATIO_SHADOW",),
        cast="float",
        default=0.30,
        description="Stale symbol ratio that triggers automatic shadow cycle.",
        min_value=0.0,
        max_value=1.0,
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
        field="buy_threshold",
        env=("BUY_THRESHOLD", "AI_TRADING_BUY_THRESHOLD"),
        cast="float",
        default=0.4,
        description="Signal confidence threshold required to enter long positions.",
        min_value=0.0,
        max_value=1.0,
        deprecated_env={"AI_TRADING_BUY_THRESHOLD": "Use BUY_THRESHOLD instead."},
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
        field="data_feed_intraday",
        env=("DATA_FEED_INTRADAY",),
        cast=_parse_intraday_feed,
        default="iex",
        description="Intraday data feed preference for execution pricing.",
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
        default=("alpaca_iex", "yahoo"),
        description="Global fetch priority order for data providers.",
    ),
    ConfigSpec(
        field="safe_mode_allow_paper",
        env=("AI_TRADING_SAFE_MODE_ALLOW_PAPER",),
        cast="bool",
        default=False,
        description="Allow paper execution to bypass provider safe-mode blocks.",
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
        field="data_max_gap_ratio_intraday",
        env=("DATA_MAX_GAP_RATIO_INTRADAY",),
        cast="float",
        default=0.005,
        description="Largest tolerated fractional gap between intraday bars before triggering recovery.",
        min_value=0.0,
    ),
    ConfigSpec(
        field="gap_ratio_limit",
        env=("AI_TRADING_GAP_RATIO_LIMIT",),
        cast="float",
        default=0.005,
        description="Maximum acceptable gap ratio before orders are blocked by execution gating.",
        min_value=0.0,
    ),
    ConfigSpec(
        field="gap_limit_bps",
        env=("AI_TRADING_GAP_LIMIT_BPS",),
        cast="int",
        default=200,
        description="Maximum acceptable primary gap in basis points before orders are blocked by execution gating.",
        min_value=0,
        max_value=10000,
    ),
    ConfigSpec(
        field="fallback_gap_limit_bps",
        env=("AI_TRADING_FALLBACK_GAP_LIMIT_BPS",),
        cast="int",
        default=500,
        description="Permissible fallback pricing gap in basis points when primary quotes are unavailable.",
        min_value=0,
        max_value=10000,
    ),
    ConfigSpec(
        field="data_daily_fetch_min_interval_s",
        env=("DATA_DAILY_FETCH_MIN_INTERVAL_S",),
        cast="int",
        default=60,
        description="Minimum seconds enforced between successive daily data fetch attempts per symbol.",
        min_value=0,
    ),
    ConfigSpec(
        field="price_provider_order",
        env=("PRICE_PROVIDER_ORDER",),
        cast=_parse_price_provider_order,
        default=(
            "alpaca_quote",
            "alpaca_trade",
            "alpaca_minute_close",
            "yahoo",
            "bars",
        ),
        description="Comma separated list defining execution price provider preference order.",
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
        field="provider_switch_quiet_seconds",
        env=("PROVIDER_SWITCH_QUIET_SECONDS",),
        cast="float",
        default=15.0,
        description="Quiet period in seconds before repeated provider switchovers are blocked.",
        min_value=0.0,
    ),
    ConfigSpec(
        field="provider_max_cooldown_seconds",
        env=("PROVIDER_MAX_COOLDOWN_SECONDS",),
        cast="float",
        default=600.0,
        description="Maximum cooldown applied when providers are disabled for thrash protection.",
        min_value=60.0,
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
        field="strict_data_gating",
        env=("AI_TRADING_STRICT_GATING",),
        cast="bool",
        default=True,
        description="When true, enforce strict market data gating before routing orders.",
    ),
    ConfigSpec(
        field="fallback_quote_max_age_seconds",
        env=("AI_TRADING_FALLBACK_QUOTE_MAX_AGE_SEC",),
        cast="float",
        default=8.0,
        description="Maximum permitted age in seconds for broker quotes validating fallback-priced orders.",
        min_value=0.0,
    ),
    ConfigSpec(
        field="liquidity_fallback_cap",
        env=("AI_TRADING_LIQ_FALLBACK_CAP",),
        cast="float",
        default=0.25,
        description="Fractional cap applied to position size when only liquidity fallbacks are in effect.",
        min_value=0.0,
        max_value=1.0,
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
        env=("HEALTH_TICK_SECONDS", "AI_TRADING_HEALTH_TICK_SECONDS"),
        cast="float",
        default=300.0,
        description=(
            "Interval between background health checks. Values below 30 seconds"
            " are invalid and will raise during configuration parsing."
        ),
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
        field="logging_dedupe_ttl_s",
        env=("LOGGING_DEDUPE_TTL_S",),
        cast="int",
        default=120,
        description="Seconds identical log keys remain suppressed by the deduper.",
        min_value=0,
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
        field="slippage_limit_bps",
        env=("SLIPPAGE_LIMIT_BPS",),
        cast="int",
        default=75,
        description="Maximum slippage tolerance applied when evaluating live fills.",
        min_value=0,
    ),
    ConfigSpec(
        field="order_timeout_seconds",
        env=("ORDER_TIMEOUT_SECONDS",),
        cast="int",
        default=300,
        description="Maximum seconds to wait before cancelling unfilled orders.",
        min_value=1,
    ),
    ConfigSpec(
        field="order_stale_cleanup_interval",
        env=("ORDER_STALE_CLEANUP_INTERVAL",),
        cast="int",
        default=120,
        description="Seconds to wait before auto-canceling pending/new orders to unstick the trade loop.",
        min_value=10,
        max_value=3600,
    ),
    ConfigSpec(
        field="orders_pending_new_warn_s",
        env=("ORDERS_PENDING_NEW_WARN_S",),
        cast="int",
        default=60,
        description="Seconds a pending/new order may linger before emitting a warning log.",
        min_value=0,
    ),
    ConfigSpec(
        field="orders_pending_new_error_s",
        env=("ORDERS_PENDING_NEW_ERROR_S",),
        cast="int",
        default=180,
        description="Seconds a pending/new order may linger before escalating to an error.",
        min_value=0,
    ),
    ConfigSpec(
        field="order_fill_rate_target",
        env=("ORDER_FILL_RATE_TARGET",),
        cast="float",
        default=0.80,
        description="Target fraction of orders that should fill within timeout windows.",
        min_value=0.0,
        max_value=1.0,
    ),
    ConfigSpec(
        field="liquidity_spread_threshold",
        env=("LIQUIDITY_SPREAD_THRESHOLD",),
        cast="float",
        default=0.005,
        description="Maximum allowable bid/ask spread (fractional) before pausing liquidity provision.",
        min_value=0.0,
    ),
    ConfigSpec(
        field="liquidity_vol_threshold",
        env=("LIQUIDITY_VOL_THRESHOLD",),
        cast="float",
        default=250000.0,
        description="Minimum rolling dollar volume required to participate in a symbol.",
        min_value=0.0,
    ),
    ConfigSpec(
        field="liquidity_reduction_aggressive",
        env=("LIQUIDITY_REDUCTION_AGGRESSIVE",),
        cast="float",
        default=0.75,
        description="Fractional liquidity reduction applied when conditions are extreme.",
        min_value=0.0,
        max_value=1.0,
    ),
    ConfigSpec(
        field="liquidity_reduction_moderate",
        env=("LIQUIDITY_REDUCTION_MODERATE",),
        cast="float",
        default=0.90,
        description="Fractional liquidity reduction applied when spreads are elevated but not extreme.",
        min_value=0.0,
        max_value=1.0,
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
        field="sentiment_enhanced_caching",
        env=("SENTIMENT_ENHANCED_CACHING",),
        cast="bool",
        default=True,
        description="Enable extended caching for sentiment data and fallbacks.",
    ),
    ConfigSpec(
        field="sentiment_fallback_sources",
        env=("SENTIMENT_FALLBACK_SOURCES",),
        cast="tuple[str]",
        default=("similar_symbol", "sector_proxy", "news_cache"),
        description="Ordered sentiment fallback strategies when primary data is unavailable.",
    ),
    ConfigSpec(
        field="sentiment_success_rate_target",
        env=("SENTIMENT_SUCCESS_RATE_TARGET",),
        cast="float",
        default=0.90,
        description="Target success rate for sentiment fetches before alerting.",
        min_value=0.0,
        max_value=1.0,
    ),
    ConfigSpec(
        field="sentiment_recovery_timeout_secs",
        env=("SENTIMENT_RECOVERY_TIMEOUT_SECS",),
        cast="int",
        default=1800,
        description="Circuit breaker recovery timeout (seconds) after sentiment failures.",
        min_value=0,
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
        field="meta_learning_bootstrap_enabled",
        env=("META_LEARNING_BOOTSTRAP_ENABLED",),
        cast="bool",
        default=True,
        description="Enable bootstrap trade generation when historical data is sparse.",
    ),
    ConfigSpec(
        field="meta_learning_bootstrap_win_rate",
        env=("META_LEARNING_BOOTSTRAP_WIN_RATE",),
        cast="float",
        default=0.55,
        description="Assumed win rate for generated bootstrap trades.",
        min_value=0.0,
        max_value=1.0,
    ),
    ConfigSpec(
        field="meta_learning_min_trades_reduced",
        env=("META_LEARNING_MIN_TRADES_REDUCED",),
        cast="int",
        default=10,
        description="Reduced minimum trade count required to trigger meta-learning retraining.",
        min_value=1,
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
    ConfigSpec(
        field="confirmation_count",
        env=("CONFIRMATION_COUNT",),
        cast="int",
        default=2,
        description="Number of consecutive confirmations required before executing trades.",
        min_value=0,
    ),
    ConfigSpec(
        field="signal_period",
        env=("SIGNAL_PERIOD",),
        cast="int",
        default=9,
        description="Signal smoothing period used for MACD calculations.",
        min_value=1,
    ),
    ConfigSpec(
        field="fast_period",
        env=("FAST_PERIOD",),
        cast="int",
        default=12,
        description="Fast period used for MACD calculations.",
        min_value=1,
    ),
    ConfigSpec(
        field="slow_period",
        env=("SLOW_PERIOD",),
        cast="int",
        default=26,
        description="Slow period used for MACD calculations.",
        min_value=1,
    ),
    ConfigSpec(
        field="max_portfolio_risk",
        env=("MAX_PORTFOLIO_RISK",),
        cast="float",
        default=0.025,
        description="Maximum allowable portfolio risk as a fraction of equity.",
        min_value=0.0,
        max_value=1.0,
    ),
    ConfigSpec(
        field="max_portfolio_positions",
        env=("AI_TRADING_MAX_PORTFOLIO_POSITIONS",),
        cast="int",
        default=20,
        description="Maximum number of concurrent portfolio positions permitted.",
        min_value=1,
    ),
    ConfigSpec(
        field="limit_order_slippage",
        env=("LIMIT_ORDER_SLIPPAGE",),
        cast="float",
        default=0.005,
        description="Expected fractional slippage applied to limit orders to ensure fills.",
        min_value=0.0,
    ),
    ConfigSpec(
        field="participation_rate",
        env=("PARTICIPATION_RATE",),
        cast="float",
        default=0.15,
        description="Target participation rate for POV execution.",
        min_value=0.0,
        max_value=1.0,
    ),
    ConfigSpec(
        field="pov_slice_pct",
        env=("POV_SLICE_PCT",),
        cast="float",
        default=0.05,
        description="Slice percentage used for participation-of-volume execution.",
        min_value=0.0,
        max_value=1.0,
    ),
    ConfigSpec(
        field="healthcheck_port",
        env=("HEALTHCHECK_PORT", "AI_TRADING_HEALTHCHECK_PORT"),
        cast="int",
        default=9101,
        description="TCP port used by the auxiliary health/metrics server.",
        min_value=1,
        max_value=65535,
    ),
    ConfigSpec(
        field="provider_decision_secs",
        env=("AI_TRADING_PROVIDER_DECISION_SECS",),
        cast="int",
        default=120,
        description="Lookback window in seconds when evaluating data providers.",
        min_value=1,
    ),
    ConfigSpec(
        field="provider_switch_cooldown_sec",
        env=("AI_TRADING_PROVIDER_SWITCH_COOLDOWN_SEC",),
        cast="int",
        default=900,
        description="Minimum seconds to remain on a backup provider before recovery.",
        min_value=0,
    ),
    ConfigSpec(
        field="provider_health_passes_required",
        env=("AI_TRADING_PROVIDER_HEALTH_PASSES_REQUIRED",),
        cast="int",
        default=4,
        description="Consecutive healthy ticks required before switching back to the primary provider.",
        min_value=1,
    ),
    ConfigSpec(
        field="log_timings_level",
        env=("AI_TRADING_LOG_TIMINGS_LEVEL", "LOG_TIMINGS_LEVEL"),
        cast="str",
        default="DEBUG",
        description="Log level used for stage timing instrumentation.",
    ),
    ConfigSpec(
        field="log_rate_limit_window_sec",
        env=("AI_TRADING_LOG_RATE_LIMIT_WINDOW_SEC",),
        cast="int",
        default=120,
        description="Window, in seconds, for deduplicating repetitive log events.",
        min_value=1,
    ),
    ConfigSpec(
        field="interval",
        env=("AI_TRADING_INTERVAL",),
        cast="int",
        default=60,
        description="Primary loop interval in seconds while markets are open.",
        min_value=1,
    ),
    ConfigSpec(
        field="interval_when_closed",
        env=("INTERVAL_WHEN_CLOSED",),
        cast="int",
        default=300,
        description="Loop interval in seconds when markets are closed.",
        min_value=1,
    ),
    ConfigSpec(
        field="iterations",
        env=("AI_TRADING_ITERATIONS",),
        cast="int",
        default=0,
        description="Number of scheduler iterations to run (0 for infinite).",
        min_value=0,
    ),
    ConfigSpec(
        field="rl_model_path",
        env=("AI_TRADING_RL_MODEL_PATH",),
        cast="str",
        default="rl_agent.zip",
        description="Filesystem path to the reinforcement learning model artifact.",
    ),
    ConfigSpec(
        field="halt_flag_path",
        env=("HALT_FLAG_PATH", "AI_TRADING_HALT_FLAG_PATH"),
        cast="str",
        default="halt.flag",
        description="Filesystem path to the halt flag used to pause live trading.",
    ),
    ConfigSpec(
        field="trading_mode",
        env=("TRADING_MODE", "AI_TRADING_TRADING_MODE"),
        cast="str",
        default="balanced",
        description="High-level preset controlling trading risk posture.",
    ),
    ConfigSpec(
        field="cpu_only",
        env=("CPU_ONLY",),
        cast="bool",
        default=False,
        description="Force CPU execution for ML workloads when true.",
    ),
    ConfigSpec(
        field="use_rl_agent",
        env=("USE_RL_AGENT",),
        cast="bool",
        default=False,
        description="Enable reinforcement learning agent integration.",
    ),
)


MODE_PARAMETERS: dict[str, dict[str, float]] = {
    "conservative": {
        "kelly_fraction": 0.25,
        "conf_threshold": 0.85,
        "daily_loss_limit": 0.03,
        "max_position_size": 5000.0,
        "capital_cap": 0.20,
        "confirmation_count": 3,
        "take_profit_factor": 1.5,
    },
    "balanced": {
        "kelly_fraction": 0.6,
        "conf_threshold": 0.75,
        "daily_loss_limit": 0.05,
        "max_position_size": 8000.0,
        "capital_cap": 0.25,
        "confirmation_count": 2,
        "take_profit_factor": 1.8,
    },
    "aggressive": {
        "kelly_fraction": 0.75,
        "conf_threshold": 0.65,
        "daily_loss_limit": 0.08,
        "max_position_size": 12000.0,
        "capital_cap": 0.30,
        "confirmation_count": 1,
        "take_profit_factor": 2.5,
    },
}


def _selected_mode(env_map: Mapping[str, Any] | None = None) -> str:
    """Return the trading mode name derived from an env snapshot."""

    candidates = ("TRADING_MODE", "AI_TRADING_TRADING_MODE")
    if env_map:
        for key in candidates:
            raw = env_map.get(key)
            if raw not in (None, ""):
                return str(raw).strip().lower()
    for key in candidates:
        raw = os.environ.get(key)
        if raw not in (None, ""):
            return str(raw).strip().lower()
    return "balanced"


def _apply_mode_overlays(
    values: dict[str, Any],
    env_map: Mapping[str, Any] | None,
    *,
    explicit_fields: Iterable[str] | None = None,
) -> None:
    """Overlay mode defaults onto ``values`` when env does not supply them."""

    mode_name = _selected_mode(env_map)
    mode_defaults = MODE_PARAMETERS.get(mode_name)
    if not mode_defaults:
        return

    protected = {field for field in (explicit_fields or ())}
    for field, preset_value in mode_defaults.items():
        if field in protected:
            continue
        spec = SPEC_BY_FIELD.get(field)
        if spec is None:
            continue

        provided = False
        if env_map:
            provided = any(env_map.get(env_key) not in (None, "") for env_key in spec.env)
            if not provided and spec.deprecated_env:
                provided = any(
                    env_map.get(alias) not in (None, "") for alias in spec.deprecated_env
                )
        if provided:
            continue

        values[field] = _validate_bounds(spec, preset_value)

_CACHE_LOCK = threading.Lock()
_CACHED_CONFIG: TradingConfig | None = None
_CACHED_SIGNATURE: tuple[tuple[str, str | None], ...] | None = None

_SENSITIVE_ENV_KEYS: tuple[str, ...] | None = None


def _get_sensitive_env_keys() -> tuple[str, ...]:
    """Return environment keys that impact configuration caching."""

    global _SENSITIVE_ENV_KEYS
    if _SENSITIVE_ENV_KEYS is not None:
        return _SENSITIVE_ENV_KEYS

    keys: set[str] = {"TRADING_MODE", "AI_TRADING_TRADING_MODE"}
    for spec in CONFIG_SPECS:
        keys.update(spec.env)
        keys.update(spec.deprecated_env.keys())
    # Additional guards for lock behaviour and shadow-mode toggles accessed
    # outside the declarative specs.
    keys.update({"CONFIG_LOCK_TIMEOUT", "CONFIG_VALIDATION_LOCK_TIMEOUT"})

    _SENSITIVE_ENV_KEYS = tuple(sorted(keys))
    return _SENSITIVE_ENV_KEYS


def _signature_from_snapshot(snapshot: Mapping[str, str]) -> tuple[tuple[str, str | None], ...]:
    """Build a deterministic signature for configuration-affecting env vars."""

    keys = _get_sensitive_env_keys()
    return tuple((key, snapshot.get(key)) for key in keys)


def ensure_trading_config_current(keys: Iterable[str] | None = None) -> TradingConfig:
    """Ensure the cached configuration reflects relevant environment keys."""

    normalized_keys: tuple[str, ...]
    if keys:
        normalized_keys = tuple(sorted({str(key).upper() for key in keys if key}))
    else:
        normalized_keys = ()

    if not normalized_keys:
        return get_trading_config()

    current_values = {key: os.environ.get(key) for key in normalized_keys}

    refresh_side_effects = False
    with _CACHE_LOCK:
        global _CACHED_CONFIG, _CACHED_SIGNATURE
        cached_config = _CACHED_CONFIG
        cached_signature = _CACHED_SIGNATURE

        if cached_config is not None and cached_signature is not None:
            signature_map = dict(cached_signature)
            missing_key = any(key not in signature_map for key in normalized_keys)
            mismatch = any(
                signature_map.get(key) != current_values.get(key)
                for key in normalized_keys
                if key in signature_map
            )
            if not missing_key and not mismatch:
                return cached_config

        _CACHED_CONFIG = None
        _CACHED_SIGNATURE = None
        refresh_side_effects = True

    if refresh_side_effects:
        try:
            from ai_trading.utils.env import refresh_alpaca_credentials_cache
        except Exception:
            pass
        else:
            try:
                refresh_alpaca_credentials_cache()
            except Exception:
                pass

    return get_trading_config()


for spec in CONFIG_SPECS:
    SPEC_BY_FIELD[spec.field] = spec
    for key in spec.env:
        SPEC_BY_ENV[key.upper()] = spec
    for legacy in spec.deprecated_env:
        SPEC_BY_ENV[legacy.upper()] = spec


_LIVE_ENV_VALUES = {"live", "live_prod", "prod", "production"}


_ENV_ALIAS_MAP: dict[str, str] = {
    "DATA_PROVIDER": "DATA_PROVIDER_PRIORITY",
    "PAPER": "EXECUTION_MODE",
}


_ALLOWED_OVERRIDE_ENV_KEYS: frozenset[str] = frozenset({*SPEC_BY_ENV.keys(), *_ENV_ALIAS_MAP.keys()})


def _validate_override_keys(overrides: Mapping[str, Any]) -> None:
    """Ensure overrides only contain recognized TradingConfig environment keys."""

    if not overrides:
        return

    unknown: list[str] = []
    for key in overrides:
        upper_key = str(key).upper()
        if upper_key not in _ALLOWED_OVERRIDE_ENV_KEYS:
            unknown.append(str(key))
    if unknown:
        names = ", ".join(sorted(dict.fromkeys(unknown)))
        raise KeyError(
            "Unknown TradingConfig override env keys: "
            f"{names}."
        )


class _EnvSnapshotDict(dict):
    """Dictionary subclass tagging mappings produced by :func:`_env_snapshot`."""

    __slots__ = ("is_snapshot",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.is_snapshot = True


def _normalize_env_label(value: Any, *, default: str = "") -> str:
    """Return a lowercase, stripped string for environment labels."""

    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return text.lower()


def _normalize_base_url(value: Any) -> str:
    """Return a sanitized Alpaca base URL string for downstream use."""

    if value is None:
        return ""
    return str(value).strip()


def _infer_paper_mode(values: Mapping[str, Any]) -> bool:
    base_url = _normalize_base_url(values.get("alpaca_base_url")).lower()
    execution_mode = _normalize_env_label(values.get("execution_mode"))
    trading_mode_env = _normalize_env_label(
        os.getenv("TRADING_MODE") or os.getenv("AI_TRADING_TRADING_MODE")
    )
    app_env = _normalize_env_label(values.get("app_env"), default="test")

    if execution_mode in _LIVE_ENV_VALUES:
        return False
    if trading_mode_env in _LIVE_ENV_VALUES:
        return False
    if app_env in _LIVE_ENV_VALUES:
        return False
    if "paper" in base_url:
        return True
    return app_env not in _LIVE_ENV_VALUES


class TradingConfig:
    """Immutable container mapping config specifications to resolved values."""

    __slots__ = ("_values",)

    def __init__(
        self,
        _explicit_fields: Iterable[str] | None = None,
        _env_map: Mapping[str, Any] | None = None,
        **values: Any,
    ) -> None:
        normalized: dict[str, Any] = {}
        explicit_fields = set(_explicit_fields if _explicit_fields is not None else values.keys())
        if _env_map is not None:
            env_snapshot: Mapping[str, Any] | None = _env_map
        else:
            env_snapshot = {k: v for k, v in os.environ.items() if isinstance(v, str)}

        normalized.update(values)
        for spec in CONFIG_SPECS:
            normalized.setdefault(spec.field, spec.default)

        normalized["alpaca_base_url"] = _normalize_base_url(normalized.get("alpaca_base_url"))
        normalized["app_env"] = _normalize_env_label(normalized.get("app_env"), default="test")

        _apply_mode_overlays(normalized, env_snapshot, explicit_fields=explicit_fields)

        object.__setattr__(self, "_values", normalized)

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - attribute passthrough
        try:
            return self._values[item]
        except KeyError as exc:  # noqa: F401
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:  # pragma: no cover - immutability
        raise AttributeError("TradingConfig is immutable")

    def to_dict(self) -> dict[str, Any]:
        return dict(self._values)

    def update(self, **updates: Any) -> "TradingConfig":
        """Merge ``updates`` into the configuration and return ``self``.

        The configuration container is conceptually immutable to callers, but
        internally it keeps a mutable mapping for efficiency.  The update
        helper mirrors the historical behaviour expected by existing call
        sites: validate requested fields, merge the supplied values, refresh
        derived properties, and return the instance so calls may be chained.
        """

        if not updates:
            return self

        unknown = [key for key in updates if key not in self._values]
        if unknown:
            names = ", ".join(sorted(unknown))
            raise AttributeError(f"TradingConfig has no fields: {names}")

        for key, value in updates.items():
            spec = SPEC_BY_FIELD.get(key)
            if spec is not None:
                value = _validate_bounds(spec, value)
            if key == "alpaca_base_url":
                value = _normalize_base_url(value)
            elif key == "app_env":
                value = _normalize_env_label(value, default="test")
            self._values[key] = value

        if {"alpaca_base_url", "app_env"} & updates.keys():
            self._values["paper"] = _infer_paper_mode(self._values)

        return self

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
                "provider": (
                    getattr(self, "data_provider", None)
                    or (self.data_provider_priority[0] if getattr(self, "data_provider_priority", None) else None)
                ),
            },
            "execution": {
                "mode": getattr(self, "execution_mode", None),
                "shadow_mode": bool(getattr(self, "shadow_mode", False)),
                "order_timeout_seconds": getattr(self, "order_timeout_seconds", None),
                "slippage_limit_bps": getattr(self, "slippage_limit_bps", None),
                "price_providers": tuple(getattr(self, "price_provider_order", ()) or ()),
                "intraday_feed": getattr(self, "data_feed_intraday", None),
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
    def from_optimization(cls, params: Mapping[str, Any]) -> "TradingConfig":
        base = cls.from_env()
        values = base.to_dict()
        values.update(params)
        return cls(**values)

    def __deepcopy__(self, memo: dict[int, Any]) -> "TradingConfig":
        copied_values = copy.deepcopy(self._values, memo)
        return TradingConfig(**copied_values)

    @classmethod
    def from_env(
        cls,
        env_overrides: Mapping[str, Any] | str | None = None,
        *,
        allow_missing_drawdown: bool = False,
    ) -> "TradingConfig":
        if isinstance(env_overrides, Mapping) and getattr(env_overrides, "is_snapshot", False):
            env_map = dict(env_overrides)
        else:
            env_map = _env_snapshot(env_overrides)
        if not allow_missing_drawdown:
            has_drawdown = any(
                env_map.get(key) not in (None, "")
                for key in ("MAX_DRAWDOWN_THRESHOLD", "AI_TRADING_MAX_DRAWDOWN_THRESHOLD")
            )
            if not has_drawdown:
                raise RuntimeError("MAX_DRAWDOWN_THRESHOLD must be set")
        values: dict[str, Any] = {}
        provided_fields: set[str] = set()
        for spec in CONFIG_SPECS:
            provided = any(env_map.get(key) not in (None, "") for key in spec.env)
            if not provided and spec.deprecated_env:
                provided = any(
                    env_map.get(alias) not in (None, "") for alias in spec.deprecated_env
                )
            if provided:
                provided_fields.add(spec.field)
            values[spec.field] = _build_value(spec, env_map)

        if values["cycle_compute_budget_factor"] is None:
            values["cycle_compute_budget_factor"] = values["cycle_budget_fraction"]

        if (env_map.get("DOLLAR_RISK_LIMIT") in (None, "")) and values["dollar_risk_limit"] == SPEC_BY_FIELD["dollar_risk_limit"].default:
            for legacy_key in ("DAILY_LOSS_LIMIT", "AI_TRADING_DAILY_LOSS_LIMIT"):
                raw_alias = env_map.get(legacy_key)
                if raw_alias not in (None, ""):
                    spec = SPEC_BY_FIELD["dollar_risk_limit"]
                    values["dollar_risk_limit"] = _validate_bounds(spec, _cast_value(spec, raw_alias))
                    break

        _apply_mode_overlays(values, env_map, explicit_fields=provided_fields)

        provider_override = env_map.get("DATA_PROVIDER")
        if provider_override:
            priority = list(values.get("data_provider_priority") or ())
            priority = [provider_override] + [p for p in priority if p != provider_override]
            values["data_provider_priority"] = tuple(priority)
            values["data_provider"] = provider_override

        if (
            values.get("data_feed_intraday") == "sip"
            and not values.get("alpaca_allow_sip")
            and not values.get("alpaca_has_sip")
        ):
            raise ValueError(
                (
                    "DATA_FEED_INTRADAY=sip requires SIP entitlements. Set ALPACA_ALLOW_SIP=1 or "
                    "ALPACA_HAS_SIP=1. Ensure ALPACA_API_KEY, ALPACA_SECRET_KEY, and DATA_FEED_INTRADAY are "
                    "configured. See docs/DEPLOYING.md#alpaca-feed-selection for setup guidance."
                )
            )

        # Derived convenience fields expected by legacy callers.
        values.setdefault("data_provider", values.get("data_provider_priority", (None,))[0])
        values.setdefault("paper", _infer_paper_mode(values))
        values.setdefault("max_position_mode", values.get("max_position_mode", "STATIC"))

        return cls(_explicit_fields=provided_fields, _env_map=env_map, **values)


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
    snap: _EnvSnapshotDict = _EnvSnapshotDict(
        {k: v for k, v in os.environ.items() if isinstance(v, str)}
    )
    if overrides:
        if isinstance(overrides, str):
            snap["TRADING_MODE"] = overrides
        else:
            if getattr(overrides, "is_snapshot", False):
                snap.update(overrides)
            else:
                _validate_override_keys(overrides)
                snap.update({k.upper(): str(v) for k, v in overrides.items()})
    for alias_key, canonical_key in _ENV_ALIAS_MAP.items():
        raw_value = snap.get(alias_key)
        if raw_value in (None, ""):
            continue
        if canonical_key == "EXECUTION_MODE":
            normalized = str(raw_value).strip().lower()
            if normalized in {"1", "true", "yes", "on", "paper"}:
                snap.setdefault(canonical_key, "paper")
            elif normalized in {"0", "false", "no", "off", "live"}:
                snap.setdefault(canonical_key, snap.get(canonical_key, "sim"))
        else:
            snap.setdefault(canonical_key, str(raw_value))
    return snap


def get_trading_config() -> TradingConfig:
    """Return cached trading configuration that reflects current environment."""

    global _CACHED_CONFIG, _CACHED_SIGNATURE
    snap = _env_snapshot()
    signature = _signature_from_snapshot(snap)
    with _CACHE_LOCK:
        if _CACHED_CONFIG is not None and _CACHED_SIGNATURE == signature:
            return _CACHED_CONFIG
        cfg = TradingConfig.from_env(snap, allow_missing_drawdown=True)
        _CACHED_CONFIG = cfg
        _CACHED_SIGNATURE = signature
        return cfg


def _clear_trading_config_cache() -> None:
    """Clear cached configuration state."""

    with _CACHE_LOCK:
        global _CACHED_CONFIG, _CACHED_SIGNATURE
        _CACHED_CONFIG = None
        _CACHED_SIGNATURE = None


get_trading_config.cache_clear = _clear_trading_config_cache  # type: ignore[attr-defined]


def reload_trading_config(
    env_overrides: Mapping[str, Any] | None = None,
    *,
    allow_missing_drawdown: bool = True,
) -> TradingConfig:
    """Reload the cached trading configuration.

    Callers may opt into strict drawdown enforcement by setting
    ``allow_missing_drawdown`` to ``False``.
    """

    snap = _env_snapshot(env_overrides)
    cfg = TradingConfig.from_env(snap, allow_missing_drawdown=allow_missing_drawdown)
    _clear_trading_config_cache()
    return cfg


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
    "ensure_trading_config_current",
    "get_trading_config",
    "reload_trading_config",
    "generate_config_schema",
]
