"""Launch-profile policy for paper and live-capital readiness gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ai_trading.config.management import get_env


@dataclass(frozen=True)
class LaunchProfile:
    name: str
    max_gross_exposure: float
    max_symbol_exposure: float
    max_order_count: int
    max_daily_loss: float | None
    max_notional_per_order: float | None
    shorts_allowed: bool
    allowed_symbols: tuple[str, ...]
    promotion_required: bool
    provider_policy: str
    execution_quote_authority: str
    backup_provider_live_policy: str
    decay_behavior: str
    manual_approval_required: bool
    max_quote_age_ms: float | None
    max_spread_bps: float | None


_PROFILE_DEFAULTS: dict[str, LaunchProfile] = {
    "paper_observe": LaunchProfile(
        name="paper_observe",
        max_gross_exposure=0.0,
        max_symbol_exposure=0.0,
        max_order_count=0,
        max_daily_loss=None,
        max_notional_per_order=None,
        shorts_allowed=False,
        allowed_symbols=(),
        promotion_required=False,
        provider_policy="research_only",
        execution_quote_authority="alpaca_only",
        backup_provider_live_policy="research_only",
        decay_behavior="observe",
        manual_approval_required=False,
        max_quote_age_ms=None,
        max_spread_bps=None,
    ),
    "paper_trade": LaunchProfile(
        name="paper_trade",
        max_gross_exposure=1.0,
        max_symbol_exposure=0.35,
        max_order_count=25,
        max_daily_loss=None,
        max_notional_per_order=None,
        shorts_allowed=True,
        allowed_symbols=(),
        promotion_required=False,
        provider_policy="paper_fallback_allowed",
        execution_quote_authority="alpaca_only",
        backup_provider_live_policy="research_only",
        decay_behavior="normal",
        manual_approval_required=False,
        max_quote_age_ms=None,
        max_spread_bps=None,
    ),
    "live_canary": LaunchProfile(
        name="live_canary",
        max_gross_exposure=0.03,
        max_symbol_exposure=0.015,
        max_order_count=3,
        max_daily_loss=25.0,
        max_notional_per_order=100.0,
        shorts_allowed=False,
        allowed_symbols=("AAPL", "AMZN"),
        promotion_required=True,
        provider_policy="strict_live",
        execution_quote_authority="alpaca_only",
        backup_provider_live_policy="research_only",
        decay_behavior="fail_closed",
        manual_approval_required=True,
        max_quote_age_ms=1000.0,
        max_spread_bps=15.0,
    ),
    "live_restricted": LaunchProfile(
        name="live_restricted",
        max_gross_exposure=0.10,
        max_symbol_exposure=0.03,
        max_order_count=8,
        max_daily_loss=100.0,
        max_notional_per_order=500.0,
        shorts_allowed=False,
        allowed_symbols=("AAPL", "AMZN"),
        promotion_required=True,
        provider_policy="strict_live",
        execution_quote_authority="alpaca_only",
        backup_provider_live_policy="research_only",
        decay_behavior="fail_closed",
        manual_approval_required=True,
        max_quote_age_ms=1500.0,
        max_spread_bps=25.0,
    ),
    "live_normal": LaunchProfile(
        name="live_normal",
        max_gross_exposure=0.35,
        max_symbol_exposure=0.08,
        max_order_count=20,
        max_daily_loss=250.0,
        max_notional_per_order=1500.0,
        shorts_allowed=False,
        allowed_symbols=(),
        promotion_required=True,
        provider_policy="strict_live",
        execution_quote_authority="alpaca_only",
        backup_provider_live_policy="research_only",
        decay_behavior="reduce_or_fail_closed",
        manual_approval_required=True,
        max_quote_age_ms=2500.0,
        max_spread_bps=35.0,
    ),
}


def _csv_symbols(value: str) -> tuple[str, ...]:
    return tuple(
        token.strip().upper()
        for token in str(value or "").split(",")
        if token.strip()
    )


def _float_env(name: str, default: float | None) -> float | None:
    raw = get_env(name, None, cast=str, resolve_aliases=False)
    if raw in (None, ""):
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = get_env(name, None, cast=str, resolve_aliases=False)
    if raw in (None, ""):
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = get_env(name, None, cast=str, resolve_aliases=False)
    if raw in (None, ""):
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _text_env(name: str, default: str) -> str:
    return str(get_env(name, default, cast=str, resolve_aliases=False) or default).strip()


def _tighten_float(default: float | None, override: float | None) -> float | None:
    if default is None:
        return override
    if override is None:
        return default
    return min(float(default), float(override))


def _tighten_symbols(default: tuple[str, ...], override: tuple[str, ...]) -> tuple[str, ...]:
    if not default:
        return override
    if not override:
        return default
    allowed = set(default)
    return tuple(symbol for symbol in override if symbol in allowed)


def active_launch_profile_name() -> str:
    raw = _text_env("AI_TRADING_LAUNCH_PROFILE", "")
    if raw:
        return raw.lower()
    mode = _text_env("EXECUTION_MODE", "paper").lower()
    if mode == "live":
        return "live_canary"
    if mode == "paper":
        return "paper_trade"
    return "paper_observe"


def resolve_launch_profile(name: str | None = None) -> LaunchProfile:
    profile_name = str(name or active_launch_profile_name()).strip().lower()
    base = _PROFILE_DEFAULTS.get(profile_name, _PROFILE_DEFAULTS["paper_observe"])
    prefix = f"AI_TRADING_LAUNCH_PROFILE_{base.name.upper()}"
    allowed_raw = _text_env(f"{prefix}_SYMBOLS", "")
    max_daily_loss = _float_env(
        f"{prefix}_MAX_DAILY_LOSS",
        _float_env("AI_TRADING_LIVE_MAX_DAILY_LOSS", base.max_daily_loss),
    )
    live_profile = base.name.startswith("live_")
    max_gross_exposure = _float_env(
        f"{prefix}_MAX_GROSS_EXPOSURE",
        base.max_gross_exposure,
    )
    max_symbol_exposure = _float_env(
        f"{prefix}_MAX_SYMBOL_EXPOSURE",
        base.max_symbol_exposure,
    )
    max_notional_per_order = _float_env(
        f"{prefix}_MAX_NOTIONAL_PER_ORDER",
        base.max_notional_per_order,
    )
    max_quote_age_ms = _float_env(f"{prefix}_MAX_QUOTE_AGE_MS", base.max_quote_age_ms)
    max_spread_bps = _float_env(f"{prefix}_MAX_SPREAD_BPS", base.max_spread_bps)
    max_order_count = max(0, _int_env(f"{prefix}_MAX_ORDER_COUNT", base.max_order_count))
    shorts_allowed = _bool_env(
        f"{prefix}_ALLOW_SHORTS",
        _bool_env("TRADING__ALLOW_SHORTS", base.shorts_allowed),
    )
    allowed_symbols = _csv_symbols(allowed_raw)
    if live_profile:
        max_gross_exposure = _tighten_float(base.max_gross_exposure, max_gross_exposure)
        max_symbol_exposure = _tighten_float(base.max_symbol_exposure, max_symbol_exposure)
        max_daily_loss = _tighten_float(base.max_daily_loss, max_daily_loss)
        max_notional_per_order = _tighten_float(
            base.max_notional_per_order,
            max_notional_per_order,
        )
        max_quote_age_ms = _tighten_float(base.max_quote_age_ms, max_quote_age_ms)
        max_spread_bps = _tighten_float(base.max_spread_bps, max_spread_bps)
        max_order_count = min(max_order_count, base.max_order_count)
        shorts_allowed = bool(base.shorts_allowed and shorts_allowed)
        allowed_symbols = _tighten_symbols(base.allowed_symbols, allowed_symbols)
    return LaunchProfile(
        name=base.name,
        max_gross_exposure=max_gross_exposure or base.max_gross_exposure,
        max_symbol_exposure=max_symbol_exposure or base.max_symbol_exposure,
        max_order_count=max_order_count,
        max_daily_loss=max_daily_loss,
        max_notional_per_order=max_notional_per_order,
        shorts_allowed=shorts_allowed,
        allowed_symbols=allowed_symbols or base.allowed_symbols,
        promotion_required=_bool_env(f"{prefix}_PROMOTION_REQUIRED", base.promotion_required),
        provider_policy=_text_env(
            "AI_TRADING_PROVIDER_AUTHORITY_POLICY",
            _text_env(f"{prefix}_PROVIDER_POLICY", base.provider_policy),
        ),
        execution_quote_authority=_text_env(
            "AI_TRADING_EXECUTION_QUOTE_AUTHORITY",
            _text_env(f"{prefix}_EXECUTION_QUOTE_AUTHORITY", base.execution_quote_authority),
        ),
        backup_provider_live_policy=_text_env(
            "AI_TRADING_BACKUP_PROVIDER_LIVE_POLICY",
            _text_env(f"{prefix}_BACKUP_PROVIDER_LIVE_POLICY", base.backup_provider_live_policy),
        ),
        decay_behavior=_text_env(f"{prefix}_DECAY_BEHAVIOR", base.decay_behavior),
        manual_approval_required=_bool_env(
            f"{prefix}_MANUAL_APPROVAL_REQUIRED",
            base.manual_approval_required,
        ),
        max_quote_age_ms=max_quote_age_ms,
        max_spread_bps=max_spread_bps,
    )


def launch_profile_payload(profile: LaunchProfile | None = None) -> dict[str, Any]:
    resolved = profile or resolve_launch_profile()
    return {
        "name": resolved.name,
        "max_gross_exposure": resolved.max_gross_exposure,
        "max_symbol_exposure": resolved.max_symbol_exposure,
        "max_order_count": resolved.max_order_count,
        "max_daily_loss": resolved.max_daily_loss,
        "max_notional_per_order": resolved.max_notional_per_order,
        "shorts_allowed": resolved.shorts_allowed,
        "allowed_symbols": list(resolved.allowed_symbols),
        "promotion_required": resolved.promotion_required,
        "provider_policy": resolved.provider_policy,
        "execution_quote_authority": resolved.execution_quote_authority,
        "backup_provider_live_policy": resolved.backup_provider_live_policy,
        "decay_behavior": resolved.decay_behavior,
        "manual_approval_required": resolved.manual_approval_required,
        "max_quote_age_ms": resolved.max_quote_age_ms,
        "max_spread_bps": resolved.max_spread_bps,
    }


def provider_authority_allows(
    *,
    profile: LaunchProfile | None = None,
    provider_state: Mapping[str, Any] | None = None,
    quote_state: Mapping[str, Any] | None = None,
    execution_mode: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    resolved = profile or resolve_launch_profile()
    provider_state = provider_state or {}
    quote_state = quote_state or {}
    mode = str(execution_mode or _text_env("EXECUTION_MODE", "paper")).strip().lower()
    active = str(provider_state.get("active") or provider_state.get("primary") or "").lower()
    using_backup = bool(provider_state.get("using_backup"))
    provider_status = str(provider_state.get("status") or "").lower()
    quote_source = str(quote_state.get("source") or "").lower()
    synthetic = bool(quote_state.get("synthetic"))
    quote_allowed = bool(quote_state.get("allowed", True))
    policy = resolved.execution_quote_authority.lower()
    reasons: list[str] = []
    if policy == "alpaca_only":
        source_text = f"{active} {quote_source}"
        if "alpaca" not in source_text:
            reasons.append("execution_quote_not_alpaca")
    if mode == "live" or resolved.name.startswith("live_"):
        if resolved.provider_policy == "strict_live" and not active:
            reasons.append("provider_unknown")
        if resolved.provider_policy == "strict_live" and not quote_source:
            reasons.append("quote_source_unknown")
        if using_backup and resolved.backup_provider_live_policy == "research_only":
            reasons.append("backup_provider_research_only")
        if provider_status in {"unknown", "degraded", "unhealthy", "down", "error"}:
            reasons.append(f"provider_{provider_status}")
        if synthetic:
            reasons.append("synthetic_quote")
        if not quote_allowed:
            reasons.append("quote_not_allowed")
    return not reasons, {
        "profile": resolved.name,
        "policy": resolved.provider_policy,
        "execution_quote_authority": resolved.execution_quote_authority,
        "backup_provider_live_policy": resolved.backup_provider_live_policy,
        "provider_active": active,
        "provider_status": provider_status,
        "using_backup": using_backup,
        "quote_source": quote_source or None,
        "quote_synthetic": synthetic,
        "quote_allowed": quote_allowed,
        "reasons": reasons,
    }


__all__ = [
    "LaunchProfile",
    "active_launch_profile_name",
    "launch_profile_payload",
    "provider_authority_allows",
    "resolve_launch_profile",
]
