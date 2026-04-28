from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import logging

from ai_trading.alpaca_api import ALPACA_AVAILABLE, get_trading_client_cls
from ai_trading.config.management import get_env
from .settings import broker_keys, get_settings

@dataclass(frozen=True)
class AlpacaConfig:
    base_url: str
    key_id: str
    secret_key: str
    use_paper: bool
    rate_limit_per_min: int | None = None
    data_feed: str = "iex"

    @property
    def key(self) -> str:
        """Return the standard key alias.

        Some callers expect a generic ``key`` attribute rather than
        ``key_id``.  Exposing this property keeps backwards compatibility
        while allowing ``ALPACA_API_KEY`` to map to a consistent field.
        """

        return self.key_id

ENV_PREFIXES = (
    "DEV_",
    "PROD_",
    "TEST_",
    "STAGING_",
    "STAGE_",
    "PAPER_",
    "LIVE_",
)

_LIVE_MODE_ALIASES = {"live", "prod", "production", "live_prod"}
_PAPER_MODE_ALIASES = {"paper", "broker", "alpaca"}


def _normalize_key_name(name: str) -> str:
    name = name.strip().upper()
    for prefix in ENV_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _normalize_execution_mode(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in _LIVE_MODE_ALIASES:
        return "live"
    if raw in _PAPER_MODE_ALIASES:
        return "paper"
    if raw in {"simulation", "sim", "test"}:
        return "sim"
    if raw in {"disabled", "off", "none"}:
        return "disabled"
    return raw


def _resolve_base_url_and_paper(settings: Any) -> tuple[str, bool]:
    raw_base_url = str(
        get_env("ALPACA_TRADING_BASE_URL", "", cast=str, resolve_aliases=False) or ""
    ).strip()
    settings_base_url = str(getattr(settings, "alpaca_base_url", "") or "").strip()
    execution_mode = _normalize_execution_mode(
        get_env(
            "EXECUTION_MODE",
            getattr(settings, "execution_mode", "sim"),
            cast=str,
        )
    )
    app_env = str(getattr(settings, "env", "dev") or "dev").strip().lower()
    default_paper_url = "https://paper-api.alpaca.markets"

    base_url = raw_base_url
    if not base_url and settings_base_url and (
        settings_base_url != default_paper_url or execution_mode != "live"
    ):
        base_url = settings_base_url

    if base_url:
        normalized_url = base_url.lower()
        if "paper" in normalized_url:
            return base_url, True
        if "api.alpaca.markets" in normalized_url:
            return base_url, False
        return base_url, execution_mode != "live" and app_env not in _LIVE_MODE_ALIASES

    if execution_mode == "live" or app_env in _LIVE_MODE_ALIASES:
        return "https://api.alpaca.markets", False

    return default_paper_url, True


def get_alpaca_config() -> AlpacaConfig:
    s = get_settings()
    keys: Any = broker_keys()
    if isinstance(keys, Mapping):
        normalized: dict[str, str] = {}
        for k, v in keys.items():
            nk = _normalize_key_name(str(k))
            normalized[nk] = str(v).strip()
        if "ALPACA_API_KEY" in normalized and "KEY" not in normalized:
            normalized["KEY"] = normalized["ALPACA_API_KEY"]
        if "ALPACA_SECRET_KEY" in normalized and "SECRET" not in normalized:
            normalized["SECRET"] = normalized["ALPACA_SECRET_KEY"]
        key_id = (
            normalized.get("KEY")
            or normalized.get("ALPACA_KEY_ID")
            or normalized.get("ALPACA_API_KEY")
            or ""
        )
        secret = normalized.get("SECRET") or normalized.get("ALPACA_SECRET_KEY", "")
    else:
        key_id, secret = [str(x).strip() for x in keys]
    base_url, use_paper = _resolve_base_url_and_paper(s)
    rate_limit = getattr(s, 'alpaca_rate_limit_per_min', None)
    feed = str(
        get_env("ALPACA_DATA_FEED", getattr(s, "alpaca_data_feed", "iex"), cast=str, resolve_aliases=False)
        or getattr(s, "alpaca_data_feed", "iex")
    ).lower()
    if not key_id or not secret:
        raise RuntimeError('Missing Alpaca credentials in broker_keys() (ALPACA_KEY_ID/ALPACA_SECRET_KEY)')
    if ALPACA_AVAILABLE:
        try:
            TradingClient = get_trading_client_cls()
            client = TradingClient(api_key=key_id, secret_key=secret, paper=use_paper)
            if hasattr(client, 'get_account'):
                acct = client.get_account()
                sub = getattr(acct, 'market_data_subscription', None) or getattr(acct, 'data_feed', None)
                if isinstance(sub, str):
                    entitled = {sub.lower()}
                elif isinstance(sub, (set, list, tuple)):
                    entitled = {str(x).lower() for x in sub}
                else:
                    entitled = set()
                if entitled and feed not in entitled:
                    alt = next(iter(entitled))
                    logging.getLogger(__name__).warning(
                        'ALPACA_FEED_UNENTITLED_SWITCH', extra={'requested': feed, 'using': alt}
                    )
                    feed = alt
            else:
                logging.getLogger(__name__).warning('ALPACA_CLIENT_NO_GET_ACCOUNT')
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            pass
    return AlpacaConfig(
        base_url=base_url,
        key_id=key_id,
        secret_key=secret,
        use_paper=use_paper,
        rate_limit_per_min=rate_limit,
        data_feed=feed,
    )
