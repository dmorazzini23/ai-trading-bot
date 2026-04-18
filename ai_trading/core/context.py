from __future__ import annotations

"""Context utilities and lightweight singleton access."""

import importlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from alpaca.data.historical.stock import (
        StockHistoricalDataClient as AlpacaStockHistoricalDataClient,
    )
    from alpaca.trading.client import TradingClient as AlpacaTradingClient

from ai_trading.settings import get_settings, get_alpaca_secret_key_plain
from ai_trading.data.feed_roles import get_execution_feed, get_reference_feed


_CTX: SimpleNamespace | None = None


def get_context() -> SimpleNamespace:
    """Return a singleton runtime context.

    The object includes a small set of configuration attributes and Alpaca
    client handles. Missing Alpaca clients are represented as ``None``.
    """

    global _CTX
    if _CTX is not None:
        return _CTX

    settings: Any
    try:
        settings = get_settings()
    except Exception:
        from ai_trading.config import safe_settings

        settings = safe_settings()
    execution_feed = get_execution_feed(getattr(settings, "alpaca_execution_feed", None))
    reference_feed = get_reference_feed(getattr(settings, "alpaca_reference_feed", None))
    log_fetch = getattr(settings, "log_market_fetch", True)
    testing = getattr(settings, "testing", False)
    api_key = getattr(settings, "alpaca_api_key", None)
    try:
        secret_key = get_alpaca_secret_key_plain()
    except Exception:
        secret_key = None
    base_url = str(getattr(settings, "alpaca_base_url", "") or "")
    is_paper = "paper" in base_url.lower()

    try:  # pragma: no cover - exercised in integration tests
        from alpaca.trading.client import TradingClient  # type: ignore
    except Exception:  # pragma: no cover - client unavailable
        trading_client: AlpacaTradingClient | None = None
    else:
        try:
            trading_client = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=is_paper,
            )
        except Exception:
            trading_client = None

    try:  # pragma: no cover - exercised in integration tests
        from alpaca.data.historical.stock import StockHistoricalDataClient  # type: ignore
    except Exception:  # pragma: no cover - client unavailable
        data_client: AlpacaStockHistoricalDataClient | None = None
    else:
        try:
            data_client = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=secret_key,
            )
        except Exception:
            data_client = None

    _CTX = SimpleNamespace(
        alpaca_trading_client=trading_client,
        alpaca_data_client=data_client,
        alpaca_data_feed=execution_feed,
        alpaca_execution_feed=execution_feed,
        alpaca_reference_feed=reference_feed,
        log_market_fetch=log_fetch,
        testing=testing,
    )
    return _CTX


def __getattr__(name: str) -> Any:
    if name in {
        "BotContext",
        "LazyBotContext",
        "get_ctx",
        "ensure_alpaca_attached",
        "maybe_init_brokers",
        "init_alpaca_clients",
    }:
        module = importlib.import_module("ai_trading.core.bot_engine")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_PUBLIC_EXPORTS = (
    "BotContext",
    "LazyBotContext",
    "get_ctx",
    "ensure_alpaca_attached",
    "maybe_init_brokers",
    "init_alpaca_clients",
    "get_context",
)

__all__ = list(_PUBLIC_EXPORTS)
