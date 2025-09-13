from __future__ import annotations

"""Context utilities and lightweight singleton access.

Historically the project exposed context helpers from ``core.bot_engine``.  To
retain backwards compatibility while offering a trimmed down runtime context we
re-export the original helpers and expose :func:`get_context` for callers that
only need a handful of default attributes.
"""

from types import SimpleNamespace

try:  # pragma: no cover - bot_engine may pull in optional heavy deps
    from .bot_engine import (
        BotContext,
        LazyBotContext,
        get_ctx,
        ensure_alpaca_attached,
        maybe_init_brokers,
        init_alpaca_clients,
    )
except Exception:  # pragma: no cover - provide fallbacks when bot_engine unavailable
    BotContext = LazyBotContext = object  # type: ignore[assignment]

    def get_ctx():  # type: ignore[no-redef]
        raise RuntimeError("bot_engine unavailable")

    def ensure_alpaca_attached(*_a, **_k):  # type: ignore[no-redef]
        raise RuntimeError("bot_engine unavailable")

    def maybe_init_brokers(*_a, **_k):  # type: ignore[no-redef]
        raise RuntimeError("bot_engine unavailable")

    def init_alpaca_clients(*_a, **_k):  # type: ignore[no-redef]
        raise RuntimeError("bot_engine unavailable")


from ai_trading.settings import get_settings, get_alpaca_secret_key_plain


class MockTradingClient:
    """Fallback Alpaca trading client used when the SDK is unavailable."""

    def __getattr__(self, name: str) -> None:  # pragma: no cover - simple proxy
        raise RuntimeError("Alpaca trading client unavailable")


class MockDataClient:
    """Fallback Alpaca data client used when the SDK is unavailable."""

    def __getattr__(self, name: str) -> None:  # pragma: no cover - simple proxy
        raise RuntimeError("Alpaca data client unavailable")


_CTX: SimpleNamespace | None = None


def get_context() -> SimpleNamespace:
    """Return a singleton runtime context.

    The object includes a small set of configuration attributes and Alpaca
    client handles.  When the real Alpaca integrations cannot be constructed a
    lightweight mock client is provided instead so callers can still import the
    context without optional dependencies.
    """

    global _CTX
    if _CTX is not None:
        return _CTX

    settings = get_settings()
    feed = getattr(settings, "alpaca_data_feed", "iex")
    log_fetch = getattr(settings, "log_market_fetch", True)
    testing = getattr(settings, "testing", False)

    try:  # pragma: no cover - exercised in integration tests
        from alpaca.trading.client import TradingClient  # type: ignore

        trading_client = TradingClient(
            settings.alpaca_api_key,
            get_alpaca_secret_key_plain(),
        )
    except Exception:  # pragma: no cover - client unavailable
        trading_client = MockTradingClient()

    try:  # pragma: no cover - exercised in integration tests
        from alpaca.data.historical.stock import StockHistoricalDataClient  # type: ignore

        data_client = StockHistoricalDataClient(
            settings.alpaca_api_key,
            get_alpaca_secret_key_plain(),
        )
    except Exception:  # pragma: no cover - client unavailable
        data_client = MockDataClient()

    _CTX = SimpleNamespace(
        alpaca_trading_client=trading_client,
        alpaca_data_client=data_client,
        alpaca_data_feed=feed,
        log_market_fetch=log_fetch,
        testing=testing,
    )
    return _CTX


__all__ = [
    "BotContext",
    "LazyBotContext",
    "get_ctx",
    "ensure_alpaca_attached",
    "maybe_init_brokers",
    "init_alpaca_clients",
    "get_context",
    "MockTradingClient",
    "MockDataClient",
]
