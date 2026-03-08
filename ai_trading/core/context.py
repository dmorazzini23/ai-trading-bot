from __future__ import annotations

"""Context utilities and lightweight singleton access.

Historically the project exposed context helpers from ``core.bot_engine``.  To
retain backwards compatibility while offering a trimmed down runtime context we
re-export the original helpers and expose :func:`get_context` for callers that
only need a handful of default attributes.
"""

from types import SimpleNamespace
from typing import Any, cast

BotContext: type[Any]
LazyBotContext: type[Any]

try:  # pragma: no cover - bot_engine may pull in optional heavy deps
    from .bot_engine import (
        BotContext as _BotContext,
        LazyBotContext as _LazyBotContext,
        get_ctx,
        ensure_alpaca_attached,
        maybe_init_brokers,
        init_alpaca_clients,
    )
    BotContext = _BotContext
    LazyBotContext = _LazyBotContext
except Exception:  # pragma: no cover - provide fallbacks when bot_engine unavailable
    class _FallbackBotContext(SimpleNamespace):
        """Fallback BotContext used when bot_engine import fails."""

    class _FallbackLazyBotContext(SimpleNamespace):
        """Fallback LazyBotContext used when bot_engine import fails."""

    BotContext = _FallbackBotContext
    LazyBotContext = _FallbackLazyBotContext

    def _unavailable(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("bot_engine unavailable")

    get_ctx = cast(Any, _unavailable)
    ensure_alpaca_attached = cast(Any, _unavailable)
    maybe_init_brokers = cast(Any, _unavailable)
    init_alpaca_clients = cast(Any, _unavailable)


from ai_trading.settings import get_settings, get_alpaca_secret_key_plain


class UnavailableTradingClient:
    """Placeholder trading client that fails fast when Alpaca is unavailable."""

    def __init__(self, *_, paper: bool | None = None, **__):
        self.paper = paper

    def __getattr__(self, name: str) -> None:  # pragma: no cover - simple proxy
        raise RuntimeError("Alpaca trading client unavailable")


class UnavailableDataClient:
    """Placeholder data client that fails fast when Alpaca is unavailable."""

    def __init__(self, *_, paper: bool | None = None, **__):
        self.paper = paper

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
    api_key = getattr(settings, "alpaca_api_key", None)
    secret_key = get_alpaca_secret_key_plain()
    base_url = str(getattr(settings, "alpaca_base_url", "") or "")
    is_paper = "paper" in base_url.lower()

    try:  # pragma: no cover - exercised in integration tests
        from alpaca.trading.client import TradingClient  # type: ignore
    except Exception:  # pragma: no cover - client unavailable
        trading_client = UnavailableTradingClient(paper=is_paper)
    else:
        try:
            trading_client = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=is_paper,
            )
        except Exception:
            trading_client = UnavailableTradingClient(paper=is_paper)

    try:  # pragma: no cover - exercised in integration tests
        from alpaca.data.historical.stock import StockHistoricalDataClient  # type: ignore
    except Exception:  # pragma: no cover - client unavailable
        data_client = UnavailableDataClient(paper=is_paper)
    else:
        try:
            data_client = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=is_paper,
            )
        except TypeError:
            try:
                data_client = StockHistoricalDataClient.__call__(
                    StockHistoricalDataClient,
                    api_key=api_key,
                    secret_key=secret_key,
                )
            except Exception:
                data_client = UnavailableDataClient(paper=is_paper)
        except Exception:
            data_client = UnavailableDataClient(paper=is_paper)

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
    "UnavailableTradingClient",
    "UnavailableDataClient",
]
