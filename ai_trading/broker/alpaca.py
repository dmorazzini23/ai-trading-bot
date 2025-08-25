from __future__ import annotations
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any
from ai_trading.alpaca_api import ALPACA_AVAILABLE
from ai_trading.logging import get_logger
from ai_trading.utils.optdeps import optional_import
from ai_trading.utils.retry import retry_call
from .alpaca_credentials import resolve_alpaca_credentials
try:
    from requests.exceptions import HTTPError
except ImportError:

    class HTTPError(Exception):
        pass
APIError = optional_import('alpaca_trade_api.rest', attr='APIError') or Exception
REST = optional_import('alpaca_trade_api.rest', attr='REST')
TradingClient = None  # backwards compatibility symbol
if REST is None:
    _tmp_logger = get_logger(__name__)
    _tmp_logger.warning('VENDOR_MISSING: alpaca-trade-api not installed; using REST fallback/offline path')
from ai_trading.exc import TRANSIENT_HTTP_EXC
_log = get_logger(__name__)
SAFE_EXC = TRANSIENT_HTTP_EXC + (APIError,)

def _retry_config() -> tuple[int, float, float, float]:
    """Load retry knobs from settings if available."""
    retries, backoff, max_backoff, jitter = (3, 0.1, 2.0, 0.1)
    try:
        from ai_trading.config import get_settings
        s = get_settings()
        retries = int(getattr(s, 'RETRY_MAX_ATTEMPTS', retries))
        backoff = float(getattr(s, 'RETRY_BASE_DELAY', backoff))
        max_backoff = float(getattr(s, 'RETRY_MAX_DELAY', max_backoff))
        jitter = float(getattr(s, 'RETRY_JITTER', jitter))
    except (AttributeError, TypeError, ValueError, ImportError):
        pass
    return (retries, backoff, max_backoff, jitter)

class AlpacaBroker:
    """Minimal wrapper around ``alpaca_trade_api.REST`` operations."""

    def __init__(self, raw_api: Any):
        self._api = raw_api

    def _call_with_retry(self, op: str, func: Callable[..., Any]) -> Any:
        retries, backoff, max_backoff, jitter = _retry_config()
        attempt = 0

        def _wrapped() -> Any:
            nonlocal attempt
            try:
                return func()
            except SAFE_EXC as e:
                attempt += 1
                _log.warning('ALPACA_RETRY', extra={'op': op, 'attempt': attempt, 'attempts': retries, 'error': str(e)})
                raise
        return retry_call(_wrapped, exceptions=SAFE_EXC, retries=retries, backoff=backoff, max_backoff=max_backoff, jitter=jitter)

    def list_open_orders(self) -> Iterable[Any]:
        """Return open orders."""
        try:
            return self._api.list_orders(status='open')
        except AttributeError:  # pragma: no cover
            raise RuntimeError('Alpaca API has neither get_orders nor list_orders')

    def list_orders(self, status: str | None=None, limit: int | None=None) -> Iterable[Any]:
        """Generic order listing with optional status and limit."""
        kwargs: dict[str, Any] = {}
        if status:
            kwargs['status'] = status
        if limit is not None:
            kwargs['limit'] = limit
        return self._call_with_retry('list_orders', lambda: self._api.list_orders(**kwargs))

    def cancel_order(self, order_id: str) -> Any:
        return self._call_with_retry('cancel_order', lambda: self._api.cancel_order(order_id))

    def cancel_all_orders(self) -> Any:
        return self._api.cancel_all_orders()

    def get_order_by_id(self, order_id: str) -> Any:
        try:
            return self._api.get_order(order_id)
        except AttributeError:
            return self._api.get_order_by_id(order_id)

    def list_open_positions(self) -> list[Any]:
        """Return open positions as objects with symbol/qty/avg_entry_price."""
        try:
            raw = self._call_with_retry('list_positions', self._api.list_positions)
        except AttributeError:
            raise RuntimeError('Alpaca API has neither get_all_positions nor list_positions')
        except SAFE_EXC as e:
            _log.warning('list_open_positions failed: %s', e, exc_info=True)
            return []
        positions: list[Any] = []
        for p in raw or []:
            try:
                sym = p.symbol if hasattr(p, 'symbol') else p['symbol']
                qty = int(p.qty) if hasattr(p, 'qty') else int(p['qty'])
                aep = float(p.avg_entry_price) if hasattr(p, 'avg_entry_price') else float(p['avg_entry_price'])
                positions.append(SimpleNamespace(symbol=sym, qty=qty, avg_entry_price=aep))
            except (KeyError, AttributeError, TypeError, ValueError):
                continue
        return positions

    def get_open_position(self, symbol: str) -> Any | None:
        """Return position for ``symbol`` or None."""
        try:
            if hasattr(self._api, 'get_position'):
                p = self._api.get_position(symbol)
            elif hasattr(self._api, 'get_open_position'):  # pragma: no cover
                p = self._api.get_open_position(symbol)
            else:  # pragma: no cover
                raise AttributeError
            sym = p.symbol if hasattr(p, 'symbol') else p['symbol']
            qty = int(p.qty) if hasattr(p, 'qty') else int(p['qty'])
            aep = float(p.avg_entry_price) if hasattr(p, 'avg_entry_price') else float(p['avg_entry_price'])
            return SimpleNamespace(symbol=sym, qty=qty, avg_entry_price=aep)
        except AttributeError:
            _log.warning('ALPACA_SDK_MISSING_METHOD', extra={'method': 'get_position'})
        except (APIError, HTTPError) as e:
            msg = str(e).lower()
            if '404' not in msg and 'not found' not in msg:
                _log.warning('ALPACA_SDK_ERROR', extra={'error': msg})
            else:
                return None
        session = getattr(self._api, '_session', None)
        if session is None:
            return None
        try:
            resp = session.get(f'/v2/positions/{symbol}')
            if getattr(resp, 'status_code', 0) == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            sym = data.get('symbol')
            qty = int(data.get('qty'))
            aep = float(data.get('avg_entry_price'))
            return SimpleNamespace(symbol=sym, qty=qty, avg_entry_price=aep)
        except HTTPError as e:
            if getattr(e.response, 'status_code', None) == 404:
                return None
            _log.warning('ALPACA_REST_ERROR', extra={'error': str(e)})
        except (KeyError, TypeError, ValueError, AttributeError):
            _log.warning('ALPACA_REST_PARSE_ERROR')
        return None
    get_position = get_open_position

    def get_account(self) -> Any:
        return self._call_with_retry('get_account', self._api.get_account)

    def submit_order(self, **kwargs) -> Any:
        """Submit an order with retry handling."""
        return self._call_with_retry('submit_order', lambda: self._api.submit_order(**kwargs))

def initialize(api_key: str | None=None, secret_key: str | None=None, base_url: str | None=None, **kwargs) -> AlpacaBroker | None:
    """Create an :class:`AlpacaBroker` if the SDK is available."""
    if not (ALPACA_AVAILABLE and REST):
        return None
    if api_key is None or secret_key is None or base_url is None:
        creds = resolve_alpaca_credentials()
        api_key = api_key or creds.API_KEY
        secret_key = secret_key or creds.SECRET_KEY
        base_url = base_url or creds.BASE_URL
    client = REST(api_key, secret_key, base_url, **kwargs)
    return AlpacaBroker(client)
__all__ = ['TradingClient', 'AlpacaBroker']