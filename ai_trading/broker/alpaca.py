from __future__ import annotations
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any
from ai_trading.alpaca_api import ALPACA_AVAILABLE
from ai_trading.logging import get_logger
from ai_trading.utils.optional_import import optional_import
from ai_trading.utils.retry import retry_call
from .alpaca_credentials import resolve_alpaca_credentials
try:
    from requests.exceptions import HTTPError
except ImportError:

    class HTTPError(Exception):
        pass
APIError = optional_import('alpaca.common.exceptions', 'APIError') or Exception
TradingClient = optional_import('alpaca.trading.client', 'TradingClient')
if TradingClient is None:
    _tmp_logger = get_logger(__name__)
    _tmp_logger.warning('VENDOR_MISSING: alpaca SDK not installed; using REST fallback/offline path')
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
    """
    Thin compatibility layer over:
      - new SDK: `alpaca-py` (`alpaca.trading.TradingClient`)
      - old SDK: `alpaca_trade_api.REST`

    Exposes a stable, minimal surface for the rest of the bot:
      - list_open_orders()
      - list_orders
      - cancel_order(order_id)
      - cancel_all_orders()
      - list_open_positions()
      - get_account()
      - submit_order  (market/limit/stop/stop_limit supported)

    All methods return the SDK-native objects (no shape conversion),
    to minimize downstream changes.
    """

    def __init__(self, raw_api: Any):
        self._api = raw_api
        self._is_new = TradingClient is not None and isinstance(raw_api, TradingClient)
        self._GetOrdersRequest = None
        self._QueryOrderStatus = None
        self._OrderSide = None
        self._OrderType = None
        self._TimeInForce = None

    def _new_imports(self):
        if not self._is_new:
            return
        if self._GetOrdersRequest is None:
            from alpaca.trading.enums import OrderSide, OrderType, QueryOrderStatus, TimeInForce
            from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest, StopLimitOrderRequest, StopOrderRequest
            self._GetOrdersRequest = GetOrdersRequest
            self._MarketOrderRequest = MarketOrderRequest
            self._LimitOrderRequest = LimitOrderRequest
            self._StopOrderRequest = StopOrderRequest
            self._StopLimitOrderRequest = StopLimitOrderRequest
            self._QueryOrderStatus = QueryOrderStatus
            self._OrderSide = OrderSide
            self._OrderType = OrderType
            self._TimeInForce = TimeInForce

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
        """
        Return open orders.
        """
        if self._is_new:
            self._new_imports()
            req = self._GetOrdersRequest(status=self._QueryOrderStatus.OPEN)
            return self._api.get_orders(req)
        try:
            return self._api.list_orders(status='open')
        except AttributeError:
            raise RuntimeError('Alpaca API has neither get_orders nor list_orders')

    def list_orders(self, status: str | None=None, limit: int | None=None) -> Iterable[Any]:
        """
        Generic order listing with optional status and limit.
        status: 'open'|'closed'|'all' (old SDK words) or maps to new enums.
        """
        if self._is_new:
            self._new_imports()
            kwargs: dict[str, Any] = dict()
            if status:
                map_status = {'open': self._QueryOrderStatus.OPEN, 'closed': self._QueryOrderStatus.CLOSED, 'all': self._QueryOrderStatus.ALL}.get(status.lower(), self._QueryOrderStatus.ALL)
                kwargs['status'] = map_status
            if limit:
                kwargs['limit'] = limit
            req = self._GetOrdersRequest(**kwargs)
            return self._api.get_orders(req)
        return self._api.list_orders(status=status or 'all', limit=limit)

    def cancel_order(self, order_id: str) -> Any:
        if self._is_new:
            return self._call_with_retry('cancel_order_by_id', lambda: self._api.cancel_order_by_id(order_id))
        return self._call_with_retry('cancel_order', lambda: self._api.cancel_order(order_id))

    def cancel_all_orders(self) -> Any:
        if self._is_new:
            return self._api.cancel_orders()
        return self._api.cancel_all_orders()

    def get_order_by_id(self, order_id: str) -> Any:
        if self._is_new:
            return self._api.get_order_by_id(order_id)
        try:
            return self._api.get_order(order_id)
        except AttributeError:
            return self._api.get_order_by_id(order_id)

    def list_open_positions(self) -> list[Any]:
        """Return open positions as objects with symbol/qty/avg_entry_price."""
        try:
            if self._is_new:
                raw = self._call_with_retry('get_all_positions', self._api.get_all_positions)
            else:
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
            if hasattr(self._api, 'get_open_position'):
                p = self._api.get_open_position(symbol)
            elif hasattr(self._api, 'get_position'):
                p = self._api.get_position(symbol)
            else:
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
        if self._is_new:
            return self._call_with_retry('get_account', self._api.get_account)
        return self._call_with_retry('get_account', self._api.get_account)

    def submit_order(self, **kwargs) -> Any:
        """Submit an order with retry handling."""
        if self._is_new:
            self._new_imports()
            side = self._OrderSide(kwargs['side'].upper())
            tif = self._TimeInForce(kwargs.get('time_in_force', 'day').upper())
            order_type = kwargs.get('type', 'market').lower()
            req_cls = {'market': self._MarketOrderRequest, 'limit': self._LimitOrderRequest, 'stop': self._StopOrderRequest, 'stop_limit': self._StopLimitOrderRequest}.get(order_type)
            if req_cls is None:
                raise ValueError(f'unsupported order type: {order_type}')
            req_kwargs = {'symbol': kwargs['symbol'], 'qty': kwargs['qty'], 'side': side, 'time_in_force': tif, 'client_order_id': kwargs.get('client_order_id')}
            if order_type in {'limit', 'stop_limit'}:
                req_kwargs['limit_price'] = kwargs.get('limit_price')
            if order_type in {'stop', 'stop_limit'}:
                req_kwargs['stop_price'] = kwargs.get('stop_price')
            req = req_cls(**req_kwargs)
            return self._call_with_retry('submit_order', lambda: self._api.submit_order(req))
        return self._call_with_retry('submit_order', lambda: self._api.submit_order(**kwargs))

def initialize(api_key: str | None=None, secret_key: str | None=None, base_url: str | None=None, **kwargs) -> AlpacaBroker | None:
    """Create an :class:`AlpacaBroker` if the SDK is available."""
    if not (ALPACA_AVAILABLE and TradingClient):
        return None
    if api_key is None or secret_key is None or base_url is None:
        creds = resolve_alpaca_credentials()
        api_key = api_key or creds.API_KEY
        secret_key = secret_key or creds.SECRET_KEY
        base_url = base_url or creds.BASE_URL
    paper = 'paper' in (base_url or '')
    client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper, **kwargs)
    return AlpacaBroker(client)
__all__ = ['TradingClient', 'AlpacaBroker']