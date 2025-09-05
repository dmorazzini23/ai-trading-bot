from json import JSONDecodeError
from unittest.mock import Mock, patch

import pytest
from ai_trading.exc import HTTPError, RequestException

from ai_trading.position_sizing import _CACHE, _get_equity_from_alpaca


class _Cfg:
    alpaca_base_url = "https://api.example.com"
    alpaca_api_key = "key"
    alpaca_secret_key_plain = "secret"


def _cfg():
    return _Cfg()


def test_get_equity_http_error_returns_zero():
    cfg = _cfg()
    _CACHE.equity = 123.0
    resp = Mock()
    resp.raise_for_status.side_effect = HTTPError(response=Mock(status_code=500))
    session = Mock(get=Mock(return_value=resp))
    with patch("ai_trading.position_sizing.get_global_session", return_value=session):
        assert _get_equity_from_alpaca(cfg, force_refresh=True) == 0.0
    assert _CACHE.equity == 0.0


def test_get_equity_request_exception_returns_zero():
    cfg = _cfg()
    _CACHE.equity = 456.0
    session = Mock()
    session.get.side_effect = RequestException("boom")
    with patch("ai_trading.position_sizing.get_global_session", return_value=session):
        assert _get_equity_from_alpaca(cfg, force_refresh=True) == 0.0
    assert _CACHE.equity == 0.0


def test_get_equity_invalid_json_returns_zero():
    cfg = _cfg()
    _CACHE.equity = 789.0
    resp = Mock()
    resp.raise_for_status.return_value = None
    resp.json.side_effect = JSONDecodeError("no json", "{}", 0)
    session = Mock(get=Mock(return_value=resp))
    with patch("ai_trading.position_sizing.get_global_session", return_value=session):
        assert _get_equity_from_alpaca(cfg, force_refresh=True) == 0.0
    assert _CACHE.equity == 0.0


def test_get_equity_unexpected_exception_propagates():
    cfg = _cfg()
    _CACHE.equity = 42.0
    with patch(
        "ai_trading.position_sizing.get_global_session", side_effect=RuntimeError("boom")
    ):
        with pytest.raises(RuntimeError):
            _get_equity_from_alpaca(cfg, force_refresh=True)
    assert _CACHE.equity is None
