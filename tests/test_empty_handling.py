import types

import pandas as pd
import pytest

from ai_trading.data.fetch.empty_handling import fetch_with_retries, _RETRY_COUNTS
from ai_trading.data.fetch import EmptyBarsError


class _Raiser:
    def __init__(self, fail_times: int, result: pd.DataFrame | None = None) -> None:
        self.fail_times = fail_times
        self.calls = 0
        self.result = result if result is not None else pd.DataFrame({'a': [1]})

    def __call__(self):
        if self.calls < self.fail_times:
            self.calls += 1
            raise EmptyBarsError('empty')
        self.calls += 1
        return self.result


def test_returns_empty_when_no_retry_delays(monkeypatch):
    symbol = 'AAPL'
    timeframe = '1Min'
    raiser = _Raiser(fail_times=1)
    monkeypatch.setattr('ai_trading.data.fetch.empty_handling.is_market_open', lambda: True)
    called: dict[str, float] = {}
    monkeypatch.setattr(
        'ai_trading.data.fetch.empty_handling.time',
        types.SimpleNamespace(sleep=lambda s: called.setdefault('sleep', s)),
    )
    df = fetch_with_retries(symbol, timeframe, raiser, [])
    assert df.empty
    assert 'sleep' not in called
    assert (symbol, timeframe) not in _RETRY_COUNTS


def test_retry_until_success(monkeypatch):
    symbol = 'MSFT'
    timeframe = '1Min'
    raiser = _Raiser(fail_times=1)
    monkeypatch.setattr('ai_trading.data.fetch.empty_handling.is_market_open', lambda: True)
    called: dict[str, float] = {}
    monkeypatch.setattr(
        'ai_trading.data.fetch.empty_handling.time',
        types.SimpleNamespace(sleep=lambda s: called.setdefault('sleep', s)),
    )
    df = fetch_with_retries(symbol, timeframe, raiser, [0])
    assert not df.empty
    assert called.get('sleep') == 0
    assert raiser.calls == 2
    assert (symbol, timeframe) not in _RETRY_COUNTS


def test_abort_when_market_closed(monkeypatch):
    symbol = 'IBM'
    timeframe = '1Min'
    raiser = _Raiser(fail_times=1)
    monkeypatch.setattr('ai_trading.data.fetch.empty_handling.is_market_open', lambda: False)
    called: dict[str, float] = {}
    monkeypatch.setattr(
        'ai_trading.data.fetch.empty_handling.time',
        types.SimpleNamespace(sleep=lambda s: called.setdefault('sleep', s)),
    )
    delays = [0, 1]
    df = fetch_with_retries(symbol, timeframe, raiser, delays)
    assert df.empty
    assert delays == [0, 1]
    assert 'sleep' not in called
    assert (symbol, timeframe) not in _RETRY_COUNTS


def test_raise_when_window_has_no_trading_session(monkeypatch):
    symbol = 'TSLA'
    timeframe = '5Min'
    raiser = _Raiser(fail_times=3)
    retry_delays = [0.25, 0.5]
    monkeypatch.setattr('ai_trading.data.fetch.empty_handling.is_market_open', lambda: True)
    sleep_called = {'count': 0}

    def _sleep(_s: float) -> None:
        sleep_called['count'] += 1

    monkeypatch.setattr(
        'ai_trading.data.fetch.empty_handling.time',
        types.SimpleNamespace(sleep=_sleep),
    )

    calls = {'check': 0}

    def _has_session() -> bool:
        calls['check'] += 1
        return False

    with pytest.raises(EmptyBarsError):
        fetch_with_retries(
            symbol,
            timeframe,
            raiser,
            retry_delays.copy(),
            window_has_trading_session=_has_session,
        )

    assert raiser.calls == 1
    assert calls['check'] == 1
    assert (symbol, timeframe) not in _RETRY_COUNTS
    assert retry_delays == [0.25, 0.5]
    assert sleep_called['count'] == 0

    success = fetch_with_retries(
        symbol,
        timeframe,
        _Raiser(fail_times=0),
        [0.1],
        window_has_trading_session=lambda: True,
    )
    assert not success.empty
    assert (symbol, timeframe) not in _RETRY_COUNTS
