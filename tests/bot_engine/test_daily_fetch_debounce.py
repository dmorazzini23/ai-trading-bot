"""Daily fetch memo ensures repeated calls reuse cached data."""

from __future__ import annotations

from datetime import UTC, datetime
import sys
import types

import pytest
from ai_trading.data.fetch.normalize import normalize_ohlcv_df
from ai_trading.utils.lazy_imports import load_pandas

pytest.importorskip("pandas")

if "ai_trading.indicators" not in sys.modules:
    indicators_stub = types.ModuleType("ai_trading.indicators")

    def _unavailable_indicator(*_args, **_kwargs):  # pragma: no cover - safety stub
        raise RuntimeError("Indicator module unavailable in tests")

    indicators_stub.compute_atr = _unavailable_indicator
    indicators_stub.atr = _unavailable_indicator
    indicators_stub.mean_reversion_zscore = _unavailable_indicator
    indicators_stub.rsi = _unavailable_indicator
    sys.modules["ai_trading.indicators"] = indicators_stub

if "ai_trading.signals" not in sys.modules:
    signals_stub = types.ModuleType("ai_trading.signals")
    signals_indicators_stub = types.ModuleType("ai_trading.signals.indicators")

    def _composite_confidence_stub(*_args, **_kwargs):  # pragma: no cover - safety stub
        return {}

    signals_indicators_stub.composite_signal_confidence = _composite_confidence_stub
    sys.modules["ai_trading.signals"] = signals_stub
    sys.modules["ai_trading.signals.indicators"] = signals_indicators_stub
    signals_stub.indicators = signals_indicators_stub

if "ai_trading.features" not in sys.modules:
    features_stub = types.ModuleType("ai_trading.features")
    features_indicators_stub = types.ModuleType("ai_trading.features.indicators")

    def _feature_passthrough(df, **_kwargs):  # pragma: no cover - safety stub
        return df

    features_indicators_stub.compute_macd = _feature_passthrough
    features_indicators_stub.compute_macds = _feature_passthrough
    features_indicators_stub.compute_vwap = _feature_passthrough
    features_indicators_stub.compute_atr = _feature_passthrough
    features_indicators_stub.compute_sma = _feature_passthrough
    features_indicators_stub.ensure_columns = _feature_passthrough
    sys.modules["ai_trading.features"] = features_stub
    sys.modules["ai_trading.features.indicators"] = features_indicators_stub
    features_stub.indicators = features_indicators_stub

if "portalocker" not in sys.modules:
    portalocker_stub = types.ModuleType("portalocker")
    portalocker_stub.LOCK_EX = 1

    def _noop_lock(*_args, **_kwargs):  # pragma: no cover - safety stub
        return None

    portalocker_stub.lock = _noop_lock
    portalocker_stub.unlock = _noop_lock
    sys.modules["portalocker"] = portalocker_stub

if "bs4" not in sys.modules:
    bs4_stub = types.ModuleType("bs4")

    class _BeautifulSoup:  # pragma: no cover - safety stub
        def __init__(self, *_args, **_kwargs):
            self.text = ""

        def find(self, *_args, **_kwargs):
            return None

    bs4_stub.BeautifulSoup = _BeautifulSoup
    sys.modules["bs4"] = bs4_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = dotenv_stub

if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.array = lambda *a, **k: a
    numpy_stub.ndarray = object
    numpy_stub.float64 = float
    numpy_stub.int64 = int
    numpy_stub.nan = float("nan")
    numpy_stub.NaN = float("nan")
    numpy_stub.random = types.SimpleNamespace(seed=lambda *_a, **_k: None)
    sys.modules["numpy"] = numpy_stub

from ai_trading.core import bot_engine as be


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: D401 - match datetime API
        base = datetime(2024, 1, 3, 12, 0, tzinfo=UTC)
        if tz is None:
            return base.replace(tzinfo=None)
        return base.astimezone(tz)


def _stub_fetcher(monkeypatch) -> be.DataFetcher:
    monkeypatch.setattr(be.DataFetcher, "__post_init__", lambda self: None)
    fetcher = be.DataFetcher()
    fetcher.settings = types.SimpleNamespace(
        alpaca_api_key="key",
        alpaca_secret_key_plain="secret",
        data_feed="iex",
        alpaca_data_feed=None,
    )
    fetcher._daily_cache = {}
    fetcher._daily_cache_hit_logged = False
    fetcher._warn_seen = {}
    fetcher._warn_once = lambda *a, **k: None
    fetcher._prepare_daily_dataframe = lambda df, symbol: df
    return fetcher


def test_daily_fetch_memo_reuses_recent_result(monkeypatch):
    assert hasattr(be, "DataFetcher")
    fetcher = _stub_fetcher(monkeypatch)
    symbol = "AAPL"

    monkeypatch.setattr(be, "datetime", FixedDateTime)
    monkeypatch.setattr(be, "is_market_open", lambda: True)
    be.daily_cache_hit = None
    be.daily_cache_miss = None
    monkeypatch.setattr(
        be,
        "bars",
        types.SimpleNamespace(TimeFrame=types.SimpleNamespace(Day="Day")),
        raising=False,
    )

    monotonic_values = iter([10.0, 11.0, 120.0, 121.0, 130.0, 140.0, 150.0])
    monkeypatch.setattr(be.time, "monotonic", lambda: next(monotonic_values))

    fetch_date = FixedDateTime.now(UTC).date()
    memo_key = (symbol, fetch_date.isoformat())

    memo_df = {"memo": True}
    cached_df = {"cached": True}

    be._DAILY_FETCH_MEMO_TTL = 60.0
    be._DAILY_FETCH_MEMO = {memo_key: (0.0, memo_df)}
    fetcher._daily_cache[symbol] = (fetch_date, cached_df)

    first = fetcher.get_daily_df(types.SimpleNamespace(), symbol)
    assert first is memo_df

    # Simulate memo expiry and verify the cached entry refreshes memo storage
    be._DAILY_FETCH_MEMO[memo_key] = (0.0, memo_df)
    second = fetcher.get_daily_df(types.SimpleNamespace(), symbol)
    assert second is cached_df
    assert memo_key in be._DAILY_FETCH_MEMO
    assert be._DAILY_FETCH_MEMO[memo_key][1] is cached_df


def test_daily_fetch_memo_handles_generator_factory():
    from ai_trading.data import fetch as fetch_module

    fetch_module._daily_memo.clear()

    key = ("GEN", "2024-01-01")
    call_count = 0

    def _generator_factory():
        nonlocal call_count
        call_count += 1
        yield {"call": call_count}

    first = fetch_module.daily_fetch_memo(key, _generator_factory)
    assert first == {"call": 1}
    assert key in fetch_module._daily_memo
    assert fetch_module._daily_memo[key][1] == {"call": 1}
    assert not isinstance(fetch_module._daily_memo[key][1], types.GeneratorType)
    assert call_count == 1

    second = fetch_module.daily_fetch_memo(key, _generator_factory)
    assert second == first
    assert call_count == 1

    empty_key = ("GEN", "empty")

    def _empty_generator():
        if False:  # pragma: no cover - generator needs a yield
            yield None

    empty = fetch_module.daily_fetch_memo(empty_key, _empty_generator)
    assert empty is None
    assert empty_key not in fetch_module._daily_memo

    fetch_module._daily_memo.clear()


def test_daily_fetch_memo_handles_runtime_error_generator(monkeypatch):
    from ai_trading.data import fetch as fetch_module

    fetch_module._daily_memo.clear()

    key = ("GEN", "runtime")
    times = [0.0]

    def _fake_time():
        return times[0]

    monkeypatch.setattr(fetch_module.time, "time", _fake_time)

    call_count = 0

    def _generator_factory():
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            yield {"call": call_count}
            return

        if False:  # pragma: no cover - generator marker
            yield None

        raise StopIteration

    first = fetch_module.daily_fetch_memo(key, _generator_factory)
    assert first == {"call": 1}
    assert call_count == 1
    assert key in fetch_module._daily_memo

    times[0] = fetch_module._DAILY_TTL_S + 0.1

    second = fetch_module.daily_fetch_memo(key, _generator_factory)
    assert second is None
    assert call_count == 2
    assert key not in fetch_module._daily_memo

    fetch_module._daily_memo.clear()


def test_get_daily_df_memoized_reuses_generator_result(monkeypatch):
    from ai_trading.core import daily_fetch_memo as memo_module

    memo_module.clear_memo()

    pd = load_pandas()
    index = pd.DatetimeIndex(["2024-01-01"], tz="UTC", name="timestamp")
    raw_df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [1000],
        },
        index=index,
    )

    call_count = 0

    def _generator_factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        assert args == ("AAPL", "1Day", "2024-01-01", "2024-01-02")
        assert kwargs == {}
        yield raw_df

    first = memo_module.get_daily_df_memoized(
        "AAPL", "1Day", "2024-01-01", "2024-01-02", _generator_factory
    )
    pd.testing.assert_frame_equal(first, normalize_ohlcv_df(raw_df))
    assert call_count == 1

    second = memo_module.get_daily_df_memoized(
        "AAPL", "1Day", "2024-01-01", "2024-01-02", _generator_factory
    )
    assert second is first
    assert call_count == 1

    memo_module.clear_memo()


def test_get_daily_df_memoized_retry_on_stop_iteration(monkeypatch):
    from ai_trading.core import daily_fetch_memo as memo_module

    memo_module.clear_memo()

    pd = load_pandas()
    index = pd.DatetimeIndex(["2024-01-01"], tz="UTC", name="timestamp")
    raw_df = pd.DataFrame(
        {
            "open": [3.0],
            "high": [4.0],
            "low": [2.5],
            "close": [3.5],
            "volume": [2000],
        },
        index=index,
    )

    times = [0.0]

    def _fake_time():
        return times[0]

    monkeypatch.setattr(memo_module.time, "time", _fake_time)

    call_count = 0

    def _generator_factory(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield raw_df
            return
        if False:  # pragma: no cover - generator marker for subsequent calls
            yield None
        return

    expected = memo_module.get_daily_df_memoized(
        "MSFT", "1Day", "2024-01-01", "2024-01-02", _generator_factory
    )
    pd.testing.assert_frame_equal(expected, normalize_ohlcv_df(raw_df))
    assert call_count == 1

    times[0] = memo_module._TTL_S + 0.5
    retry = memo_module.get_daily_df_memoized(
        "MSFT", "1Day", "2024-01-01", "2024-01-02", _generator_factory
    )
    assert call_count == 3
    assert retry is expected

    third = memo_module.get_daily_df_memoized(
        "MSFT", "1Day", "2024-01-01", "2024-01-02", _generator_factory
    )
    assert third is expected
    assert call_count == 3

    memo_module.clear_memo()


def test_get_daily_df_memoized_runtime_error_reuses_last_good(monkeypatch):
    from ai_trading.core import daily_fetch_memo as memo_module

    memo_module.clear_memo()

    pd = load_pandas()
    index = pd.DatetimeIndex(["2024-01-01"], tz="UTC", name="timestamp")
    raw_df = pd.DataFrame(
        {
            "open": [5.0],
            "high": [6.0],
            "low": [4.5],
            "close": [5.5],
            "volume": [3000],
        },
        index=index,
    )

    times = [0.0]

    def _fake_time():
        return times[0]

    monkeypatch.setattr(memo_module.time, "time", _fake_time)

    call_count = 0

    def _generator_factory(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield raw_df
            return

        if False:  # pragma: no cover - generator marker
            yield None

        raise StopIteration

    expected = memo_module.get_daily_df_memoized(
        "GOOG", "1Day", "2024-01-01", "2024-01-02", _generator_factory
    )
    pd.testing.assert_frame_equal(expected, normalize_ohlcv_df(raw_df))
    assert call_count == 1

    times[0] = memo_module._TTL_S + 0.5

    retry = memo_module.get_daily_df_memoized(
        "GOOG", "1Day", "2024-01-01", "2024-01-02", _generator_factory
    )
    assert retry is expected
    assert call_count == 3

    memo_module.clear_memo()


def test_get_daily_df_memoized_stop_iteration_preserves_cache(monkeypatch):
    from ai_trading.core import daily_fetch_memo as memo_module

    memo_module.clear_memo()

    pd = load_pandas()
    index = pd.DatetimeIndex(["2024-02-01"], tz="UTC", name="timestamp")
    raw_df = pd.DataFrame(
        {
            "open": [7.0],
            "high": [8.0],
            "low": [6.5],
            "close": [7.5],
            "volume": [4000],
        },
        index=index,
    )

    times = [0.0]

    def _fake_time():
        return times[0]

    monkeypatch.setattr(memo_module.time, "time", _fake_time)

    call_count = 0

    def _generator_factory(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield raw_df
            return
        if False:  # pragma: no cover - generator marker
            yield None
        raise StopIteration

    key = ("TSLA", "1Day", "2024-02-01", "2024-02-02")

    first = memo_module.get_daily_df_memoized(*key, _generator_factory)
    pd.testing.assert_frame_equal(first, normalize_ohlcv_df(raw_df))
    assert call_count == 1

    times[0] = memo_module._TTL_S + 1.0

    second = memo_module.get_daily_df_memoized(*key, _generator_factory)
    assert second is first
    assert call_count == 3
    assert memo_module._MEMO[key].ts == times[0]

    times[0] += 0.5

    third = memo_module.get_daily_df_memoized(*key, _generator_factory)
    assert third is first
    assert call_count == 3

    memo_module.clear_memo()


def test_get_daily_df_memoized_none_result_reuses_last_good(monkeypatch):
    from ai_trading.core import daily_fetch_memo as memo_module

    memo_module.clear_memo()

    pd = load_pandas()
    index = pd.DatetimeIndex(["2024-03-01"], tz="UTC", name="timestamp")
    raw_df = pd.DataFrame(
        {
            "open": [9.0],
            "high": [10.0],
            "low": [8.5],
            "close": [9.5],
            "volume": [5000],
        },
        index=index,
    )

    times = [0.0]

    def _fake_time():
        return times[0]

    monkeypatch.setattr(memo_module.time, "time", _fake_time)

    call_count = 0

    def _factory(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return raw_df
        if call_count == 2:
            return None
        return raw_df

    key = ("NFLX", "1Day", "2024-03-01", "2024-03-02")

    first = memo_module.get_daily_df_memoized(*key, _factory)
    pd.testing.assert_frame_equal(first, normalize_ohlcv_df(raw_df))
    assert call_count == 1

    times[0] = memo_module._TTL_S + 2.0

    second = memo_module.get_daily_df_memoized(*key, _factory)
    assert second is first
    assert call_count == 2
    assert memo_module._MEMO[key].ts == times[0]

    times[0] += 0.25

    third = memo_module.get_daily_df_memoized(*key, _factory)
    assert third is first
    assert call_count == 2

    memo_module.clear_memo()


def test_daily_missing_columns_error_sticky(monkeypatch):
    fetcher = _stub_fetcher(monkeypatch)
    symbol = "AAPL"

    monkeypatch.setattr(be, "datetime", FixedDateTime)
    monkeypatch.setattr(be, "is_market_open", lambda: True)
    be.daily_cache_hit = None
    be.daily_cache_miss = None
    be._DAILY_FETCH_MEMO.clear()
    fetcher._daily_cache.clear()
    fetcher._daily_error_state.clear()

    monotonic_values = iter([10.0, 11.0, 12.0, 130.0])
    monkeypatch.setattr(be, "monotonic_time", lambda: next(monotonic_values))

    error_calls = []

    def _raise_missing(*_args, **_kwargs):
        error_calls.append(1)
        err = be.data_fetcher_module.MissingOHLCVColumnsError("ohlcv_columns_missing")
        setattr(err, "fetch_reason", "ohlcv_columns_missing")
        setattr(err, "missing_columns", ("close",))
        setattr(err, "symbol", symbol)
        setattr(err, "timeframe", "1Day")
        raise err

    provider_updates: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def _capture_update(*args, **kwargs):
        provider_updates.append((args, kwargs))

    monkeypatch.setattr(be.data_fetcher_module, "get_daily_df", _raise_missing)
    monkeypatch.setattr(be.provider_monitor, "update_data_health", _capture_update)

    with pytest.raises(be.data_fetcher_module.MissingOHLCVColumnsError):
        fetcher.get_daily_df(types.SimpleNamespace(), symbol)
    assert len(error_calls) == 1
    assert provider_updates
    assert provider_updates[0][1].get("severity") == "hard_fail"

    with pytest.raises(be.data_fetcher_module.MissingOHLCVColumnsError):
        fetcher.get_daily_df(types.SimpleNamespace(), symbol)
    assert len(error_calls) == 1


def test_daily_fetch_skips_direct_when_safe_missing(monkeypatch):
    pd = load_pandas()
    fetcher = _stub_fetcher(monkeypatch)
    symbol = "MSFT"

    monkeypatch.setattr(be, "datetime", FixedDateTime)
    monkeypatch.setattr(be, "is_market_open", lambda: True)
    be.daily_cache_hit = None
    be.daily_cache_miss = None
    be._DAILY_FETCH_MEMO.clear()
    fetcher._daily_cache.clear()
    fetcher._daily_error_state.clear()

    bars_stub = types.SimpleNamespace(
        TimeFrame=types.SimpleNamespace(Day="Day"),
        _create_empty_bars_dataframe=lambda _tf: pd.DataFrame(),
    )
    monkeypatch.setattr(be, "bars", bars_stub, raising=False)

    captured: dict[str, object] = {}

    def fake_get_daily_df(symbol_arg, start, end, *, feed=None, adjustment=None):
        captured["symbol"] = symbol_arg
        captured["start"] = start
        captured["feed"] = feed
        idx = pd.to_datetime([start, end])
        return pd.DataFrame(
            {
                "open": [1.0, 1.1],
                "high": [1.2, 1.3],
                "low": [0.9, 1.0],
                "close": [1.05, 1.15],
                "volume": [100, 110],
            },
            index=idx,
        )

    monkeypatch.setattr(be.data_fetcher_module, "get_daily_df", fake_get_daily_df)
    monkeypatch.setattr(be.provider_monitor, "update_data_health", lambda *a, **k: None)

    result = fetcher.get_daily_df(types.SimpleNamespace(), symbol)

    assert isinstance(result, pd.DataFrame)
    assert captured["symbol"] == symbol
    assert not hasattr(be.bars, "safe_get_stock_bars")
