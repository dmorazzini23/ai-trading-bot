import json
from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.data import fetch

pd = pytest.importorskip("pandas")


@pytest.fixture
def capmetrics(monkeypatch: pytest.MonkeyPatch):
    bucket: list[tuple[str, dict]] = []

    def record(name: str, value: float = 1.0, tags: dict | None = None):
        bucket.append((name, tags or {}))

    monkeypatch.setattr(fetch.metrics, "incr", record, raising=False)
    return bucket


class _Resp:
    def __init__(self, payload: dict, *, status: int = 200, correlation: str | None = None):
        self.status_code = status
        self.headers = {"Content-Type": "application/json"}
        if correlation is not None:
            self.headers["x-request-id"] = correlation
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses: list[_Resp]):
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def get(self, url, params=None, headers=None, timeout=None):  # noqa: D401 - simple stub
        self.calls.append(params or {})
        return self._responses.pop(0)


def _reset_state():
    fetch._FEED_OVERRIDE_BY_TF.clear()
    fetch._FEED_FAILOVER_ATTEMPTS.clear()
    fetch._FEED_SWITCH_LOGGED.clear()
    fetch._FEED_SWITCH_HISTORY.clear()
    fetch._IEX_EMPTY_COUNTS.clear()
    fetch._ALPACA_EMPTY_ERROR_COUNTS.clear()
    fetch._CYCLE_FALLBACK_FEED.clear()


def test_empty_payload_switches_to_preferred_feed(monkeypatch, capmetrics):
    _reset_state()
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setattr(fetch, "_ALLOW_SIP", True)
    monkeypatch.setattr(fetch, "_HAS_SIP", True)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False)
    monkeypatch.setattr(fetch, "alpaca_feed_failover", lambda: ("sip",))
    monkeypatch.setattr(fetch, "provider_priority", lambda: ["alpaca_iex", "alpaca_sip"])
    monkeypatch.setattr(fetch, "alpaca_empty_to_backup", lambda: False)

    start = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)
    end = start + timedelta(minutes=1)

    session = _Session(
        [
            _Resp({"bars": []}, correlation="iex"),
            _Resp(
                {
                    "bars": [
                        {"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}
                    ]
                },
                correlation="sip",
            ),
        ]
    )
    monkeypatch.setattr(fetch, "_HTTP_SESSION", session)

    df = fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")

    assert hasattr(df, "empty")
    assert not getattr(df, "empty", True)
    normalized = fetch.normalize_ohlcv_df(df, include_columns=("timestamp",))
    assert list(normalized.columns[:6]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert normalized.index.name == "timestamp"
    assert session.calls[0]["feed"] == "iex"
    assert session.calls[1]["feed"] == "sip"
    assert fetch._FEED_OVERRIDE_BY_TF[("AAPL", "1Min")] == "sip"
    assert ("AAPL", "1Min", "sip") in fetch._FEED_SWITCH_LOGGED
    assert fetch._FEED_SWITCH_HISTORY == [("AAPL", "1Min", "sip")]
    names = [name for name, _ in capmetrics]
    assert "data.fetch.fallback_attempt" in names
    assert "data.fetch.success" in names
    assert "data.fetch.fallback_success" in names
    idx_attempt = names.index("data.fetch.fallback_attempt")
    idx_fb_success = names.index("data.fetch.fallback_success")
    idx_success = names.index("data.fetch.success")
    assert idx_attempt < idx_fb_success < idx_success
    fallback_success_tags = capmetrics[idx_fb_success][1]
    assert fallback_success_tags.get("feed") == "sip"
    success_tags = capmetrics[idx_success][1]
    assert success_tags.get("feed") == "sip"


def test_empty_payload_switch_records_override_without_preferred_list(monkeypatch, capmetrics):
    _reset_state()
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setattr(fetch, "_ALLOW_SIP", True)
    monkeypatch.setattr(fetch, "_HAS_SIP", True)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False)
    monkeypatch.setattr(fetch, "alpaca_feed_failover", lambda: ())
    monkeypatch.setattr(fetch, "provider_priority", lambda: ["alpaca_iex"])
    monkeypatch.setattr(fetch, "max_data_fallbacks", lambda: 1)
    monkeypatch.setattr(fetch, "alpaca_empty_to_backup", lambda: False)

    start = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)
    end = start + timedelta(minutes=1)

    session = _Session(
        [
            _Resp({"bars": []}, correlation="iex"),
            _Resp(
                {
                    "bars": [
                        {"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}
                    ]
                },
                correlation="sip",
            ),
        ]
    )
    monkeypatch.setattr(fetch, "_HTTP_SESSION", session)

    df = fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")

    assert hasattr(df, "empty")
    assert not getattr(df, "empty", True)
    normalized = fetch.normalize_ohlcv_df(df, include_columns=("timestamp",))
    assert list(normalized.columns[:6]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert normalized.index.name == "timestamp"
    assert session.calls[0]["feed"] == "iex"
    assert session.calls[1]["feed"] == "sip"
    assert fetch._FEED_OVERRIDE_BY_TF[("AAPL", "1Min")] == "sip"
    assert fetch._FEED_SWITCH_HISTORY == [("AAPL", "1Min", "sip")]
    names = [name for name, _ in capmetrics]
    assert "data.fetch.fallback_attempt" in names
    assert "data.fetch.success" in names
    assert "data.fetch.fallback_success" in names
    idx_attempt = names.index("data.fetch.fallback_attempt")
    idx_fb_success = names.index("data.fetch.fallback_success")
    idx_success = names.index("data.fetch.success")
    assert idx_attempt < idx_fb_success < idx_success
    fallback_success_tags = capmetrics[idx_fb_success][1]
    assert fallback_success_tags.get("feed") == "sip"
    success_tags = capmetrics[idx_success][1]
    assert success_tags.get("feed") == "sip"


def test_window_no_session_prefers_alpaca_fallback(monkeypatch, capmetrics):
    _reset_state()
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setattr(fetch, "_ALLOW_SIP", True)
    monkeypatch.setattr(fetch, "_HAS_SIP", True)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False)
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "alpaca_feed_failover", lambda: ("sip",))
    monkeypatch.setattr(fetch, "provider_priority", lambda: ["alpaca_iex", "alpaca_sip"])
    monkeypatch.setattr(fetch, "alpaca_empty_to_backup", lambda: False)

    def _fail_backup(*_a, **_k):
        raise AssertionError("backup provider should not run before alpaca fallbacks")

    monkeypatch.setattr(fetch, "_backup_get_bars", _fail_backup)

    start = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)
    end = start + timedelta(minutes=1)

    session = _Session(
        [
            _Resp({"bars": []}, correlation="iex"),
            _Resp(
                {
                    "bars": [
                        {"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}
                    ]
                },
                correlation="sip",
            ),
        ]
    )
    monkeypatch.setattr(fetch, "_HTTP_SESSION", session)

    df = fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")

    assert hasattr(df, "empty")
    assert not getattr(df, "empty", True)
    assert len(session.calls) == 2
    assert session.calls[0]["feed"] == "iex"
    assert session.calls[1]["feed"] == "sip"
    assert fetch._FEED_OVERRIDE_BY_TF[("AAPL", "1Min")] == "sip"
    names = [name for name, _ in capmetrics]
    assert "data.fetch.fallback_attempt" in names
    assert "data.fetch.fallback_success" in names
    assert "data.fetch.success" in names
    idx_attempt = names.index("data.fetch.fallback_attempt")
    idx_fb_success = names.index("data.fetch.fallback_success")
    idx_success = names.index("data.fetch.success")
    assert idx_attempt < idx_fb_success < idx_success


def test_no_session_empty_payload_uses_sip_before_backup(monkeypatch):
    _reset_state()
    fetch._SKIPPED_SYMBOLS.clear()
    fetch._IEX_EMPTY_COUNTS.clear()
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr(fetch, "_ALLOW_SIP", True)
    monkeypatch.setattr(fetch, "_HAS_SIP", True)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False)
    monkeypatch.setattr(fetch, "_ENABLE_HTTP_FALLBACK", True, raising=False)
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "max_data_fallbacks", lambda: 2)

    fallback_calls: list[tuple[str, ...]] = []

    def fake_feed_failover():
        feeds = ("sip", "yahoo")
        fallback_calls.append(feeds)
        return feeds

    monkeypatch.setattr(fetch, "alpaca_feed_failover", fake_feed_failover)
    monkeypatch.setattr(fetch, "provider_priority", lambda: ["alpaca_iex", "alpaca_sip", "yahoo"])
    monkeypatch.setattr(fetch, "alpaca_empty_to_backup", lambda: False)

    def _fail_backup(*_a, **_k):
        raise AssertionError("backup provider should not execute during no-session handling")

    monkeypatch.setattr(fetch, "_backup_get_bars", _fail_backup)

    start = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)
    end = start + timedelta(minutes=1)

    session = _Session([
        _Resp({"bars": []}, correlation="iex"),
        _Resp({"bars": []}, correlation="sip"),
    ])
    monkeypatch.setattr(fetch, "_HTTP_SESSION", session)

    df = fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")

    assert fallback_calls, "alpaca_feed_failover should be evaluated"
    assert len(session.calls) == 2
    assert session.calls[0]["feed"] == "iex"
    assert session.calls[1]["feed"] == "sip"
    assert hasattr(df, "empty")
    assert getattr(df, "empty", True)
    tf_key = ("AAPL", "1Min")
    assert tf_key not in fetch._IEX_EMPTY_COUNTS or fetch._IEX_EMPTY_COUNTS[tf_key] == 0
    assert "sip" in fetch._FEED_FAILOVER_ATTEMPTS.get(tf_key, set())


def test_cached_override_respects_ttl(monkeypatch):
    _reset_state()
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    base_time = 1_000.0

    def faux_time():
        return faux_time.current

    faux_time.current = base_time
    monkeypatch.setattr(fetch.time, "time", lambda: faux_time())
    monkeypatch.setattr(fetch, "_OVERRIDE_TTL_S", 10.0, raising=False)

    fetch._record_feed_switch("AAPL", "1Min", "iex", "sip")

    assert fetch._get_cached_or_primary("AAPL", "iex") == "sip"
    faux_time.current = base_time + 5
    assert fetch._get_cached_or_primary("AAPL", "iex") == "sip"
    faux_time.current = base_time + 15
    assert fetch._get_cached_or_primary("AAPL", "iex") == "iex"


def test_feed_override_used_on_subsequent_requests(monkeypatch):
    _reset_state()
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setattr(fetch, "_ALLOW_SIP", True)
    monkeypatch.setattr(fetch, "_HAS_SIP", True)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False)
    monkeypatch.setattr(fetch, "alpaca_feed_failover", lambda: ("sip",))
    monkeypatch.setattr(fetch, "alpaca_empty_to_backup", lambda: False)
    monkeypatch.setattr(fetch, "_verify_minute_continuity", lambda df, *a, **k: df)
    monkeypatch.setattr(
        fetch,
        "_repair_rth_minute_gaps",
        lambda df, **k: (df, {"status": "ok"}, False),
    )
    monkeypatch.setattr(fetch, "provider_priority", lambda: ["alpaca_iex", "alpaca_sip"])
    monkeypatch.setattr(fetch, "max_data_fallbacks", lambda: 2)
    monkeypatch.setattr(
        fetch,
        "_backup_get_bars",
        lambda *a, **k: pd.DataFrame(
            {
                "t": [start],
                "o": [2.0],
                "h": [2.0],
                "l": [2.0],
                "c": [2.0],
                "v": [2],
            }
        ),
    )

    start = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)
    end = start + timedelta(minutes=1)

    first_session = _Session(
        [
            _Resp({"bars": []}, correlation="iex"),
            _Resp(
                {
                    "bars": [
                        {"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}
                    ]
                },
                correlation="sip",
            ),
        ]
    )
    monkeypatch.setattr(fetch, "_HTTP_SESSION", first_session)

    df_first = fetch.get_minute_df("AAPL", start, end)
    assert hasattr(df_first, "empty")
    assert not getattr(df_first, "empty", True)
    assert list(df_first.columns[:6]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert first_session.calls[0]["feed"] == "iex"
    assert first_session.calls[1]["feed"] == "sip"
    assert fetch._FEED_OVERRIDE_BY_TF[("AAPL", "1Min")] == "sip"

    second_session = _Session([
        _Resp({"bars": []}, correlation="iex2"),
        _Resp(
            {
                "bars": [
                    {"t": "2024-01-01T00:00:00Z", "o": 2, "h": 2, "l": 2, "c": 2, "v": 2}
                ]
            },
            correlation="sip2",
        ),
    ])
    monkeypatch.setattr(fetch, "_HTTP_SESSION", second_session)

    df_second = fetch.get_minute_df("AAPL", start, end)
    assert hasattr(df_second, "empty")
    assert not getattr(df_second, "empty", True)
    assert list(df_second.columns[:6]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert fetch._FEED_SWITCH_HISTORY == [("AAPL", "1Min", "sip")]


def test_window_no_session_override_persists(monkeypatch):
    _reset_state()
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setattr(fetch, "_ALLOW_SIP", True)
    monkeypatch.setattr(fetch, "_HAS_SIP", True)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False)
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "alpaca_feed_failover", lambda: ("sip",))
    monkeypatch.setattr(fetch, "alpaca_empty_to_backup", lambda: False)
    monkeypatch.setattr(fetch, "provider_priority", lambda: ["alpaca_iex", "alpaca_sip"])
    monkeypatch.setattr(fetch, "max_data_fallbacks", lambda: 2)
    monkeypatch.setattr(fetch, "_verify_minute_continuity", lambda df, *a, **k: df)
    monkeypatch.setattr(
        fetch,
        "_repair_rth_minute_gaps",
        lambda df, **k: (df, {"status": "ok"}, False),
    )

    def _fail_backup(*_a, **_k):
        raise AssertionError("backup provider should not run before alpaca fallbacks")

    monkeypatch.setattr(fetch, "_backup_get_bars", _fail_backup)

    start = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)
    end = start + timedelta(minutes=1)

    first_session = _Session(
        [
            _Resp({"bars": []}, correlation="iex"),
            _Resp(
                {
                    "bars": [
                        {"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}
                    ]
                },
                correlation="sip",
            ),
        ]
    )
    monkeypatch.setattr(fetch, "_HTTP_SESSION", first_session)

    df_first = fetch.get_minute_df("AAPL", start, end)
    assert hasattr(df_first, "empty")
    assert not getattr(df_first, "empty", True)
    assert first_session.calls[0]["feed"] == "iex"
    assert first_session.calls[1]["feed"] == "sip"
    assert fetch._FEED_OVERRIDE_BY_TF[("AAPL", "1Min")] == "sip"

    second_session = _Session(
        [
            _Resp(
                {
                    "bars": [
                        {"t": "2024-01-02T00:00:00Z", "o": 2, "h": 2, "l": 2, "c": 2, "v": 2}
                    ]
                },
                correlation="sip2",
            )
        ]
    )
    monkeypatch.setattr(fetch, "_HTTP_SESSION", second_session)

    df_second = fetch.get_minute_df("AAPL", start, end)
    assert hasattr(df_second, "empty")
    assert not getattr(df_second, "empty", True)
    assert len(second_session.calls) == 1
    assert second_session.calls[0]["feed"] == "sip"
    assert fetch._FEED_SWITCH_HISTORY == [("AAPL", "1Min", "sip")]

def test_alt_feed_switch_records_override(monkeypatch):
    _reset_state()
    fetch._SKIPPED_SYMBOLS.clear()
    fetch._EMPTY_BAR_COUNTS.clear()
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv("ENABLE_FINNHUB", "0")
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setattr(fetch, "_has_alpaca_keys", lambda: True)
    monkeypatch.setattr(fetch, "_sip_configured", lambda: True)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False)
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(fetch, "max_data_fallbacks", lambda: 2)
    monkeypatch.setattr(fetch, "provider_priority", lambda: ["alpaca_iex", "alpaca_sip"])
    monkeypatch.setattr(fetch, "_verify_minute_continuity", lambda df, *a, **k: df)
    monkeypatch.setattr(fetch, "_post_process", lambda df, *a, **k: df)
    monkeypatch.setattr(fetch, "_backup_get_bars", lambda *a, **k: pd.DataFrame())

    symbol = "AAPL"
    start = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    tf_key = (symbol, "1Min")
    fetch._EMPTY_BAR_COUNTS[tf_key] = fetch._EMPTY_BAR_THRESHOLD

    def fake_fetch(_, __, ___, timeframe, *, feed=None):
        if feed == "iex":
            raise fetch.EmptyBarsError("empty")
        assert feed == "sip"
        df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp(start)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1],
            }
        )
        return df

    monkeypatch.setattr(fetch, "_fetch_bars", fake_fetch)

    df = fetch.get_minute_df(symbol, start, end)

    assert hasattr(df, "empty")
    assert not getattr(df, "empty", True)
    assert list(df.columns[:6]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert fetch._FEED_OVERRIDE_BY_TF[tf_key] == "sip"
    assert (symbol, "1Min", "sip") in fetch._FEED_SWITCH_LOGGED
    assert fetch._FEED_SWITCH_HISTORY == [(symbol, "1Min", "sip")]
