import json
import types
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import ai_trading.data.fetch as fetch
import ai_trading.net.http as http
from ai_trading.data.fetch.metrics import inc_provider_fallback

pytest.importorskip("pandas")


def _reset_feed_state() -> None:
    fetch._IEX_EMPTY_COUNTS.clear()
    fetch._FEED_OVERRIDE_BY_TF.clear()
    fetch._FEED_FAILOVER_ATTEMPTS.clear()
    fetch._FEED_SWITCH_HISTORY.clear()
    fetch._FEED_SWITCH_LOGGED.clear()
    fetch._cycle_feed_override.clear()


@pytest.mark.parametrize(
    ("preferred_feeds", "provider_order"),
    [
        (("sip",), ["alpaca_iex", "alpaca_sip"]),
        ((), ["alpaca_iex"]),
    ],
    ids=["preferred_list", "no_preference"],
)
def test_iex_empty_switches_to_sip(monkeypatch, caplog, preferred_feeds, provider_order):
    symbol = "AAPL"
    start = datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC"))
    end = start + timedelta(days=1)

    _reset_feed_state()
    fetch._IEX_EMPTY_COUNTS[(symbol, "1Min")] = fetch._IEX_EMPTY_THRESHOLD + 1
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "dummy")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "dummy")
    monkeypatch.setattr(fetch, "_ALLOW_SIP", True, raising=False)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False, raising=False)
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "is_market_open", lambda: False)
    monkeypatch.setattr(fetch, "alpaca_feed_failover", lambda: preferred_feeds)
    monkeypatch.setattr(fetch, "provider_priority", lambda: provider_order)

    feeds = []

    def fake_get(url, params=None, headers=None, timeout=None):
        feeds.append(params.get("feed"))
        if params.get("feed") == "iex":
            data = {"bars": []}
        else:
            data = {"bars": [{"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]}
        return types.SimpleNamespace(
            status_code=200,
            text=json.dumps(data),
            headers={"Content-Type": "application/json"},
            json=lambda: data,
        )

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    monkeypatch.setattr(http.requests, "get", fake_get, raising=False)
    monkeypatch.setattr(fetch, "_HTTP_SESSION", types.SimpleNamespace(get=fake_get), raising=False)

    monkeypatch.setattr(fetch, "_sip_fallback_allowed", lambda *a, **k: True)

    before = inc_provider_fallback("alpaca_iex", "alpaca_sip")
    with caplog.at_level("INFO"):
        df = fetch._fetch_bars(symbol, start, end, "1Min", feed="iex")
    after = inc_provider_fallback("alpaca_iex", "alpaca_sip")

    assert feeds[:2] == ["iex", "sip"]
    assert not df.empty
    assert fetch._FEED_OVERRIDE_BY_TF[(symbol, "1Min")] == "sip"
    assert fetch._cycle_feed_override.get(symbol) == "sip"
    assert "sip" in fetch._FEED_FAILOVER_ATTEMPTS.get((symbol, "1Min"), set())
    assert after >= before + 1


def test_prepare_sip_fallback_records_history_and_metrics(monkeypatch):
    _reset_feed_state()

    metrics_calls: list[tuple[str, dict]] = []
    fallback_calls: list[tuple[tuple[str, str], dict]] = []
    warnings: list[tuple[str, dict | None]] = []
    caplog_calls: list[tuple[str, dict]] = []

    def record_metric(name: str, value: float = 1.0, tags: dict | None = None):
        metrics_calls.append((name, tags or {}))

    def record_fallback(from_provider: str, to_provider: str) -> int:
        fallback_calls.append(((from_provider, to_provider), {}))
        return 0

    def record_warning(message: str, *, extra: dict | None = None):
        warnings.append((message, extra))

    def push_to_caplog(message: str, **kwargs):
        caplog_calls.append((message, kwargs))

    monkeypatch.setattr(fetch.metrics, "incr", record_metric, raising=False)
    monkeypatch.setattr(fetch, "inc_provider_fallback", record_fallback, raising=False)
    monkeypatch.setattr(fetch.logger, "warning", record_warning, raising=False)

    fetch._prepare_sip_fallback(
        "AAPL",
        "1Min",
        "iex",
        occurrences=4,
        correlation_id="corr-1",
        push_to_caplog=push_to_caplog,
        tags_factory=lambda: {"test": "1"},
    )

    assert fetch._FEED_SWITCH_HISTORY == [("AAPL", "1Min", "sip")]
    assert fetch._FEED_OVERRIDE_BY_TF[("AAPL", "1Min")] == "sip"
    assert any(name == "data.fetch.feed_switch" for name, _ in metrics_calls)
    assert fallback_calls and fallback_calls[-1][0] == ("alpaca_iex", "alpaca_sip")
    assert warnings and warnings[-1][0] == "ALPACA_IEX_FALLBACK_SIP"
    assert caplog_calls and caplog_calls[-1][0] == "ALPACA_IEX_FALLBACK_SIP"


def test_prepare_sip_fallback_persists_override(monkeypatch):
    _reset_feed_state()
    monkeypatch.setattr(fetch, "_OVERRIDE_TTL_S", 60.0, raising=False)
    monkeypatch.setattr(fetch.metrics, "incr", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(fetch, "inc_provider_fallback", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr(fetch.logger, "warning", lambda *a, **k: None, raising=False)

    fetch._prepare_sip_fallback(
        "MSFT",
        "1Min",
        "iex",
        occurrences=1,
        correlation_id=None,
        push_to_caplog=lambda *a, **k: None,
        tags_factory=lambda: {},
    )

    assert fetch._get_cached_or_primary("MSFT", "iex") == "sip"
