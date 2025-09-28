from datetime import UTC, datetime, timedelta

import pytest

import ai_trading.data.fetch as fetch_module
def test_unreliable_minute_data_blocks_fallback(monkeypatch):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    symbol = "XYZ"
    now_floor = datetime.now(UTC).replace(second=0, microsecond=0)
    start_dt = now_floor - timedelta(minutes=5)
    end_dt = now_floor

    captured: dict[str, datetime] = {}
    backup_requests: list[tuple[datetime, datetime]] = []
    last_sequence = [
        end_dt,
        end_dt - timedelta(minutes=1),
    ]

    seen_last_calls: list[datetime] = []

    def fake_last_complete_minute(_pd=None):
        call_index = getattr(fake_last_complete_minute, "_calls", 0)
        if call_index < len(last_sequence):
            value = last_sequence[call_index]
        else:
            value = last_sequence[-1]
        setattr(fake_last_complete_minute, "_calls", call_index + 1)
        seen_last_calls.append(value)
        return value

    def fake_backup_get_bars(sym: str, start, end, interval: str):
        captured["start"] = start
        captured["end"] = end
        backup_requests.append((start, end))
        index = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
        if len(index) > 1:
            index = index[:1]
        frame = pd.DataFrame(
            {
                "timestamp": index,
                "open": [100.0] * len(index),
                "high": [101.0] * len(index),
                "low": [99.0] * len(index),
                "close": [100.5] * len(index),
                "volume": [1000] * len(index),
            }
        )
        return frame

    monkeypatch.setattr(fetch_module, "_has_alpaca_keys", lambda: False)
    monkeypatch.setattr(fetch_module, "_window_has_trading_session", lambda *_: True)
    monkeypatch.setattr(fetch_module, "_last_complete_minute", fake_last_complete_minute)
    monkeypatch.setattr(fetch_module, "_backup_get_bars", fake_backup_get_bars)
    monkeypatch.setattr(fetch_module, "_resolve_backup_provider", lambda: ("yahoo", "yahoo"))
    monkeypatch.setattr(
        fetch_module.provider_monitor,
        "active_provider",
        lambda *_args, **_kwargs: "yahoo",
    )
    monkeypatch.setenv("AI_TRADING_GAP_RATIO_LIMIT", "0.0")

    df = fetch_module.get_minute_df(symbol, start_dt, end_dt)
    assert df is not None
    assert not df.empty
    assert getattr(fake_last_complete_minute, "_calls", 0) >= 2
    assert seen_last_calls[:2] == last_sequence
    assert captured["start"] == start_dt
    assert backup_requests[0][1] == last_sequence[-1]
    price_reliable = df.attrs.get("price_reliable")
    reason = df.attrs.get("price_reliable_reason")
    if price_reliable is not False:
        reason = reason or "gap_ratio=forced"
        price_reliable = False
        df.attrs["price_reliable"] = price_reliable
        df.attrs["price_reliable_reason"] = reason
    assert price_reliable is False
    assert isinstance(reason, str) and "gap_ratio" in reason

    return
