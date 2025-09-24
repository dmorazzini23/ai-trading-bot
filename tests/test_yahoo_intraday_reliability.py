import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import ai_trading.data.fetch as fetch_module
@pytest.mark.usefixtures("caplog")
def test_unreliable_minute_data_blocks_fallback(monkeypatch, caplog):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    import ai_trading.core.bot_engine as bot_engine_module
    from ai_trading.core.bot_engine import BotState
    symbol = "XYZ"
    now_floor = datetime.now(UTC).replace(second=0, microsecond=0)
    start_dt = now_floor - timedelta(minutes=5)
    end_dt = now_floor

    captured: dict[str, datetime] = {}

    def fake_backup_get_bars(sym: str, start, end, interval: str):
        captured["start"] = start
        captured["end"] = end
        index = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
        if len(index) > 1:
            index = index.delete(len(index) - 1)
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
    monkeypatch.setattr(fetch_module, "_backup_get_bars", fake_backup_get_bars)
    monkeypatch.setattr(fetch_module, "_resolve_backup_provider", lambda: ("yahoo", "yahoo"))

    caplog.set_level(logging.INFO)
    df = fetch_module.get_minute_df(symbol, start_dt, end_dt)
    assert df is not None
    assert not df.empty
    assert captured["end"] == now_floor - timedelta(minutes=1)
    price_reliable = df.attrs.get("price_reliable")
    reason = df.attrs.get("price_reliable_reason")
    assert price_reliable is False
    assert isinstance(reason, str) and "gap_ratio" in reason

    state = BotState()
    state.price_reliability[symbol] = (price_reliable, reason)

    feat_df = df.copy()
    feat_df["atr"] = 1.0

    ctx = SimpleNamespace(portfolio_weights={}, api=None, config=SimpleNamespace(exposure_cap_aggressive=0.88))

    monkeypatch.setattr(
        bot_engine_module,
        "_resolve_order_quote",
        lambda sym, prefer_backup=False: (float(df["close"].iloc[-1]), "yahoo_close"),
    )

    submit_called = False

    def fake_submit_order(*_, **__):
        nonlocal submit_called
        submit_called = True
        return "order-id"

    monkeypatch.setattr(bot_engine_module, "submit_order", fake_submit_order)

    result = bot_engine_module._enter_long(
        ctx,
        state,
        symbol,
        balance=10_000.0,
        feat_df=feat_df,
        final_score=1.0,
        conf=0.7,
        strat="test",
    )

    assert result is True
    assert submit_called is False
    assert any("ORDER_SKIPPED_UNRELIABLE_PRICE" in record.message for record in caplog.records)
