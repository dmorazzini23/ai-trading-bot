from ai_trading.execution import live_trading


def test_mark_fill_reported_tracks_reported_quantity():
    engine = live_trading.ExecutionEngine.__new__(live_trading.ExecutionEngine)

    # Should be a no-op even when signal metadata is missing.
    engine.mark_fill_reported("order-1", 1)

    meta = live_trading._SignalMeta(signal=None, requested_qty=10, signal_weight=None)
    engine._order_signal_meta = {"order-1": meta}

    engine.mark_fill_reported("order-1", 4)
    assert engine._order_signal_meta["order-1"].reported_fill_qty == 4

    engine.mark_fill_reported("order-1", 10)
    assert "order-1" not in engine._order_signal_meta
