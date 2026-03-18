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


def test_reconcile_pending_tca_from_fill_calls_reconciler(monkeypatch):
    engine = live_trading.ExecutionEngine.__new__(live_trading.ExecutionEngine)

    monkeypatch.setattr(
        live_trading,
        "_resolve_bool_env",
        lambda key: True if key in {"AI_TRADING_TCA_ENABLED", "AI_TRADING_TCA_UPDATE_ON_FILL"} else None,
    )
    monkeypatch.setattr(live_trading, "_runtime_env", lambda _key, default=None: default)
    monkeypatch.setattr(
        live_trading,
        "resolve_runtime_artifact_path",
        lambda configured_path, default_relative: live_trading.Path(configured_path),
    )

    captured: dict[str, object] = {}

    def _fake_reconcile(path: str, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return True, "reconciled"

    monkeypatch.setattr(live_trading, "reconcile_pending_tca_with_fill", _fake_reconcile)

    engine._reconcile_pending_tca_from_fill(
        symbol="AAPL",
        side="buy",
        fill_qty=5.0,
        fill_price=100.5,
        timestamp=live_trading.datetime.now(live_trading.UTC),
        order_id="oid-1",
        client_order_id="cid-1",
        order_status="filled",
        fee_amount=0.25,
        source="unit_test",
    )

    assert captured["path"] == "runtime/tca_records.jsonl"
    kwargs_obj = captured["kwargs"]
    assert isinstance(kwargs_obj, dict)
    assert kwargs_obj["client_order_id"] == "cid-1"
    assert kwargs_obj["order_id"] == "oid-1"
    assert kwargs_obj["fill_qty"] == 5.0
    assert kwargs_obj["fill_price"] == 100.5
