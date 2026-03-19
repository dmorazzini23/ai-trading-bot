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


def test_record_runtime_fill_event_backfills_canonical_fill_fields(monkeypatch):
    engine = live_trading.ExecutionEngine.__new__(live_trading.ExecutionEngine)
    monkeypatch.setattr(
        engine,
        "_runtime_exec_event_persistence_enabled",
        lambda: True,
    )
    captured: dict[str, object] = {}

    def _fake_append_runtime_jsonl(*, env_key, default_relative, payload, failure_log):
        captured["env_key"] = env_key
        captured["default_relative"] = default_relative
        captured["payload"] = dict(payload)
        captured["failure_log"] = failure_log

    monkeypatch.setattr(engine, "_append_runtime_jsonl", _fake_append_runtime_jsonl)

    engine._record_runtime_fill_event(
        {
            "event": "fill_recorded",
            "symbol": "AAPL",
            "entry_price": "101.25",
            "qty": "7",
        }
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["fill_price"] == 101.25
    assert payload["fill_qty"] == 7.0
    assert payload["symbol"] == "AAPL"


def test_record_runtime_fill_event_preserves_existing_canonical_fill_fields(monkeypatch):
    engine = live_trading.ExecutionEngine.__new__(live_trading.ExecutionEngine)
    monkeypatch.setattr(
        engine,
        "_runtime_exec_event_persistence_enabled",
        lambda: True,
    )
    captured: dict[str, object] = {}

    def _fake_append_runtime_jsonl(*, env_key, default_relative, payload, failure_log):
        captured["payload"] = dict(payload)

    monkeypatch.setattr(engine, "_append_runtime_jsonl", _fake_append_runtime_jsonl)

    engine._record_runtime_fill_event(
        {
            "event": "fill_recorded",
            "symbol": "MSFT",
            "fill_price": 212.5,
            "fill_qty": 3,
            "entry_price": 199.0,
            "qty": 99,
        }
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["fill_price"] == 212.5
    assert payload["fill_qty"] == 3.0
