import logging
import time
from types import SimpleNamespace

import pytest

from ai_trading.execution import timing as execution_timing
from ai_trading.utils.prof import StageTimer


def test_execution_span_records_non_zero_duration(caplog):
    execution_timing.reset_cycle()
    logger = logging.getLogger("ai_trading.execution.test")

    with caplog.at_level(logging.INFO):
        with execution_timing.execution_span(logger, symbol="AAPL", side="buy", qty=10):
            time.sleep(0.002)

    total = execution_timing.cycle_seconds()
    assert total > 0.0

    messages = [record.message for record in caplog.records]
    assert "EXECUTE_TIMING_START" in messages
    assert "EXECUTE_TIMING_END" in messages

    with caplog.at_level(logging.DEBUG):
        with StageTimer(logger, "CYCLE_EXECUTE", override_ms=total * 1000):
            pass

    stage_records = [
        record
        for record in caplog.records
        if "STAGE_TIMING" in str(record.message) and getattr(record, "stage", None) == "CYCLE_EXECUTE"
    ]
    assert stage_records
    assert any(getattr(record, "elapsed_ms", 0) >= int(total * 1000) for record in stage_records)


def test_stage_timer_minimum_elapsed_ms():
    logger = logging.getLogger("ai_trading.utils.prof.test")
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    logger.addHandler(handler)
    try:
        logger.setLevel(logging.DEBUG)
        with StageTimer(logger, "TINY", override_ms=0.2):
            pass
    finally:
        logger.removeHandler(handler)

    tiny = [record for record in records if getattr(record, "stage", None) == "TINY"]
    assert tiny
    assert getattr(tiny[0], "elapsed_ms", None) == 1


def test_submit_order_records_execution_timing(monkeypatch):
    from ai_trading.core import bot_engine

    class _ExecEngine:
        @staticmethod
        def execute_order(symbol, side, qty, price=None, **kwargs):  # noqa: ARG004
            time.sleep(0.002)
            return {"id": "order-1", "symbol": symbol, "side": side, "qty": qty}

    execution_timing.reset_cycle()
    monkeypatch.setattr(bot_engine, "_exec_engine", _ExecEngine())
    monkeypatch.setattr(bot_engine, "_resolve_trading_config", lambda _ctx: SimpleNamespace(rth_only=False, allow_extended=True))
    monkeypatch.setattr(bot_engine, "_kill_switch_active", lambda _cfg: (False, None))

    order = bot_engine.submit_order(SimpleNamespace(), "AAPL", 1, "buy", price=100.0)
    assert order is not None
    assert execution_timing.cycle_seconds() > 0.0


def test_submit_order_forwards_annotations_and_lineage(monkeypatch):
    from ai_trading.core import bot_engine

    captured: dict[str, object] = {}

    class _ExecEngine:
        @staticmethod
        def execute_order(symbol, side, qty, price=None, **kwargs):  # noqa: ARG004
            captured["kwargs"] = dict(kwargs)
            return {"id": "order-2", "symbol": symbol, "side": side, "qty": qty}

    monkeypatch.setattr(bot_engine, "_exec_engine", _ExecEngine())
    monkeypatch.setattr(
        bot_engine,
        "_resolve_trading_config",
        lambda _ctx: SimpleNamespace(rth_only=False, allow_extended=True),
    )
    monkeypatch.setattr(bot_engine, "_kill_switch_active", lambda _cfg: (False, None))
    monkeypatch.setattr(bot_engine, "get_price_source", lambda _symbol: "primary")

    order = bot_engine.submit_order(
        SimpleNamespace(),
        "AAPL",
        2,
        "buy",
        price=101.0,
        annotations={"strategy_label": "unit", "expected_net_edge_bps": 5.0},
        using_fallback_price=True,
        price_hint=101.0,
        model_id="ml-main",
        model_version="v2026.04.07",
        config_snapshot_hash="cfg-abc",
        metadata={"model_id": "ml-main"},
    )

    assert order is not None
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["annotations"]["strategy_label"] == "unit"
    assert kwargs["annotations"]["using_fallback_price"] is True
    assert kwargs["price_hint"] == 101.0
    assert kwargs["model_id"] == "ml-main"
    assert kwargs["model_version"] == "v2026.04.07"
    assert kwargs["config_snapshot_hash"] == "cfg-abc"
