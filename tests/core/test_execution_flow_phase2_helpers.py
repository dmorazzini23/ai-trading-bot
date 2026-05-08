from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_trading.core import bot_engine, execution_flow


def test_poll_order_fill_status_normalizes_numeric_attrs_and_stops_on_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    orders = iter(
        [
            SimpleNamespace(status="accepted", filled_qty="1", qty="3"),
            SimpleNamespace(status="filled", filled_quantity="3", quantity="3"),
        ]
    )
    ctx = SimpleNamespace(api=SimpleNamespace(get_order=lambda _order_id: next(orders)))
    monkeypatch.setattr(execution_flow.pytime, "sleep", lambda seconds: sleeps.append(seconds))

    execution_flow.poll_order_fill_status(ctx, "order-1", timeout=1)

    assert sleeps == [0.2]


def test_poll_order_fill_status_times_out_with_last_status(monkeypatch: pytest.MonkeyPatch) -> None:
    timeline = {"time": 0.0}
    sleeps: list[float] = []
    ctx = SimpleNamespace(api=SimpleNamespace(get_order=lambda _order_id: SimpleNamespace(status="new", qty="5")))

    monkeypatch.setattr(execution_flow.pytime, "time", lambda: timeline["time"])
    monkeypatch.setattr(execution_flow.pytime, "monotonic", lambda: timeline["time"])

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)
        timeline["time"] += seconds

    monkeypatch.setattr(execution_flow.pytime, "sleep", _sleep)

    execution_flow.poll_order_fill_status(ctx, "order-2", timeout=0.3)

    assert sleeps == [0.2, pytest.approx(0.1)]


def test_poll_order_fill_status_returns_on_broker_error() -> None:
    ctx = SimpleNamespace(api=SimpleNamespace(get_order=lambda _order_id: (_ for _ in ()).throw(OSError("down"))))

    execution_flow.poll_order_fill_status(ctx, "order-3", timeout=1)


def test_exit_all_positions_skips_zero_quantities_and_uses_abs_qty(monkeypatch: pytest.MonkeyPatch) -> None:
    exits: list[tuple[str, int, str]] = []
    monkeypatch.setattr(
        execution_flow,
        "send_exit_order",
        lambda _ctx, symbol, qty, _price, reason, **_kwargs: exits.append((symbol, qty, reason)),
    )
    ctx = SimpleNamespace(
        api=SimpleNamespace(
            list_positions=lambda: [
                SimpleNamespace(symbol="AAPL", qty="-3"),
                SimpleNamespace(symbol="MSFT", qty="0"),
                SimpleNamespace(symbol="GOOG", qty="2"),
            ]
        )
    )

    execution_flow.exit_all_positions(ctx)

    assert exits == [("AAPL", 3, "eod_exit"), ("GOOG", 2, "eod_exit")]


def test_liquidate_positions_if_needed_exits_active_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bot_engine, "check_halt_flag", lambda _runtime: False)
    monkeypatch.setattr(
        execution_flow,
        "_should_trigger_eod_flatten",
        lambda: (True, {"reason": "session_close_window"}),
    )
    monkeypatch.setattr(execution_flow.pytime, "monotonic", lambda: 500.0)
    calls: list[str] = []
    monkeypatch.setattr(execution_flow, "exit_all_positions", lambda _runtime: calls.append("exit"))
    runtime = SimpleNamespace(api=SimpleNamespace(list_positions=lambda: [SimpleNamespace(qty="5")]))

    execution_flow.liquidate_positions_if_needed(runtime)

    assert calls == ["exit"]
    assert runtime._last_eod_flatten_attempt_mono == 500.0


def test_exit_all_positions_routes_eod_flatten_through_canonical_execution() -> None:
    calls: list[dict[str, object]] = []
    runtime = SimpleNamespace(
        api=SimpleNamespace(
            list_positions=lambda: [
                SimpleNamespace(symbol="AAPL", qty="3"),
                SimpleNamespace(symbol="MSFT", qty="-2"),
            ]
        ),
        execute_order=lambda symbol, side, qty, **kwargs: calls.append(
            {"symbol": symbol, "side": side, "qty": qty, **kwargs}
        ),
    )

    execution_flow.exit_all_positions(runtime)

    assert calls == [
        {
            "symbol": "AAPL",
            "side": "sell",
            "qty": 3,
            "order_type": "market",
            "closing_position": True,
            "reduce_only": True,
            "metadata": {"reason": "eod_exit"},
        },
        {
            "symbol": "MSFT",
            "side": "buy",
            "qty": 2,
            "order_type": "market",
            "closing_position": True,
            "reduce_only": True,
            "metadata": {"reason": "eod_exit"},
        },
    ]


def test_liquidate_positions_if_needed_respects_halt_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bot_engine, "check_halt_flag", lambda _runtime: True)
    monkeypatch.setattr(
        execution_flow,
        "_should_trigger_eod_flatten",
        lambda: (_ for _ in ()).throw(AssertionError("should not inspect flatten window")),
    )

    execution_flow.liquidate_positions_if_needed(SimpleNamespace())
