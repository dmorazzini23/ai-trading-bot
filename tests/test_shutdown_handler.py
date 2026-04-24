from __future__ import annotations

import asyncio
import signal

from ai_trading import shutdown_handler as sh


def _handler(monkeypatch):
    monkeypatch.setattr(sh.signal, "signal", lambda sig, handler: f"old-{sig}")
    return sh.ShutdownHandler()


async def _noop(*_args, **_kwargs) -> None:
    return None


def test_shutdown_handler_registration_and_state_helpers(monkeypatch):
    handler = _handler(monkeypatch)
    calls: list[str] = []

    def pre_hook():
        calls.append("pre")

    def position_handler():
        return [{"symbol": "AAPL"}]

    def order_handler():
        return [{"id": "order-1"}]

    handler.register_pre_shutdown_hook(None)
    handler.register_pre_shutdown_hook(pre_hook)
    handler.register_position_handler(position_handler)
    handler.register_order_handler(order_handler)
    handler.register_cleanup_hook(pre_hook)
    handler.register_post_shutdown_hook(pre_hook)

    positions = [{"symbol": "MSFT"}]
    orders = [{"id": "order-2"}]
    handler.update_system_state({"mode": "paper"})
    handler.set_active_positions(positions)
    handler.set_pending_orders(orders)
    positions.append({"symbol": "TSLA"})
    orders.append({"id": "order-3"})

    assert len(handler._pre_shutdown_hooks) == 1
    assert len(handler._position_handlers) == 1
    assert len(handler._order_handlers) == 1
    assert handler._system_state == {"mode": "paper"}
    assert handler._active_positions == [{"symbol": "MSFT"}]
    assert handler._pending_orders == [{"id": "order-2"}]
    assert not handler.is_shutting_down()
    assert not handler.wait_for_shutdown(timeout=0)
    assert handler.get_shutdown_status().current_action == "System running normally"


def test_graceful_shutdown_runs_hooks_and_cancels_orders(monkeypatch):
    handler = _handler(monkeypatch)
    events: list[str] = []

    def pre_hook():
        events.append("pre")

    async def cleanup_hook():
        events.append("cleanup")

    def post_hook():
        events.append("post")

    async def cancel_order(order):
        events.append(f"cancel:{order['id']}")

    async def save_state():
        events.append("save")

    handler.register_pre_shutdown_hook(pre_hook)
    handler.register_order_handler(lambda: [{"id": "order-1"}, {"id": "order-2"}])
    handler.register_cleanup_hook(cleanup_hook)
    handler.register_post_shutdown_hook(post_hook)
    handler._cancel_single_order = cancel_order
    handler._save_system_state = save_state

    ok = asyncio.run(handler.shutdown(sh.ShutdownReason.SCHEDULED_MAINTENANCE))

    assert ok
    assert handler.get_shutdown_status().phase is sh.ShutdownPhase.COMPLETED
    assert handler.get_shutdown_status().orders_to_cancel == 2
    assert handler.get_shutdown_status().orders_canceled == 2
    assert events == ["pre", "cancel:order-1", "cancel:order-2", "save", "cleanup", "post"]
    assert handler.wait_for_shutdown(timeout=0)


def test_graceful_shutdown_refuses_second_shutdown(monkeypatch):
    handler = _handler(monkeypatch)
    handler._status.is_shutting_down = True

    assert asyncio.run(handler.shutdown()) is False


def test_cancel_pending_orders_records_handler_and_cancel_errors(monkeypatch):
    handler = _handler(monkeypatch)

    def broken_handler():
        raise OSError("order source down")

    async def cancel_order(order):
        if order["id"] == "bad":
            raise TimeoutError("cancel timeout")

    handler.register_order_handler(broken_handler)
    handler.register_order_handler(lambda: [{"id": "bad"}, {"id": "good"}])
    handler._cancel_single_order = cancel_order

    ok = asyncio.run(handler._cancel_pending_orders())

    assert not ok
    assert handler.get_shutdown_status().orders_to_cancel == 2
    assert handler.get_shutdown_status().orders_canceled == 1
    assert any("Order handler error" in err for err in handler.get_shutdown_status().errors)
    assert any("Order cancel error" in err for err in handler.get_shutdown_status().errors)


def test_close_positions_records_handler_and_close_errors(monkeypatch):
    handler = _handler(monkeypatch)

    def broken_handler():
        raise ConnectionError("position source down")

    async def close_position(position):
        if position["symbol"] == "BAD":
            raise TimeoutError("close timeout")

    handler.register_position_handler(broken_handler)
    handler.register_position_handler(lambda: [{"symbol": "BAD"}, {"symbol": "GOOD"}])
    handler._close_single_position = close_position

    ok = asyncio.run(handler._close_positions())

    assert not ok
    assert handler.get_shutdown_status().positions_to_close == 2
    assert handler.get_shutdown_status().positions_closed == 1
    assert any("Position handler error" in err for err in handler.get_shutdown_status().errors)
    assert any("Position close error" in err for err in handler.get_shutdown_status().errors)


def test_emergency_shutdown_runs_critical_steps(monkeypatch):
    handler = _handler(monkeypatch)
    events: list[str] = []

    async def cancel_orders():
        events.append("cancel")
        return True

    async def save_critical():
        events.append("save")

    async def emergency_cleanup():
        events.append("cleanup")

    handler._cancel_pending_orders = cancel_orders
    handler._save_critical_state = save_critical
    handler._emergency_cleanup = emergency_cleanup

    ok = asyncio.run(handler.shutdown(sh.ShutdownReason.EMERGENCY_STOP, emergency=True))

    assert ok
    assert events == ["cancel", "save", "cleanup"]
    assert handler.get_shutdown_status().phase is sh.ShutdownPhase.COMPLETED
    assert handler.get_shutdown_status().estimated_completion is not None


def test_signal_setup_failure_and_signal_handler_delegation(monkeypatch):
    calls: list[str] = []

    def fail_signal(*_args):
        raise OSError("signals unavailable")

    monkeypatch.setattr(sh.signal, "signal", fail_signal)
    handler = sh.ShutdownHandler()
    assert handler.get_shutdown_status().phase is sh.ShutdownPhase.INITIATED

    async def fake_shutdown(reason):
        calls.append(reason.value)

    def fake_create_task(coro):
        coro.close()
        calls.append("scheduled")
        return None

    monkeypatch.setattr(handler, "shutdown", fake_shutdown)
    monkeypatch.setattr(sh.asyncio, "create_task", fake_create_task)

    handler._signal_handler(signal.SIGTERM, None)

    assert calls == ["scheduled"]


def test_global_shutdown_helpers_reuse_single_handler(monkeypatch):
    handler = _handler(monkeypatch)
    monkeypatch.setattr(sh, "_shutdown_handler", handler)
    registered: list[str] = []

    def pre_shutdown():
        registered.append("pre")

    sh.register_shutdown_hooks(pre_shutdown=pre_shutdown)

    assert sh.get_shutdown_handler() is handler
    assert len(handler._pre_shutdown_hooks) == 1
    assert not sh.is_shutting_down()
