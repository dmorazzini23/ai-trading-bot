from types import SimpleNamespace

from alpaca.common.exceptions import APIError

from ai_trading.core import execution_flow


def test_ambiguous_exit_failure_preserves_identity_and_continues(caplog):
    calls = []

    def execute(symbol, side, qty, **kwargs):
        calls.append((symbol, kwargs['client_order_id']))
        if symbol == 'AAPL':
            raise APIError('{"code":504,"message":"Gateway Timeout"}')

    ctx = SimpleNamespace(
        api=SimpleNamespace(get_all_positions=lambda: [
            SimpleNamespace(symbol='AAPL', qty=1),
            SimpleNamespace(symbol='MSFT', qty=1),
        ]),
        execute_order=execute,
    )
    execution_flow.exit_all_positions(ctx)
    execution_flow.exit_all_positions(ctx)
    assert [symbol for symbol, _ in calls] == ['AAPL', 'MSFT', 'AAPL', 'MSFT']
    assert calls[0][1] == calls[2][1]
    assert calls[1][1] == calls[3][1]
    assert 'EOD_EXIT_SUBMISSION_UNCONFIRMED' in caplog.text


def _eod_ctx(calls, cancels, positions):
    return SimpleNamespace(
        api=SimpleNamespace(
            get_all_positions=lambda: positions,
            cancel_order_by_id=lambda order_id: cancels.append(order_id),
        ),
        execute_order=lambda symbol, side, qty, **kwargs: calls.append((symbol, side, qty, kwargs["client_order_id"])),
    )


def test_eod_cancels_pending_strategy_order_then_exits_next_attempt(monkeypatch, caplog):
    from ai_trading.core import alpaca_client

    calls, cancels = [], []
    open_orders = [SimpleNamespace(id="lim-1", symbol="MSFT", side="sell", client_order_id="25da9023c04c16f8")]
    monkeypatch.setattr(alpaca_client, "list_open_orders", lambda api, symbols=None: list(open_orders))
    ctx = _eod_ctx(calls, cancels, [SimpleNamespace(symbol="MSFT", qty=1), SimpleNamespace(symbol="AAPL", qty=2)])

    execution_flow.exit_all_positions(ctx)

    assert cancels == ["lim-1"]
    assert [c[0] for c in calls] == ["AAPL"]  # MSFT deferred, AAPL unaffected
    assert "EOD_EXIT_DEFERRED_PENDING_CANCEL" in caplog.text

    open_orders.clear()  # cancel confirmed before the next EOD attempt
    execution_flow.exit_all_positions(ctx)
    assert ("MSFT", "sell", 1) in [c[:3] for c in calls]
    assert cancels == ["lim-1"]


def test_eod_does_not_cancel_its_own_exit_order(monkeypatch):
    from ai_trading.core import alpaca_client

    calls, cancels = [], []
    own = SimpleNamespace(id="eod-1", symbol="MSFT", side="sell", client_order_id="eod-2026-10-16-MSFT-sell")
    monkeypatch.setattr(alpaca_client, "list_open_orders", lambda api, symbols=None: [own])
    ctx = _eod_ctx(calls, cancels, [SimpleNamespace(symbol="MSFT", qty=1)])

    execution_flow.exit_all_positions(ctx)

    assert cancels == []
    assert [c[:3] for c in calls] == [("MSFT", "sell", 1)]
    assert calls[0][3].startswith("eod-")


def test_eod_cancel_failure_still_defers_without_selling(monkeypatch, caplog):
    from alpaca.common.exceptions import APIError as _APIError
    from ai_trading.core import alpaca_client

    calls = []
    monkeypatch.setattr(
        alpaca_client,
        "list_open_orders",
        lambda api, symbols=None: [SimpleNamespace(id="lim-2", symbol="AAPL", side="sell", client_order_id="x")],
    )

    def failing_cancel(order_id):
        raise _APIError('{"code":500,"message":"boom"}')

    ctx = SimpleNamespace(
        api=SimpleNamespace(get_all_positions=lambda: [SimpleNamespace(symbol="AAPL", qty=1)], cancel_order_by_id=failing_cancel),
        execute_order=lambda *a, **k: calls.append(a),
    )
    execution_flow.exit_all_positions(ctx)

    assert calls == []
    assert "EOD_PENDING_ORDER_CANCEL_FAILED" in caplog.text
