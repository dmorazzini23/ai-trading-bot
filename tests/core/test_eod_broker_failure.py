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
