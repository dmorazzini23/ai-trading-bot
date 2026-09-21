from types import SimpleNamespace as NS
from unittest.mock import Mock

import pandas as pd
import pytest

from ai_trading import signals
from ai_trading.core import bot_engine as bot, strategy_cycle


@pytest.fixture
def cycle(monkeypatch):
    signal = NS(symbol='AAPL', side='buy', strength=1., confidence=.8,
                weight=.2, asset_class='equity', strategy='test')
    result = NS(side='buy', reconciled=True, filled_quantity=2, requested_quantity=4)
    engine = NS(execute_order=Mock(return_value=result), end_cycle=Mock(),
                open_order_totals=Mock(return_value=(0, 0)), mark_fill_reported=Mock())
    risk = NS(position_exists=Mock(return_value=False), position_size=Mock(return_value=4),
              register_fill=Mock())
    ctx = NS(strategies=[NS(name='test', generate=lambda ctx: [signal])],
             allocator=NS(allocate=lambda grouped: grouped['test']),
             api=NS(list_positions=lambda: [], get_account=lambda: NS(cash=1000)),
             execution_engine=engine, risk_engine=risk, portfolio_weights={'AAPL': .2},
             stop_targets={'AAPL': 95.}, take_profit_targets={'AAPL': 110.})
    monkeypatch.setattr(bot, 'RL_AGENT', None)
    monkeypatch.setattr(signals, 'generate_position_hold_signals', lambda *a: [])
    monkeypatch.setattr(signals, 'enhance_signals_with_position_logic', lambda values, *a: values)
    monkeypatch.setattr(bot, 'to_trade_signal', lambda value: value)
    monkeypatch.setattr(bot, '_dedupe_cycle_intents', lambda values: (values, {}))
    monkeypatch.setattr(bot, '_signal_strength_threshold', lambda ctx: .1)
    monkeypatch.setattr(bot, 'get_latest_price', lambda symbol: 100.)
    monkeypatch.setattr(bot, 'get_price_source', lambda symbol: 'alpaca')
    monkeypatch.setattr(bot, '_is_reliable_quote', lambda *args: True)
    monkeypatch.setattr(bot, '_resolve_limit_price', lambda *args: (100., 'alpaca'))
    monkeypatch.setattr(bot, '_current_qty', lambda *args: 0)
    monkeypatch.setattr(bot, '_resolve_order_intent', lambda *a, **k:
                        bot.OrderIntentDecision(True, 'buy', ('buy',), None, {}))
    return ctx, signal, result


def test_partial_fill_registers_only_filled_weight_and_bracket(cycle):
    ctx, signal, result = cycle
    strategy_cycle.run_multi_strategy_cycle(ctx)
    call = ctx.execution_engine.execute_order.call_args
    assert call is not None
    assert call.kwargs['stop_loss'] == 95.
    assert call.kwargs['take_profit'] == 110.
    assert call.kwargs['order_class'] == 'bracket'
    filled = ctx.risk_engine.register_fill.call_args.args[0]
    assert filled.weight == pytest.approx(.1)
    assert signal.weight == .2
    ctx.execution_engine.mark_fill_reported.assert_called_once_with(str(result), 2)
    ctx.execution_engine.end_cycle.assert_called_once()


@pytest.mark.parametrize('condition', ['no_price', 'duplicate', 'weak', 'zero_qty',
    'nan_qty', 'invalid_side', 'blocked_intent', 'pending_order'])
def test_pretrade_blocks_never_submit(cycle, monkeypatch, condition):
    ctx, signal, _ = cycle
    if condition == 'no_price':
        monkeypatch.setattr(bot, '_resolve_limit_price', lambda *a: (None, None))
        monkeypatch.setattr(bot, 'fetch_minute_df_safe', lambda *a: pd.DataFrame())
    elif condition == 'duplicate':
        ctx.risk_engine.position_exists.return_value = True
    elif condition == 'weak':
        signal.strength = .01
    elif condition in {'zero_qty', 'nan_qty'}:
        ctx.risk_engine.position_size.return_value = 0 if condition == 'zero_qty' else float('nan')
    elif condition == 'invalid_side':
        signal.side = 'hold'
    elif condition == 'blocked_intent':
        monkeypatch.setattr(bot, '_resolve_order_intent', lambda *a, **k:
                            bot.OrderIntentDecision(False, None, (), 'cycle_duplicate', {}))
    else:
        ctx.execution_engine.open_order_totals.return_value = (4, 0)
    strategy_cycle.run_multi_strategy_cycle(ctx)
    ctx.execution_engine.execute_order.assert_not_called()
    ctx.risk_engine.register_fill.assert_not_called()


@pytest.mark.parametrize('condition', ['unreconciled', 'unfilled', 'wrong_side', 'no_result', 'zero_weight'])
def test_unconfirmed_execution_is_not_registered_as_fill(cycle, condition):
    ctx, signal, result = cycle
    if condition == 'unreconciled':
        result.reconciled = False
    elif condition == 'unfilled':
        result.filled_quantity = 0
    elif condition == 'wrong_side':
        result.side = 'sell'
    elif condition == 'no_result':
        ctx.execution_engine.execute_order.return_value = None
    else:
        signal.weight = 0
    strategy_cycle.run_multi_strategy_cycle(ctx)
    ctx.execution_engine.execute_order.assert_called_once()
    ctx.risk_engine.register_fill.assert_not_called()


def test_fallback_price_gaps_block_execution(cycle, monkeypatch):
    ctx, _, _ = cycle
    frame = pd.DataFrame({'close': [100.]}, index=pd.date_range('2026-09-21', periods=1, tz='UTC'))
    frame.attrs['_coverage_meta'] = {'gap_ratio': .8}
    monkeypatch.setattr(bot, '_is_reliable_quote', lambda *a: False)
    monkeypatch.setattr(bot, 'fetch_minute_df_safe', lambda *a: frame)
    monkeypatch.setattr(bot, 'should_skip_symbol', lambda *a, **k: True)
    strategy_cycle.run_multi_strategy_cycle(ctx)
    ctx.execution_engine.execute_order.assert_not_called()
