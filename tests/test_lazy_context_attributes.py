import importlib
import types


def test_lazy_context_portfolio_and_rebalance_attrs(monkeypatch):
    be = importlib.reload(__import__('ai_trading.core.bot_engine', fromlist=['dummy']))

    def fake_ensure(self):
        self._context = types.SimpleNamespace(
            portfolio_weights={},
            rebalance_buys={},
            rebalance_ids={},
            rebalance_attempts={},
            trade_logger=None,
            allocator=None,
            strategies=[],
            drawdown_circuit_breaker=None,
            logger=None,
            tickers=[]
        )
        self._initialized = True

    monkeypatch.setattr(be.LazyBotContext, '_ensure_initialized', fake_ensure)
    lbc = be.LazyBotContext()

    assert lbc.portfolio_weights == {}
    assert lbc.rebalance_buys == {}
    assert lbc.rebalance_ids == {}
    assert lbc.rebalance_attempts == {}
    assert lbc.trade_logger is None
    assert lbc.allocator is None
    assert lbc.strategies == []
    assert lbc.drawdown_circuit_breaker is None
    assert lbc.logger is None
    assert lbc.tickers == []
