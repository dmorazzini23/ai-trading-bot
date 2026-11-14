from ai_trading import rebalancer


def test_maybe_rebalance_triggers(monkeypatch):
    calls = []
    monkeypatch.setattr(rebalancer, "rebalance_interval_min", lambda: 1)
    rebalancer._last_rebalance = rebalancer.datetime.now(rebalancer.UTC) - rebalancer.timedelta(minutes=2)
    monkeypatch.setattr(rebalancer, "rebalance_portfolio", lambda ctx: calls.append(ctx))
    rebalancer.maybe_rebalance("ctx")
    assert calls == ["ctx"]


def test_maybe_rebalance_skip(monkeypatch):
    calls = []
    rebalancer._last_rebalance = rebalancer.datetime.now(rebalancer.UTC)
    monkeypatch.setattr(rebalancer, "rebalance_portfolio", lambda ctx: calls.append(ctx))
    rebalancer.maybe_rebalance("ctx")
    assert calls == []


