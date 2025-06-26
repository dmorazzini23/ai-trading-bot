import types
import rebalancer


def test_maybe_rebalance_triggers(monkeypatch):
    calls = []
    monkeypatch.setattr(rebalancer, "REBALANCE_INTERVAL_MIN", 1)
    rebalancer._last_rebalance = rebalancer.datetime.now(rebalancer.timezone.utc) - rebalancer.timedelta(minutes=2)
    monkeypatch.setattr(rebalancer, "rebalance_portfolio", lambda ctx: calls.append(ctx))
    rebalancer.maybe_rebalance("ctx")
    assert calls == ["ctx"]


def test_maybe_rebalance_skip(monkeypatch):
    calls = []
    rebalancer._last_rebalance = rebalancer.datetime.now(rebalancer.timezone.utc)
    monkeypatch.setattr(rebalancer, "rebalance_portfolio", lambda ctx: calls.append(ctx))
    rebalancer.maybe_rebalance("ctx")
    assert calls == []


def test_start_rebalancer(monkeypatch):
    called = []
    def fake_thread(target, daemon=False):
        class T:
            def start(self):
                target()
        return T()
    monkeypatch.setattr(rebalancer.threading, "Thread", fake_thread)
    monkeypatch.setattr(rebalancer, "maybe_rebalance", lambda ctx: called.append(True))
    t = rebalancer.start_rebalancer("ctx")
    assert called and isinstance(t, object)
