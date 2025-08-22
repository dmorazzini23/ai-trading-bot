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


def test_start_rebalancer(monkeypatch):
    called = []

    def fake_thread(target, daemon=False):
        class T:
            def start(self):
                called.append("start-called")
                # Execute the target once instead of starting infinite loop
                try:
                    target()
                except Exception:
                    # Catch any exceptions from the loop to prevent infinite execution
                    pass
        return T()

    # Mock the infinite loop to exit after one iteration
    call_count = [0]

    def mock_maybe_rebalance(ctx):
        call_count[0] += 1
        called.append("maybe-rebalance")
        # Exit after first call to prevent infinite loop
        if call_count[0] >= 1:
            raise StopIteration("Test complete")

    monkeypatch.setattr(rebalancer.threading, "Thread", fake_thread)
    monkeypatch.setattr(rebalancer, "maybe_rebalance", mock_maybe_rebalance)

    rebalancer.start_rebalancer("ctx")
    assert "start-called" in called
