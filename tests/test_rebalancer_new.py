import rebalancer


def test_rebalance_portfolio(monkeypatch):
    """Slack alert is triggered during rebalance."""
    called = {}
    monkeypatch.setattr(rebalancer, 'send_slack_alert', lambda msg: called.setdefault('m', msg))
    rebalancer.rebalance_portfolio(None)
    assert called['m']
