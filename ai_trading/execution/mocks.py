class MockTradingClient:
    """Minimal mock used by live_trading when PYTEST_RUNNING is set."""
    # AI-AGENT-REF: internal mock for tests
    def __init__(self, *args, **kwargs):
        pass

    def submit_order(self, *args, **kwargs):
        return {"id": "MOCK_ORDER", "status": "accepted"}

    def get_account(self):
        return type("Acct", (), {"cash": "100000", "status": "ACTIVE"})()

    def get_all_positions(self):
        return []
