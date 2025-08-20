from ai_trading.execution.engine import ExecutionEngine


class DummyBroker:
    def list_positions(self):
        return []


def test_check_stops_noop():
    eng = ExecutionEngine()
    eng.broker = DummyBroker()
    eng.check_stops()  # should not raise

