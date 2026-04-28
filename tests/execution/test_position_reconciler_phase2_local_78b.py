from __future__ import annotations

from types import SimpleNamespace

from ai_trading.execution import position_reconciler as pr


def test_reconcile_classification_resolution_history_and_force_sync() -> None:
    reconciler = pr.PositionReconciler()
    reconciler.update_bot_position("AAPL", 5)
    reconciler.adjust_bot_position("MSFT", -2)
    reconciler.get_broker_positions = lambda: {"AAPL": 7, "MSFT": 0, "GOOG": 1}

    discrepancies = reconciler.reconcile_positions()

    assert {d.symbol: d.discrepancy_type for d in discrepancies} == {
        "AAPL": "quantity_mismatch",
        "MSFT": "phantom_position",
        "GOOG": "missing_position",
    }
    assert reconciler.get_current_discrepancies() == discrepancies
    assert reconciler.get_reconciliation_history(limit=1)[0]["discrepancies_count"] == 3
    assert reconciler.get_discrepancy_history(symbol="GOOG")[0].severity == "medium"

    resolved = reconciler.auto_resolve_discrepancies(discrepancies)
    assert resolved == 3
    assert reconciler.get_bot_positions() == {"AAPL": 7, "MSFT": 0, "GOOG": 1}

    reconciler.get_broker_positions = lambda: {"TSLA": 4}
    assert reconciler.force_sync_from_broker() == {"TSLA": 4}
    stats = reconciler.get_reconciliation_stats()
    assert stats["current_discrepancies"] == 3
    assert stats["severity_breakdown"]["medium"] == 3


def test_high_severity_and_direction_mismatch_are_not_auto_resolved() -> None:
    reconciler = pr.PositionReconciler()
    discrepancy = pr.PositionDiscrepancy("SPY", 20, -5, reconciler._classify_discrepancy(20, -5), "high")

    assert discrepancy.discrepancy_type == "direction_mismatch"
    assert discrepancy.difference == -25
    assert reconciler._determine_severity("SPY", 0, 0.5) == "low"
    assert reconciler._determine_severity("SPY", 0, 25) == "high"
    assert reconciler.auto_resolve_discrepancies([discrepancy]) == 0
    assert reconciler.get_bot_positions() == {}


def test_periodic_reconciliation_thread_and_loop_branches(monkeypatch) -> None:
    reconciler = pr.PositionReconciler()
    started: list[str] = []

    class FakeThread:
        def __init__(self, target, daemon, name):
            self.target = target
            self.daemon = daemon
            self.name = name

        def start(self):
            started.append(self.name)

        def join(self, timeout=None):
            started.append(f"join:{timeout}")

    monkeypatch.setattr(pr, "Thread", FakeThread)
    reconciler.start_periodic_reconciliation(interval=1)
    reconciler.start_periodic_reconciliation(interval=2)
    assert started == ["PositionReconciler"]
    assert reconciler.running is True
    reconciler.stop_periodic_reconciliation()
    assert started[-1] == "join:10"

    loop_reconciler = pr.PositionReconciler()
    loop_reconciler.running = True
    low = pr.PositionDiscrepancy("AAPL", 0, 1, "missing_position", "low")
    loop_reconciler.reconcile_positions = lambda: [low]
    loop_reconciler.auto_resolve_discrepancies = lambda discrepancies: len(discrepancies)
    monkeypatch.setattr(pr.time, "sleep", lambda _seconds: setattr(loop_reconciler, "running", False))
    loop_reconciler._reconciliation_loop()

    error_reconciler = pr.PositionReconciler()
    error_reconciler.running = True

    def fail_reconcile():
        raise ValueError("bad reconciliation")

    error_reconciler.reconcile_positions = fail_reconcile
    monkeypatch.setattr(pr.time, "sleep", lambda _seconds: setattr(error_reconciler, "running", False))
    error_reconciler._reconciliation_loop()


def test_global_position_reconciler_wrappers(monkeypatch) -> None:
    reconciler = pr.PositionReconciler()
    monkeypatch.setattr(pr, "_position_reconciler", reconciler)

    pr.update_bot_position("AAPL", 3, reason="test")
    pr.adjust_bot_position("AAPL", 2, reason="fill")
    assert reconciler.get_bot_positions() == {"AAPL": 5}

    reconciler.get_broker_positions = lambda: {"AAPL": 5}
    assert pr.force_position_reconciliation() == []
    pr.start_position_monitoring(interval=7)
    assert reconciler.running is True
    pr.stop_position_monitoring()
    assert pr.get_position_discrepancies() == []
    assert pr.get_reconciliation_statistics()["bot_positions_count"] == 1


def test_get_broker_positions_fetches_real_positions_and_signs_shorts() -> None:
    api = SimpleNamespace(
        get_all_positions=lambda: [
            SimpleNamespace(symbol="aapl", qty="5", side="long"),
            SimpleNamespace(symbol="tsla", qty="3", side="short"),
        ],
    )
    reconciler = pr.PositionReconciler(api_client=api)

    assert reconciler.get_broker_positions() == {"AAPL": 5.0, "TSLA": -3.0}


def test_reconcile_skips_authoritative_empty_on_broker_fetch_error() -> None:
    class FailingApi:
        def get_all_positions(self):
            raise TimeoutError("broker timeout")

    reconciler = pr.PositionReconciler(api_client=FailingApi())
    reconciler.update_bot_position("AAPL", 5)
    reconciler._broker_positions = {"AAPL": 5.0}

    assert reconciler.reconcile_positions() == []
    assert reconciler.get_bot_positions() == {"AAPL": 5}
    assert reconciler.force_sync_from_broker() == {"AAPL": 5}
