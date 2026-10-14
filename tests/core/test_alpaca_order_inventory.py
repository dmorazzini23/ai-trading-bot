"""Active inventory must remain complete across repeated broker reads."""
from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.core import alpaca_client as ac

def base():
    return datetime.now(UTC) - timedelta(days=30)


class Broker:
    def __init__(self):
        self.rows = [self.order("old", "filled", 0), self.order("newest", "filled", 10)]
        self.omitted = set()
        self.calls = []

    @staticmethod
    def order(identity, status, seconds):
        return {"id": identity, "status": status, "submitted_at": base() + timedelta(seconds=seconds)}

    def get_orders(self, *, filter):
        status = filter.status.value
        self.calls.append((status, filter.after))
        rows = sorted(self.rows, key=lambda row: row["submitted_at"] or base())
        if status == "open":
            rows = [r for r in rows if r["status"] in {"accepted", "pending_new", "partially_filled"} and r["id"] not in self.omitted]
        if filter.after:
            rows = [r for r in rows if r["submitted_at"] and r["submitted_at"] > filter.after]
        return rows[:filter.limit]


@pytest.mark.parametrize("status", ["pending_new", "accepted"])
@pytest.mark.parametrize("age_days", [0, 31, 91])
def test_newly_visible_hidden_pending_order_is_not_lost_behind_a_cursor(status, age_days):
    broker = Broker()
    assert ac.list_open_orders(broker) == []
    hidden = broker.order("hidden", status, 12)
    hidden["submitted_at"] = datetime.now(UTC) - timedelta(days=age_days)
    broker.rows.extend([broker.order("shown", "accepted", 11), hidden])
    broker.omitted.add("hidden")
    assert {r["id"] for r in ac.list_open_orders(broker)} == {"shown", "hidden"}
    status_name, after = broker.calls[-1]
    assert status_name == "all" and after is not None
    assert after <= datetime.now(UTC) - timedelta(days=91)


def test_history_scan_skips_orders_older_than_active_lifetime():
    broker = Broker()
    expired = datetime.now(UTC) - timedelta(days=400)
    broker.rows = [
        {"id": f"old{i}", "status": "filled", "submitted_at": expired + timedelta(seconds=i)}
        for i in range(2000)
    ]
    broker.rows.append(broker.order("live", "pending_new", 1))
    broker.omitted.add("live")
    assert [r["id"] for r in ac.list_open_orders(broker)] == ["live"]
    assert broker.calls == [("open", None), ("all", broker.calls[-1][1])]
    assert broker.calls[-1][1] is not None


@pytest.mark.parametrize("age_days", [None, 200])
def test_open_order_outside_lifetime_bound_forces_full_history(age_days):
    broker = Broker()
    stale = broker.order("stale", "accepted", 0)
    stale["submitted_at"] = None if age_days is None else datetime.now(UTC) - timedelta(days=age_days)
    hidden = broker.order("hidden", "pending_new", 0)
    hidden["submitted_at"] = datetime.now(UTC) - timedelta(days=300)
    broker.rows.extend([stale, hidden])
    broker.omitted.add("hidden")
    assert {r["id"] for r in ac.list_open_orders(broker)} == {"stale", "hidden"}
    assert broker.calls[-1] == ("all", None)


@pytest.mark.parametrize("initial", ["accepted", "partially_filled", "done_for_day", "held", "suspended"])
def test_old_order_status_is_read_again_after_reactivation_and_fill(initial):
    broker = Broker()
    broker.rows[0]["status"] = initial
    broker.omitted.add("old")
    ac.list_open_orders(broker)
    broker.rows[0]["status"] = "accepted"
    assert [r["id"] for r in ac.list_open_orders(broker)] == ["old"]
    broker.rows[0]["status"] = "filled"
    assert ac.list_open_orders(broker) == []


def test_more_than_500_open_orders_and_overlapping_boundary_are_complete():
    broker = Broker()
    broker.rows = [broker.order(str(i), "accepted", i) for i in range(503)]
    assert len(ac.list_open_orders(broker)) == 503
    assert sum(status == "open" for status, _ in broker.calls) == 2


def test_nonadvancing_full_page_raises():
    broker = Broker()
    broker.rows = [broker.order(str(i), "accepted", 1) for i in range(501)]
    with pytest.raises(RuntimeError, match="did not advance"):
        ac.list_open_orders(broker)


@pytest.mark.parametrize("field", ["submitted_at", "id"])
def test_full_open_page_without_pagination_evidence_raises(field):
    broker = Broker()
    broker.rows = [broker.order(str(i), "accepted", i) for i in range(500)]
    broker.rows[-1][field] = None
    with pytest.raises(RuntimeError, match="without"):
        ac.list_open_orders(broker)
