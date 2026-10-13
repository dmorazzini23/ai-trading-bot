from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from ai_trading.broker.adapters import list_alpaca_orders
from ai_trading.core.alpaca_client import list_open_orders


def test_symbol_lookup_pages_complete_history_without_scanning_other_symbols():
    start = datetime(2026, 1, 2, tzinfo=UTC)
    rows = [SimpleNamespace(id=str(i), symbol="AMZN", status="filled", submitted_at=start+timedelta(seconds=i)) for i in range(501)]
    rows[-1].status = "pending_new"
    rows.extend(SimpleNamespace(id=f"other-{i}", symbol="ZZZZ", status="filled", submitted_at=start+timedelta(seconds=i)) for i in range(2000))
    outside = SimpleNamespace(id="outside", symbol="ZZZZ", status="accepted", submitted_at=start+timedelta(days=1))
    rows.append(outside)
    class Api:
        def __init__(self):
            self.calls = []
        def get_orders(self, *, filter):
            status = getattr(filter.status, "value", filter.status)
            self.calls.append((status, filter.symbols))
            selected = [r for r in rows if not filter.symbols or r.symbol in filter.symbols]
            if status == "open":
                return [r for r in selected if r is outside]
            selected = sorted(selected, key=lambda r:r.submitted_at)
            return [r for r in selected if filter.after is None or r.submitted_at > filter.after][:500]
    api = Api()
    assert [r.id for r in list_alpaca_orders(api, status="open", symbols=["AMZN"])] == ["500"]
    assert api.calls == [("open", ["AMZN"]), ("all", ["AMZN"]), ("all", ["AMZN"])]
    # Unscoped account safety reads still discover outside-universe orders.
    assert {r.id for r in list_open_orders(api)} == {"500", "outside"}


def test_scoped_lookup_still_merges_missing_old_timestamp_order():
    class Api:
        def get_orders(self, *, filter):
            assert filter.symbols == ["AMZN"]
            status = getattr(filter.status, "value", filter.status)
            return [SimpleNamespace(id="new", status="new", symbol="AMZN")] if status == "open" else [SimpleNamespace(id="old", status="pending_new", submitted_at=None, symbol="AMZN")]
    assert {r.id for r in list_open_orders(Api(), symbols=["AMZN"])} == {"new", "old"}
