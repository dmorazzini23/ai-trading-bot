from datetime import UTC, datetime

import pytest

from ai_trading.tools.paper_evidence_review import cap_selection_review, compare_execution_costs, latest_completed_session


def test_cost_pairs_require_account_and_matching_benchmark():
    simulated, fills = [], []
    for i in range(30):
        key = str(i)
        simulated.append({"client_order_id": key, "symbol": "AAPL", "side": "buy", "reference_price": 100, "fill_price": 100.2, "fill_qty": 2})
        fills.append({"client_order_id": key, "fill_id": key, "account_id": "paper-test", "trading_mode": "paper", "symbol": "AAPL", "side": "buy", "expected_price": 100, "fill_price": 100.1, "fill_qty": 2, "ts": f"2026-08-{3+i%5:02d}T15:00:00Z", "fee_amount": 0, "fee_source": "broker_payload"})
    replay = {"replay_summary": {"markout_observations": simulated}}
    result = compare_execution_costs(replay, fills, account_id="paper-test")
    assert result["status"] == "comparison_available"
    assert result["sessions"] == 5
    assert result["mean_slippage_error_bps"] == pytest.approx(10)
    assert result["pairs"][0]["observed_fee_bps"] == 0
    mismatched = [{**row, "expected_price": 99} for row in fills]
    result = compare_execution_costs(replay, mismatched, account_id="paper-test")
    assert result["paired_orders"] == 0
    assert result["rejection_counts"]["arrival_benchmark_or_order_contract_mismatch"] == 30
    result = compare_execution_costs(replay, fills, account_id="different")
    assert result["paired_orders"] == 0
    conflicting = [*fills, {**fills[0], "fill_price": 110}]
    assert compare_execution_costs(replay, conflicting, account_id="paper-test")["paired_orders"] == 29


def test_cap_diagnosis_does_not_select_winners_from_realized_markouts():
    base = [{"client_order_id": "bad", "symbol": "MSFT", "net_edge_bps": -2}, {"client_order_id": "good", "symbol": "MSFT", "net_edge_bps": 1}]
    report = cap_selection_review({"baseline_summary": {"markout_observations": base}, "replay_summary": {"markout_observations": base[:1]}, "cap_adjustments": [{"cap_kind": "symbol"}]})
    assert report["symbols"][0]["retained_minus_removed_bps"] == -3
    assert report["limits_changed"] is False
    assert report["adjustments_by_cap_kind"] == {"symbol": 1}


def test_session_selection_waits_for_close_and_handles_holiday():
    assert latest_completed_session(datetime(2026, 9, 7, 22, tzinfo=UTC)) == "2026-09-04"
    assert latest_completed_session(datetime(2026, 9, 8, 19, tzinfo=UTC)) == "2026-09-04"
    assert latest_completed_session(datetime(2026, 9, 8, 21, tzinfo=UTC)) == "2026-09-08"


def test_replay_fill_prices_do_not_depend_on_cross_symbol_timestamp_order():
    from ai_trading.execution.simulated_broker import SimulatedBroker
    from ai_trading.replay.event_loop import ReplayEventLoop

    bars = [{"ts": ts, "symbol": symbol, "close": price, "client_order_id": symbol + ts} for ts, price in [("2026-08-03T15:00:00Z", 100), ("2026-08-03T15:05:00Z", 110)] for symbol in ["AAPL", "MSFT"]]

    def run(rows):
        def strategy(bar):
            if bar["ts"].endswith("05:00Z"):
                return None
            return {"symbol": bar["symbol"], "side": "buy", "type": "market", "qty": 1, "price": 100, "client_order_id": bar["client_order_id"]}
        replay = ReplayEventLoop(strategy=strategy, broker=SimulatedBroker(seed=42, fill_probability=1, partial_fill_probability=0), max_symbol_notional=25000, max_gross_notional=150000).run(rows)
        return {row["symbol"]: row["fill_price"] for row in replay["events"] if row["event_type"] == "fill"}

    assert run(bars) == run(list(reversed(bars)))
    assert all(price > 109 for price in run(bars).values())
    with pytest.raises(ValueError, match="conflicting"):
        run([*bars, {**bars[0], "close": 90}])
