from datetime import UTC, datetime
import json

import pytest

from ai_trading.tools.paper_evidence_review import cap_selection_review, compare_execution_costs, completion_gaps, latest_completed_session


def test_slippage_support_does_not_complete_missing_net_cost_validation():
    audit = {"status": "paper_session_reconciled"}
    costs = {"status": "comparison_available", "net_cost_validation": "unavailable_without_simulated_and_observed_fee_contracts"}
    assert completion_gaps(audit, costs, []) == ["net_cost_validation_unavailable"]
    assert completion_gaps(audit, {"status": "comparison_available"}, []) == ["net_cost_validation_unavailable"]
    validated = {"status": "comparison_available", "net_cost_validation": "validated"}
    assert completion_gaps(audit, validated, []) == []
    assert completion_gaps(audit, validated, ["fills"]) == ["input_evidence_gaps"]
    assert completion_gaps({"status": "evidence_gaps"}, validated, []) == ["paper_session_not_reconciled"]


def test_daily_cli_keeps_supported_session_and_slippage_pending_without_fee_contract(tmp_path, monkeypatch):
    from ai_trading.tools.paper_evidence_review import main

    fills = [{"fill_id": str(i), "order_id": str(i), "client_order_id": str(i), "account_id": "test", "trading_mode": "paper", "symbol": "AAPL", "side": "buy", "fill_qty": 1, "fill_price": 100.1, "expected_price": 100, "ts": f"2026-08-{3+i%5:02d}T15:00:00Z", "fee_amount": 0, "fee_source": "broker_payload"} for i in range(30)]
    boundaries = [{"timestamp": ts, "account_id": "test", "trading_mode": "paper", "positions_complete": True, "positions": {"AAPL": qty}} for ts, qty in [("2026-08-07T13:30:00Z", 24), ("2026-08-07T20:00:00Z", 30)]]
    orders = [{"order_id": row["order_id"], "ts": row["ts"], "filled_qty": 1} for row in fills]
    for name, rows in {"decision_records": fills, "order_events": orders, "fill_events": fills, "tca_records": fills, "broker_position_boundaries": boundaries}.items():
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps({"replay_summary": {"markout_observations": [{**row, "reference_price": 100, "fill_price": 100.2} for row in fills]}}))
    output = tmp_path / "review.json"
    training = tmp_path / "training.json"
    training.write_text(json.dumps({"status": "complete", "candidates": [{"model_id": "fresh"}], "data_provenance": {"quality_passed": True}}))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"status": "blocked", "decisions": [{"candidate_model_id": "fresh", "samples": 0, "reasons": ["insufficient_samples"]}]}))
    monkeypatch.setattr("sys.argv", ["paper_evidence_review", "--runtime-dir", str(tmp_path), "--replay-report", str(replay), "--training-report", str(training), "--selection-report", str(selection), "--session", "2026-08-07", "--output", str(output)])
    main()
    report = json.loads(output.read_text())
    assert report["session_audit"]["status"] == "paper_session_reconciled"
    assert report["execution_cost_comparison"]["status"] == "comparison_available"
    assert report["status"] == "evidence_pending"
    assert report["completion_gaps"] == ["net_cost_validation_unavailable"]
    assert report["training_readiness"]["candidate_count"] == 1
    assert report["training_readiness"]["selection_decisions"][0]["reasons"] == ["insufficient_samples"]
    assert report["training_readiness"]["runtime_model_replaced"] is False
    for row in fills:
        row.update(fee_currency="USD", fee_basis="per_fill_total")
    (tmp_path / "fill_events.jsonl").write_text("\n".join(json.dumps(row) for row in fills) + "\n")
    replay.write_text(json.dumps({"replay_summary": {"markout_observations": [{**row, "reference_price": 100, "fill_price": 100.2, "fee_source": "simulated_notional_bps_v1", "fee_rate_bps": 0} for row in fills]}}))
    main()
    report = json.loads(output.read_text())
    assert report["status"] == "review_complete"
    assert report["execution_cost_comparison"]["net_cost_validation"] == "validated"
    assert report["execution_cost_comparison"]["mean_net_cost_error_bps"] == pytest.approx(10)


@pytest.mark.parametrize("field,value", [("fee_amount", None), ("fee_amount", -1), ("fee_amount", float("nan")), ("fee_currency", "EUR"), ("fee_basis", "cumulative_order"), ("fee_source", "configured_estimate")])
def test_net_cost_contract_rejects_unknown_or_incomparable_fees(field, value):
    from ai_trading.tools.paper_evidence_review import _fee_total
    row = {"fee_amount": 0, "fee_currency": "USD", "fee_basis": "per_fill_total", "fee_source": "broker_payload"}
    assert _fee_total([row], simulated=False) == 0
    assert _fee_total([{**row, field: value}], simulated=False) is None


def test_net_cost_accounts_for_partial_fills_and_each_quantity():
    simulated, fills = [], []
    for i in range(30):
        row = {"client_order_id": str(i), "symbol": "AAPL", "side": "sell", "fill_qty": 2, "fill_price": 99, "reference_price": 100, "fee_amount": .0198, "fee_currency": "USD", "fee_basis": "per_fill_total", "fee_source": "simulated_notional_bps_v1", "fee_rate_bps": 1}
        simulated.append(row)
        for part in range(2):
            fills.append({**row, "fill_id": f"{i}-{part}", "fill_qty": 1, "fill_price": 98, "expected_price": 100, "fee_amount": .02, "fee_source": "broker_activity", "ts": f"2026-08-{3+i%5:02d}T15:00:00Z", "trading_mode": "paper", "account_id": "test"})
    result = compare_execution_costs({"replay_summary": {"markout_observations": simulated}}, fills, account_id="test")
    assert result["net_cost_validation"] == "validated"
    assert result["mean_net_cost_error_bps"] == pytest.approx(100.99 - 202)
    simulated[0]["fee_amount"] = 99
    result = compare_execution_costs({"replay_summary": {"markout_observations": simulated}}, fills, account_id="test")
    assert result["net_cost_paired_orders"] == 29
    assert result["net_cost_validation"] != "validated"


def test_capture_readiness_reports_next_session_without_claiming_completion():
    from ai_trading.tools.paper_evidence_review import capture_readiness
    now = datetime(2026, 9, 7, 12, tzinfo=UTC)
    rows = [{"timestamp": now.isoformat(), "account_id": "test", "trading_mode": "paper", "positions_complete": True}]
    report = capture_readiness(rows, account_id="test", now=now)
    assert report["status"] == "capture_active"
    assert report["next_full_session"] == "2026-09-08"
    assert report["opening_window_utc"] == ["2026-09-08T13:15:00+00:00", "2026-09-08T13:30:00+00:00"]
    assert not report["complete_session_observed"]
    assert capture_readiness(rows, account_id="different", now=now)["status"] == "capture_missing_or_stale"


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
