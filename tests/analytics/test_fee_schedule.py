"""Declared fee-schedule provenance: parser, labeling, default-off and consumers.

Rates below are synthetic test fixtures, not Alpaca figures.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_trading.analytics import fee_schedule as fs
from ai_trading.analytics.tca import reconcile_pending_tca_with_fill

REPO_ROOT = Path(__file__).resolve().parents[2]
FILL_TS = datetime(2026, 10, 19, 15, 0, tzinfo=UTC)


def _schedule_payload() -> dict:
    return {
        "schema_version": 1,
        "schedule_id": "test_schedule_v1",
        "evidence_class": fs.FEE_EVIDENCE_CLASS_DECLARED,
        "promotion_authority": False,
        "currency": "USD",
        "provenance": {
            "source_url": "https://example.test/fees",
            "retrieved_at": "2026-10-18T12:00:00+00:00",
            "declared_by": "test-owner",
            "declared_at": "2026-10-18T12:05:00+00:00",
        },
        "effective_from": "2026-10-19",
        "effective_to": None,
        "rounding": {"mode": "ceil", "scope": "per_fill", "increment_usd": "0.01"},
        "venues": {
            "DEFAULT": {
                "components": [
                    {"name": "commission", "category": "commission", "basis": "per_share", "rate": "0.005", "sides": ["buy", "sell"]},
                    {"name": "sec_section_31", "category": "regulatory", "basis": "per_notional", "rate": "0.00001", "sides": ["sell"]},
                    {"name": "finra_taf", "category": "regulatory", "basis": "per_share", "rate": "0.001", "max_usd_per_fill": "0.5", "sides": ["sell"]},
                    {"name": "cat", "category": "regulatory", "basis": "per_share", "rate": "0.0001", "sides": ["buy", "sell"]},
                ]
            }
        },
    }


def _write_schedule(tmp_path: Path, payload: dict | None = None) -> Path:
    path = tmp_path / "fee_schedule.json"
    path.write_text(json.dumps(payload or _schedule_payload()), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    fs._SCHEDULE_CACHE.clear()
    fs._LOGGED_FAILURES.clear()
    monkeypatch.delenv(fs.FEE_SCHEDULE_SOURCE_ENV, raising=False)
    monkeypatch.delenv(fs.FEE_SCHEDULE_PATH_ENV, raising=False)
    yield
    fs._SCHEDULE_CACHE.clear()
    fs._LOGGED_FAILURES.clear()


def _enable(monkeypatch, path: Path) -> None:
    monkeypatch.setenv(fs.FEE_SCHEDULE_SOURCE_ENV, "declared")
    monkeypatch.setenv(fs.FEE_SCHEDULE_PATH_ENV, str(path))


# --- parser -----------------------------------------------------------------


def test_parser_loads_schedule_hashes_bytes_and_applies_side_rules(tmp_path):
    path = _write_schedule(tmp_path)
    schedule = fs.load_fee_schedule(path)
    assert schedule.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert schedule.provenance["declared_by"] == "test-owner"

    sell, status = fs.compute_declared_fee(schedule, side="sell", qty=10, price=100.0, fill_ts=FILL_TS)
    assert status == "applied" and sell is not None
    # 0.05 commission + 0.01 SEC + 0.01 TAF + 0.001 CAT = 0.071 -> ceil 0.08
    assert sell.fee_amount == pytest.approx(0.08)
    assert sell.fee_bps == pytest.approx(0.8)
    assert set(sell.components) == {"commission", "sec_section_31", "finra_taf", "cat"}

    buy, _ = fs.compute_declared_fee(schedule, side="buy", qty=10, price=100.0, fill_ts=FILL_TS)
    assert buy is not None
    assert set(buy.components) == {"commission", "cat"}  # TAF/SEC are sells-only
    assert buy.fee_amount == pytest.approx(0.06)

    short, _ = fs.compute_declared_fee(schedule, side="sell_short", qty=1000, price=10.0, fill_ts=FILL_TS)
    assert short is not None and short.components["finra_taf"] == pytest.approx(0.5)  # capped


def test_parser_rejects_unfilled_example_file():
    with pytest.raises(fs.FeeScheduleError, match="placeholder"):
        fs.load_fee_schedule(REPO_ROOT / "config" / "fee_schedule.paper_alpaca.example.json")


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda p: p.pop("provenance"), "provenance block is mandatory"),
        (lambda p: p["provenance"].pop("declared_by"), "declared_by"),
        (lambda p: p["provenance"].update(source_url="file:///tmp/x"), "source_url"),
        (lambda p: p["provenance"].update(retrieved_at="2026-10-18T12:00:00"), "timezone"),
        (lambda p: p.update(promotion_authority=True), "promotion_authority"),
        (lambda p: p.update(effective_to="2026-10-01"), "effective_to precedes"),
        (lambda p: p["venues"]["DEFAULT"]["components"][2].update(sides=["buy", "sell"]), "finra_taf must apply"),
        (lambda p: p["venues"]["DEFAULT"]["components"][3].update(sides=["sell"]), "cat must apply"),
        (lambda p: p["venues"]["DEFAULT"]["components"].pop(0), "commission component is mandatory"),
        (lambda p: p["venues"]["DEFAULT"]["components"][1].update(rate="-1"), "non-negative"),
        (lambda p: p["venues"]["DEFAULT"]["components"][1].update(rate="<FILL>"), "placeholder"),
        (lambda p: p["rounding"].update(mode="banker"), "rounding.mode"),
        (lambda p: p.update(currency="EUR"), "currency"),
    ],
)
def test_parser_rejects_invalid_schedules(mutate, match):
    payload = copy.deepcopy(_schedule_payload())
    mutate(payload)
    with pytest.raises(fs.FeeScheduleError, match=match):
        fs.parse_fee_schedule(payload, sha256="x")


def test_effective_window_and_venue_fallback(tmp_path):
    payload = _schedule_payload()
    payload["venues"] = {"IEX": payload["venues"]["DEFAULT"]}
    schedule = fs.parse_fee_schedule(payload, sha256="x")
    before = datetime(2026, 10, 16, 15, 0, tzinfo=UTC)
    assert fs.compute_declared_fee(schedule, side="buy", qty=1, price=10.0, fill_ts=before) == (None, "outside_effective_window")
    assert fs.compute_declared_fee(schedule, side="buy", qty=1, price=10.0, fill_ts=FILL_TS, venue="NYSE") == (None, "venue_not_declared")
    fee, status = fs.compute_declared_fee(schedule, side="buy", qty=1, price=10.0, fill_ts=FILL_TS, venue="iex")
    assert status == "applied" and fee is not None


# --- config surface and default-off ------------------------------------------


def test_disabled_by_default_returns_no_fields(tmp_path):
    _write_schedule(tmp_path)
    assert fs.declared_fee_fields(side="buy", qty=1, price=10.0, fill_ts=FILL_TS) == {}


def test_enabled_without_valid_schedule_never_imputes_a_fee(tmp_path, monkeypatch):
    monkeypatch.setenv(fs.FEE_SCHEDULE_SOURCE_ENV, "declared")
    assert fs.declared_fee_fields(side="buy", qty=1, price=10.0, fill_ts=FILL_TS) == {"fee_schedule_status": "schedule_path_unset"}
    monkeypatch.setenv(fs.FEE_SCHEDULE_PATH_ENV, str(REPO_ROOT / "config" / "fee_schedule.paper_alpaca.example.json"))
    fields = fs.declared_fee_fields(side="buy", qty=1, price=10.0, fill_ts=FILL_TS)
    assert "fee_amount" not in fields and fields["fee_schedule_status"].startswith("schedule_invalid:")
    monkeypatch.setenv(fs.FEE_SCHEDULE_SOURCE_ENV, "broker")
    assert fs.declared_fee_fields(side="buy", qty=1, price=10.0, fill_ts=FILL_TS)["fee_schedule_status"] == "invalid_source:broker"


def test_enabled_fields_carry_provenance_and_hash(tmp_path, monkeypatch):
    path = _write_schedule(tmp_path)
    _enable(monkeypatch, path)
    fields = fs.declared_fee_fields(side="sell", qty=10, price=100.0, fill_ts=FILL_TS)
    assert fields["fee_source"] == "declared_schedule"
    assert fields["fee_basis"] == fs.FEE_BASIS_DECLARED_SCHEDULE
    assert fields["fee_currency"] == "USD"
    assert fields["fee_amount"] == pytest.approx(0.08)
    assert fields["fee_bps"] == pytest.approx(0.8)
    assert fields["fee_schedule_hash"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert fields["fee_schedule_id"] == "test_schedule_v1"
    assert fs.fee_provenance_class(fields) == fs.FEE_PROVENANCE_DECLARED
    assert not fs.is_broker_verified_fee(fields)


@pytest.mark.parametrize(
    "row,expected",
    [
        ({"fee_source": "broker_activity", "fee_basis": "per_fill_total", "fee_currency": "USD", "fee_amount": 0.01}, fs.FEE_PROVENANCE_BROKER),
        ({"fee_source": "declared_schedule", "fee_basis": fs.FEE_BASIS_DECLARED_SCHEDULE, "fee_currency": "USD", "fee_amount": 0.01, "fee_schedule_hash": "h"}, fs.FEE_PROVENANCE_DECLARED),
        ({"fee_source": "declared_schedule", "fee_basis": "per_fill_total", "fee_currency": "USD", "fee_amount": 0.01}, fs.FEE_PROVENANCE_UNVERIFIED),
        ({"fee_source": "configured_estimate", "fee_amount": 0.01}, fs.FEE_PROVENANCE_ESTIMATE),
        ({"fee_source": "missing", "fee_amount": None}, fs.FEE_PROVENANCE_MISSING),
        ({}, fs.FEE_PROVENANCE_MISSING),
    ],
)
def test_fee_provenance_classes_are_distinct(row, expected):
    assert fs.fee_provenance_class(row) == expected


# --- fill record writer -------------------------------------------------------


def _persist_fill(monkeypatch, *, runtime_payload=None, side="sell"):
    from ai_trading.execution import live_trading

    engine = live_trading.ExecutionEngine.__new__(live_trading.ExecutionEngine)
    engine._cycle_account = {"id": "paper-account"}
    engine.execution_mode = "paper"
    captured: dict[str, object] = {}
    monkeypatch.setattr(engine, "_runtime_exec_event_persistence_enabled", lambda: True)
    monkeypatch.setattr(live_trading, "record_trade_fill", lambda _payload: None)
    monkeypatch.setattr(engine, "_record_runtime_fill_event", lambda payload: captured.update(payload=dict(payload)))
    monkeypatch.setattr(engine, "_update_symbol_loss_cooldown_from_fill", lambda **_kwargs: None)
    monkeypatch.setattr(engine, "_arm_symbol_reentry_cooldown_from_fill", lambda **_kwargs: None)
    monkeypatch.setattr(engine, "_reconcile_pending_tca_from_fill", lambda **kwargs: captured.update(tca=dict(kwargs)))
    monkeypatch.setattr(live_trading, "_config_float", lambda _name, default: default)
    engine._persist_fill_derived_trade_record(
        symbol="AAPL",
        side=side,
        filled_qty=10.0,
        fill_price=100.0,
        expected_price=100.0,
        order_id="oid-1",
        client_order_id="cid-1",
        order_status="filled",
        signal=None,
        timestamp=FILL_TS,
        runtime_payload=runtime_payload if runtime_payload is not None else {"source": "live"},
        closing_position=False,
    )
    return captured


def test_fill_record_unchanged_when_disabled(monkeypatch):
    captured = _persist_fill(monkeypatch)
    payload = captured["payload"]
    assert payload["fee_source"] == "missing"
    assert payload["fee_amount"] is None and payload["fee_bps"] is None
    assert payload["fee_currency"] is None and payload["fee_basis"] is None
    assert not any(key.startswith("fee_schedule") for key in payload)
    assert captured["tca"]["fee_provenance"] is None


def test_fill_record_labeled_declared_schedule_when_enabled(tmp_path, monkeypatch):
    path = _write_schedule(tmp_path)
    _enable(monkeypatch, path)
    captured = _persist_fill(monkeypatch)
    payload = captured["payload"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert payload["fee_source"] == "declared_schedule"
    assert payload["fee_amount"] == pytest.approx(0.08)
    assert payload["fee_basis"] == fs.FEE_BASIS_DECLARED_SCHEDULE
    assert payload["fee_schedule_hash"] == digest
    assert captured["tca"]["fee_amount"] == pytest.approx(0.08)
    assert captured["tca"]["fee_provenance"]["fee_source"] == "declared_schedule"
    assert captured["tca"]["fee_provenance"]["fee_schedule_hash"] == digest


def test_broker_fee_takes_precedence_over_declared_schedule(tmp_path, monkeypatch):
    _enable(monkeypatch, _write_schedule(tmp_path))
    captured = _persist_fill(
        monkeypatch,
        runtime_payload={"source": "live", "fee_amount": 0.03, "fee_currency": "USD", "fee_basis": "per_fill_total"},
    )
    payload = captured["payload"]
    assert payload["fee_source"] == "broker_payload"
    assert payload["fee_amount"] == 0.03
    assert "fee_schedule_hash" not in payload


# --- TCA and consumers ------------------------------------------------------


def test_tca_resolution_carries_declared_label(tmp_path):
    path = tmp_path / "tca.jsonl"
    pending = {"ts": "2026-10-19T14:59:00+00:00", "symbol": "AAPL", "side": "sell", "qty": 10,
               "order_id": "oid-1", "client_order_id": "cid-1", "pending_event": True, "decision_price": 100.0}
    path.write_text(json.dumps(pending) + "\n", encoding="utf-8")
    provenance = {"fee_source": "declared_schedule", "fee_basis": fs.FEE_BASIS_DECLARED_SCHEDULE,
                  "fee_currency": "USD", "fee_schedule_hash": "h", "fee_schedule_id": "test_schedule_v1"}
    ok, reason = reconcile_pending_tca_with_fill(
        str(path), client_order_id="cid-1", order_id="oid-1", symbol="AAPL", side="sell",
        fill_price=100.0, fill_qty=10, status="filled", fill_ts=FILL_TS, fee_amount=0.08,
        fee_provenance=provenance,
    )
    assert (ok, reason) == (True, "reconciled")
    resolved = json.loads(path.read_text().splitlines()[-1])
    assert resolved["fees"] == pytest.approx(0.08)
    assert resolved["fee_source"] == "declared_schedule"
    assert resolved["fee_schedule_hash"] == "h"
    assert resolved["total_fee_verified"] is False


def _declared_row(**extra):
    return {"fee_amount": 0.01, "fee_source": "declared_schedule", "fee_basis": fs.FEE_BASIS_DECLARED_SCHEDULE,
            "fee_currency": "USD", "fee_schedule_hash": "h", **extra}


def test_runtime_performance_report_labels_declared_schedule(tmp_path):
    from ai_trading.tools import runtime_performance_report as rpt

    trade_path = tmp_path / "trade_history.json"
    gate_path = tmp_path / "gate.json"
    tca_path = tmp_path / "tca.jsonl"
    trade_path.write_text(json.dumps([{
        "symbol": "AAPL", "side": "buy", "qty": 10, "entry_price": 100.0, "exit_price": 101.0,
        "entry_time": "2026-02-01T14:30:00+00:00", "exit_time": "2026-02-01T15:00:00+00:00", "reward": 10.0,
    }]), encoding="utf-8")
    gate_path.write_text("{}", encoding="utf-8")
    tca_path.write_text("\n".join(json.dumps(_declared_row(symbol="AAPL", side=side, qty=10, fill_price=price, ts=ts))
                                  for side, price, ts in (("buy", 100.0, "2026-02-01T14:30:00+00:00"),
                                                          ("sell", 101.0, "2026-02-01T15:00:00+00:00"))) + "\n",
                        encoding="utf-8")
    trade = rpt.build_report(trade_history_path=trade_path, gate_summary_path=gate_path, tca_path=tca_path)["trade_history"]
    assert trade["pnl_available"] is True
    assert trade["pnl_basis"] == "declared_fee_schedule_not_broker_verified"
    assert trade["cost_attribution"]["fee_evidence"] == {"declared_schedule": 1}
    assert trade["cost_attribution"]["declared_schedule_fee_trades"] == 1
    assert trade["cost_attribution"]["declared_schedule_note"]
    assert trade["total_fee_cost"] == pytest.approx(0.02)


def test_research_reset_chain_never_counts_declared_as_broker_evidence():
    from ai_trading.tools import research_reset as reset

    decision = {"symbol": "SPY", "correlation_id": "d", "metrics": {"decision_ts": "2026-09-08T14:00:00Z", "decision_ts_basis": "explicit", "source_timestamp": "2026-09-08T13:59:00Z", "quote_timestamp": "2026-09-08T13:59:59.500Z", "raw_quote_bid": 100, "raw_quote_ask": 100.01}}
    order = {"symbol": "SPY", "side": "buy", "correlation_id": "d", "order_id": "o", "account_id": "paper", "trading_mode": "paper", "ts": "2026-09-08T14:00:00.1Z"}
    fill = _declared_row(symbol="SPY", side="buy", correlation_id="d", order_id="o", fill_id="f", fill_qty=1, fill_price=100.01, account_id="paper", trading_mode="paper", ts="2026-09-08T14:00:00.2Z")
    result = reset.execution_chain([decision], [order], [fill])
    assert result["complete_chains"] == 0
    assert result["gap_counts"] == {"verified_total_fees_missing": 1}
    assert result["fee_provenance_counts"] == {"declared_schedule": 1}


def test_broker_accounting_keeps_declared_distinct_from_explicit_total():
    from ai_trading.tools.broker_accounting_evidence import reconcile

    snapshot = {"account_id": "paper", "trading_mode": "paper", "pagination_complete": True, "after": "2026-09-01",
                "activities": [{"id": "a", "activity_type": "FILL", "order_id": "o", "qty": "2"}]}
    fills = [_declared_row(account_id="paper", trading_mode="paper", fill_id="f", order_id="o", fill_qty=2, ts="2026-09-02T15:00:00Z")]
    result = reconcile(snapshot, fills)
    assert result["fee_coverage"] == {"declared_schedule_fee_broker_total_unknown": 1}
    assert result["net_fee_validation"] != "verified"


def test_execution_reconciliation_labels_declared_but_keeps_fee_missing():
    from ai_trading.tools.execution_evidence_reconciliation import reconcile_evidence

    buy = _declared_row(fill_id="buy", order_id="buy", symbol="AAPL", side="buy", fill_qty=1, fill_price=100, ts="2026-01-02T15:00:00Z")
    sell = _declared_row(fill_id="sell", order_id="sell", symbol="AAPL", side="sell", fill_qty=1, fill_price=101, ts="2026-01-02T16:00:00Z")
    report = reconcile_evidence(decisions=[buy, sell], orders=[buy, sell], tca=[buy, sell], fills=[buy, sell])
    assert report["counts"]["fees_declared_schedule"] == 2
    assert report["counts"]["fees_missing"] == 2
    assert report["entry_exit_links"][0]["net_pnl"] is None


def test_paper_evidence_review_reports_declared_net_cost_without_validating_contract():
    from ai_trading.tools.paper_evidence_review import compare_execution_costs

    simulated, fills = [], []
    for i in range(30):
        row = {"client_order_id": str(i), "symbol": "AAPL", "side": "sell", "fill_qty": 2, "fill_price": 99, "reference_price": 100,
               "fee_amount": .0198, "fee_currency": "USD", "fee_basis": "per_fill_total", "fee_source": "simulated_notional_bps_v1", "fee_rate_bps": 1}
        simulated.append(row)
        fills.append({**row, **_declared_row(fee_amount=.02), "fill_id": str(i), "fill_price": 98, "expected_price": 100,
                      "ts": f"2026-08-{3 + i % 5:02d}T15:00:00Z", "trading_mode": "paper", "account_id": "test"})
    result = compare_execution_costs({"replay_summary": {"markout_observations": simulated}}, fills, account_id="test")
    assert result["net_cost_validation"] != "validated"
    assert result["net_cost_paired_orders"] == 0
    assert result["declared_schedule_paired_orders"] == 30
    assert result["observed_fee_provenance_counts"] == {"declared_schedule": 30}
    pair = result["pairs"][0]
    assert pair["observed_fee_bps"] is None
    assert pair["declared_schedule_observed_fee_bps"] == pytest.approx(.02 / (98 * 2) * 10000)
