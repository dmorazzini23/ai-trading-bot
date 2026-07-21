from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import trading_day_report


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _parity_shadow_decision(
    *,
    bar_ts: str = "2026-07-17T14:30:00Z",
    symbol: str = "AAPL",
    side: str = "buy",
    qty: object = 5,
    submitted: bool = False,
    client_order_id: str = "shadow-aapl-1",
) -> dict[str, object]:
    return {
        "gates": ["REPLAY_LIVE_PARITY_GATE_FAILED"],
        "metrics": {"event": "replay_live_parity_controlled_skip"},
        "decision_journal": {
            "event": "replay_live_parity_controlled_skip",
            "bar_ts": bar_ts,
            "submitted": submitted,
            "reasons": ["REPLAY_LIVE_PARITY_GATE_FAILED"],
            "order_intent": {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "limit_price": 212.5,
                "client_order_id": client_order_id,
            },
        },
    }


def test_trading_day_report_attributes_rejections_and_symbol_pnl():
    report = trading_day_report.build_trading_day_report(
        report_date="2026-05-05",
        order_intents=[
            {"ts": "2026-05-05T14:00:00Z", "symbol": "AAPL", "status": "SUBMITTED"},
            {"ts": "2026-05-04T14:00:00Z", "symbol": "AMZN", "status": "SUBMITTED"},
        ],
        fills=[
            {
                "ts": "2026-05-05T14:01:00Z",
                "symbol": "AAPL",
                "realized_pnl": "3.25",
                "realized_net_edge_bps": "1.5",
                "expected_net_edge_bps": "2.0",
                "slippage_bps": "0.5",
            },
            {
                "ts": "2026-05-05T14:01:30Z",
                "symbol": "AAPL",
                "realized_pnl": "1.75",
                "realized_net_edge_bps": "2.5",
                "expected_net_edge_bps": "4.0",
                "slippage_bps": "1.5",
            },
            {
                "ts": "2026-05-05T14:02:00Z",
                "symbol": "AMZN",
                "pnl": "-1.0",
                "realized_net_edge_bps": "-3.0",
                "expected_net_edge_bps": "1.0",
                "slippage_bps": "3.0",
            },
        ],
        shadow_rows=[
            {
                "ts": "2026-05-05T14:00:00Z",
                "symbol": "MSFT",
                "challenger_would_trade": True,
                "champion_would_trade": False,
            }
        ],
        gate_rows=[
            {"ts": "2026-05-05T14:00:01Z", "symbol": "AAPL", "status": "blocked", "reason": "spread_cap"},
            {"ts": "2026-05-05T14:00:02Z", "symbol": "AMZN", "action": "reject", "gate": "quote_age"},
            {"ts": "2026-05-05T14:00:03Z", "symbol": "MSFT", "status": "blocked", "side": "sell_short", "reason": "long_only_config"},
        ],
        live_cost_model={"status": {"status": "ready"}},
        symbol_scorecard={"summary": {"allow": 2}, "symbols": []},
        regime_entry_throttle={"actions": {"reduce_size": 1}},
        expected_edge_calibration={
            "status": "overestimated",
            "execution_capture_diagnosis": {"attribution_counts": {"weak_execution_capture": 3}},
        },
        weekend_research={"status": "complete", "monday_preparation": {"mode": "paper"}},
        huggingface_discovery={"status": "discovered", "summary": {"candidate_count": 2}},
    )

    assert report["desired_trades"]["count"] == 1
    assert report["submitted_trades"]["count"] == 1
    assert report["rejected_trades"]["reasons"] == {
        "long_only_config": 1,
        "quote_age": 1,
        "spread_cap": 1,
    }
    assert report["symbol_contribution"] == {"AAPL": 5.0, "AMZN": -1.0}
    assert report["symbol_realized_edge_bps"] == {"AAPL": 2.0, "AMZN": -3.0}
    assert report["symbol_expected_edge_bps"] == {"AAPL": 3.0, "AMZN": 1.0}
    assert report["symbol_slippage_bps"] == {"AAPL": 1.0, "AMZN": 3.0}
    assert report["edge_quality"]["mean_realized_edge_bps"] == 1 / 3
    assert report["expected_edge_calibration"]["status"] == "overestimated"
    assert report["execution_capture_diagnosis"]["attribution_counts"] == {
        "weak_execution_capture": 3
    }
    assert report["regime_entry_throttle"] == {"actions": {"reduce_size": 1}}
    assert report["weekend_research"]["status"] == "complete"
    assert report["long_only_side_semantics"]["counts"]["sell_short_blocked"] == 1
    assert report["health_report_summary"]["desired"] == 1
    assert report["health_report_summary"]["rejected"] == 3
    assert report["health_report_summary"]["expected_edge_calibration_status"] == "overestimated"
    assert report["health_report_summary"]["huggingface_research_status"] == "discovered"
    assert report["huggingface_research"]["runtime_authority"] is False
    assert report["openclaw_summary"]["severity"] == "warning"
    assert report["symbol_trade_flow"]["AAPL"] == {
        "desired": 1,
        "submitted": 1,
        "rejected": 1,
        "fills": 2,
    }
    assert report["missed_opportunities"]["shadow_only_count"] == 1
    assert report["missed_opportunities"]["symbols"] == {"MSFT": 1}


def test_trading_day_report_preserves_blocked_registry_shadow_identity():
    challenger = {
        "model_id": "one-bar-shadow",
        "artifact_viability": {
            "ok": True,
            "path": "/models/one-bar-shadow.joblib",
        },
    }
    report = trading_day_report.build_trading_day_report(
        report_date="2026-07-18",
        order_intents=[],
        fills=[],
        shadow_rows=[],
        gate_rows=[],
        live_cost_model={},
        symbol_scorecard={},
        model_registry={
            "status": "blocked",
            "blocked_reasons": ["champion_artifact_missing"],
            "active_champion": None,
            "active_challenger": challenger,
            "promotion_authority": True,
        },
    )

    assert report["model_registry"]["status"] == "blocked"
    assert report["model_registry"]["active_challenger"] == challenger
    assert report["model_registry"]["promotion_authority"] is False
    assert report["model_registry"]["live_money_authority"] is False
    assert (
        report["health_report_summary"]["model_registry_active_challenger_id"]
        == "one-bar-shadow"
    )
    assert (
        report["health_report_summary"]["model_registry_active_challenger_path"]
        == "/models/one-bar-shadow.joblib"
    )


def test_trading_day_report_separates_metrics_control_skips_from_rejects():
    report = trading_day_report.build_trading_day_report(
        report_date="2026-06-11",
        order_intents=[
            {
                "ts": "2026-06-11T14:00:00Z",
                "symbol": "AAPL",
                "status": "REJECTED",
                "last_error": "pre_execution_order_checks_failed:metrics_improvement_control",
            }
        ],
        fills=[],
        shadow_rows=[],
        gate_rows=[
            {
                "ts": "2026-06-11T14:00:00Z",
                "symbol": "AAPL",
                "status": "skipped",
                "reason": "pre_execution_order_checks_failed",
                "detail": "metrics_improvement_control",
            },
            {
                "ts": "2026-06-11T14:01:00Z",
                "symbol": "MSFT",
                "status": "blocked",
                "reason": "quote_age",
            },
        ],
        live_cost_model={"status": {"status": "ready"}},
        symbol_scorecard={"summary": {"allow": 2}, "symbols": []},
    )

    assert report["rejected_trades"] == {
        "count": 1,
        "reasons": {"quote_age": 1},
    }
    assert report["controlled_skips"] == {
        "count": 2,
        "reasons": {"metrics_improvement_control": 2},
        "broker_rejects": False,
    }
    assert report["health_report_summary"]["rejected"] == 1
    assert report["health_report_summary"]["controlled_skips"] == 2
    assert report["openclaw_summary"]["severity"] == "warning"
    assert "controlled_skips=2" in report["openclaw_summary"]["summary"]


def test_trading_day_report_metrics_control_only_is_info_severity():
    report = trading_day_report.build_trading_day_report(
        report_date="2026-06-11",
        order_intents=[],
        fills=[],
        shadow_rows=[],
        gate_rows=[
            {
                "ts": "2026-06-11T14:00:00Z",
                "symbol": "AAPL",
                "status": "skipped",
                "reason": "pre_execution_order_checks_failed",
                "detail": "metrics_improvement_control",
            }
        ],
        live_cost_model={"status": {"status": "ready"}},
        symbol_scorecard={"summary": {"allow": 2}, "symbols": []},
    )

    assert report["rejected_trades"]["count"] == 0
    assert report["controlled_skips"]["count"] == 1
    assert report["openclaw_summary"]["severity"] == "info"
    assert report["openclaw_summary"]["suggested_action"] == (
        "review controlled skips and live-capital readiness before next session"
    )


def test_trading_day_report_excludes_shadow_outcomes_from_realized_fills():
    report = trading_day_report.build_trading_day_report(
        report_date="2026-07-21",
        order_intents=[],
        fills=[
            {
                "ts": "2026-07-21T15:00:00Z",
                "symbol": "AAPL",
                "status": "filled",
                "fill_based_evidence": True,
            },
            {
                "ts": "2026-07-21T15:00:00Z",
                "symbol": "AAPL",
                "evidence_type": "shadow_counterfactual",
                "evidence_partition": "shadow",
                "fill_based_evidence": False,
                "executed": False,
            },
        ],
        shadow_rows=[],
        gate_rows=[],
        live_cost_model={},
        symbol_scorecard={},
    )

    assert report["realized_fills"]["count"] == 1
    assert report["non_fill_evidence"]["excluded_from_realized_fills"] == 1
    assert report["symbol_trade_flow"]["AAPL"]["fills"] == 1


def test_trading_day_report_counts_only_canonical_parity_shadow_candidates():
    valid = _parity_shadow_decision()
    ordinary = _parity_shadow_decision(client_order_id="ordinary")
    ordinary["gates"] = ["OK_TRADE"]
    ordinary["metrics"] = {"event": "decision_record"}
    ordinary_journal = ordinary["decision_journal"]
    assert isinstance(ordinary_journal, dict)
    ordinary_journal["event"] = "decision_record"
    ordinary_journal["reasons"] = ["OK_TRADE"]
    hold = _parity_shadow_decision(client_order_id="hold")
    hold_journal = hold["decision_journal"]
    assert isinstance(hold_journal, dict)
    hold_journal["order_intent"] = None

    report = trading_day_report.build_trading_day_report(
        report_date="2026-07-17",
        order_intents=[],
        fills=[],
        shadow_rows=[],
        gate_rows=[],
        live_cost_model={"status": {"status": "ready"}},
        symbol_scorecard={"summary": {}, "symbols": []},
        decisions=[
            valid,
            _parity_shadow_decision(),
            ordinary,
            hold,
            _parity_shadow_decision(submitted=True, client_order_id="submitted"),
            _parity_shadow_decision(qty=0, client_order_id="zero"),
            _parity_shadow_decision(qty=-2, client_order_id="negative"),
            _parity_shadow_decision(side="hold", client_order_id="unsupported"),
            _parity_shadow_decision(
                bar_ts="2026-07-16T14:30:00Z",
                client_order_id="prior-day",
            ),
        ],
    )

    assert report["desired_trades"]["count"] == 1
    assert report["submitted_trades"]["count"] == 0
    assert report["rejected_trades"]["count"] == 0
    assert report["controlled_skips"] == {
        "count": 1,
        "reasons": {"replay_live_parity_gate": 1},
        "broker_rejects": False,
    }
    assert report["shadow_candidates"] == {
        "count": 1,
        "reasons": {"replay_live_parity_gate": 1},
        "symbols": {"AAPL": 1},
    }
    assert report["realized_fills"]["count"] == 0
    assert report["symbol_contribution"] == {}
    assert report["symbol_trade_flow"]["AAPL"] == {
        "desired": 1,
        "submitted": 0,
        "rejected": 0,
        "fills": 0,
    }
    assert report["health_report_summary"]["shadow_candidates"] == 1
    assert report["health_report_summary"]["controlled_skips"] == 1
    assert report["openclaw_summary"]["severity"] == "info"
    assert "controlled_skips=1" in report["openclaw_summary"]["summary"]
    assert "shadow_candidates=1" in report["openclaw_summary"]["summary"]
    assert "rejected=0" in report["openclaw_summary"]["summary"]


def test_parity_shadow_candidate_deduplicates_against_existing_intent():
    report = trading_day_report.build_trading_day_report(
        report_date="2026-07-17",
        order_intents=[
            {
                "ts": "2026-07-17T14:30:00Z",
                "client_order_id": "shadow-aapl-1",
                "symbol": "AAPL",
                "side": "buy",
                "qty": 5,
                "limit_price": 212.5,
                "status": "PENDING",
            }
        ],
        fills=[],
        shadow_rows=[],
        gate_rows=[],
        live_cost_model={"status": {"status": "ready"}},
        symbol_scorecard={"summary": {}, "symbols": []},
        decisions=[_parity_shadow_decision()],
    )

    assert report["desired_trades"]["count"] == 1
    assert report["shadow_candidates"]["count"] == 0
    assert report["symbol_trade_flow"]["AAPL"]["desired"] == 1


def test_date_match_supports_top_level_and_nested_bar_timestamps_safely():
    assert trading_day_report._date_match(
        {"bar_ts": "2026-07-17T14:30:00Z"},
        "2026-07-17",
    )
    assert trading_day_report._date_match(
        {"decision_journal": {"bar_ts": "2026-07-17T14:30:00Z"}},
        "2026-07-17",
    )
    assert not trading_day_report._date_match(
        {"decision_journal": "not-a-mapping"},
        "2026-07-17",
    )


def test_trading_day_report_cli_writes_latest_json_and_markdown(tmp_path: Path):
    intents = tmp_path / "intents.jsonl"
    fills = tmp_path / "fills.jsonl"
    gates = tmp_path / "gates.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    throttle = tmp_path / "throttle.json"
    calibration = tmp_path / "calibration.json"
    hf_discovery = tmp_path / "hf_discovery.json"
    weekend = tmp_path / "weekend.json"
    out = tmp_path / "trading_day.json"
    latest = tmp_path / "latest.json"
    md = tmp_path / "latest.md"
    _write_jsonl(intents, [{"ts": "2026-05-05T14:00:00Z", "status": "FILLED"}])
    _write_jsonl(fills, [{"ts": "2026-05-05T14:00:02Z", "symbol": "AAPL", "pnl": 2.0}])
    _write_jsonl(gates, [{"ts": "2026-05-05T14:00:01Z", "status": "blocked", "reason": "spread_cap"}])
    decision = _parity_shadow_decision(
        bar_ts="2026-05-05T14:00:03Z",
        client_order_id="shadow-cli-1",
    )
    _write_jsonl(decisions, [decision])
    throttle.write_text(
        json.dumps({"actions": {"block_new_entries": 1}, "evaluations": 1}),
        encoding="utf-8",
    )
    calibration.write_text(json.dumps({"status": "calibrated"}), encoding="utf-8")
    hf_discovery.write_text(json.dumps({"status": "discovered"}), encoding="utf-8")
    weekend.write_text(json.dumps({"status": "complete"}), encoding="utf-8")

    rc = trading_day_report.main(
        [
            "--report-date",
            "2026-05-05",
            "--order-intents-jsonl",
            str(intents),
            "--fills-jsonl",
            str(fills),
            "--gate-jsonl",
            str(gates),
            "--decisions-jsonl",
            str(decisions),
            "--regime-entry-throttle-json",
            str(throttle),
            "--expected-edge-calibration-json",
            str(calibration),
            "--huggingface-discovery-json",
            str(hf_discovery),
            "--weekend-research-json",
            str(weekend),
            "--output-json",
            str(out),
            "--latest-json",
            str(latest),
            "--latest-md",
            str(md),
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["realized_fills"]["count"] == 1
    assert payload["shadow_candidates"]["count"] == 1
    assert payload["regime_entry_throttle"]["actions"] == {"block_new_entries": 1}
    assert payload["expected_edge_calibration"]["status"] == "calibrated"
    assert payload["huggingface_research"]["discovery"]["status"] == "discovered"
    assert payload["weekend_research"]["status"] == "complete"
    assert payload["health_report_summary"]["fills"] == 1
    assert payload["openclaw_summary"]["service"] == "ai-trading-research"
    assert latest.is_file()
    markdown = md.read_text(encoding="utf-8")
    assert "Trading Day 2026-05-05" in markdown
    assert "Regime entry throttle" in markdown
    assert "Hugging Face research" in markdown
