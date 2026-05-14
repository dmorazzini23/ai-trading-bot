from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_trading.tools import runtime_performance_report as rpr


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[object]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row) if isinstance(row, dict) else str(row) for row in rows),
        encoding="utf-8",
    )
    return path


def test_missing_and_nondict_artifacts_return_deterministic_invalid_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rpr,
        "_summarize_broker_open_positions",
        lambda: rpr._broker_open_positions_unavailable("unit-test"),
    )
    missing_trade = tmp_path / "missing-trades.json"
    missing_gate = tmp_path / "missing-gate.json"
    nondict = _write_json(tmp_path / "nondict.json", ["not", "a", "mapping"])

    trade = rpr.summarize_trade_history(missing_trade)
    gate = rpr.summarize_gate_effectiveness(missing_gate)

    assert trade["exists"] is False
    assert trade["records"] == 0
    assert trade["broker_open_positions_error"] == "unit-test"
    assert gate == {
        "path": str(missing_gate),
        "exists": False,
        "gate_log_path": str(tmp_path / "gate_effectiveness.jsonl"),
        "gate_log_exists": False,
        "daily_gate_stats": [],
    }
    for summarizer in (
        rpr.summarize_edge_realism_state,
        rpr.summarize_policy_ablation_state,
        rpr.summarize_policy_runtime_toggles,
        rpr.summarize_uncertainty_capital_state,
        rpr.summarize_counterfactual_learning_state,
    ):
        summary = summarizer(nondict)
        assert summary["exists"] is True
        assert summary["valid"] is False


def test_fifo_reconstruction_aggregates_partial_lots_and_fill_source_rollups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rpr,
        "_summarize_broker_open_positions",
        lambda: {
            "broker_open_positions_available": True,
            "broker_open_position_count": 1,
            "broker_open_positions": {"AAPL": 2.0},
            "broker_open_position_snapshots": {},
            "broker_open_positions_error": None,
        },
    )
    monkeypatch.setattr(
        rpr,
        "get_env",
        lambda name, default=None, cast=None, resolve_aliases=True: {
            "AI_TRADING_RUNTIME_PERF_RECONCILIATION_FALLBACK_TO_BROKER_ENABLED": False,
            "AI_TRADING_RUNTIME_PERF_FEE_BPS_FALLBACK": 2.0,
        }.get(name, default),
    )
    rpr._runtime_fee_bps_fallback.cache_clear()
    trades_path = _write_jsonl(
        tmp_path / "fills.jsonl",
        [
            {
                "symbol": "AAPL",
                "side": "buy",
                "qty": 5,
                "fill_price": 100.0,
                "expected_price": 99.5,
                "ts": "2026-04-24T14:00:00Z",
                "source": "live",
                "strategy_id": "alpha",
                "venue": "arcx",
            },
            {
                "symbol": "AAPL",
                "side": "sell",
                "qty": 3,
                "fill_price": 103.0,
                "expected_price": 103.5,
                "ts": "2026-04-24T15:00:00Z",
                "source": "broker_reconcile",
                "fee_bps": 1.0,
            },
            {"symbol": "BAD", "side": "hold", "qty": 1, "price": 1.0},
        ],
    )

    summary = rpr.summarize_trade_history(trades_path, fill_events_path=trades_path)

    assert summary["pnl_source"] == "fifo_reconstructed_from_fills"
    assert summary["closed_trades"] == 1
    assert summary["pnl_sum"] == pytest.approx(5.9088086884)
    assert summary["total_fee_cost"] == pytest.approx(0.0909)
    assert summary["total_slippage_cost"] == pytest.approx(3.0002913116)
    assert summary["closed_trades_by_fill_source"] == {"reconcile_backfill": 1}
    assert summary["daily_trade_stats"] == [
        {
            "date": "2026-04-24",
            "trades": 1,
            "wins": 1,
            "losses": 0,
            "gross_win_pnl": pytest.approx(5.9088086884),
            "gross_loss_pnl": 0.0,
            "gross_pnl": pytest.approx(9.0),
            "net_pnl": pytest.approx(5.9088086884),
            "fee_cost": pytest.approx(0.0909),
            "slippage_cost": pytest.approx(3.0002913116),
            "win_rate": 1.0,
            "profit_factor": None,
        }
    ]
    assert summary["reconstructed_open_positions"] == {"AAPL": 2.0}
    assert summary["open_positions"] == {"AAPL": 2.0}
    assert summary["open_positions_basis"] == "trade_history"
    assert summary["open_position_reconciliation"]["symbol_mismatch_count"] == 0
    assert summary["slippage_root_cause_attribution"]["overall_slippage_drag_bps"] == pytest.approx(100.0097103877)
    rpr._runtime_fee_bps_fallback.cache_clear()


def test_runtime_report_exposes_same_day_fill_pnl_and_broker_position_basis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rpr,
        "_summarize_broker_open_positions",
        lambda: {
            "broker_open_positions_available": True,
            "broker_open_position_count": 0,
            "broker_open_positions": {},
            "broker_open_position_snapshots": {},
            "broker_open_positions_error": None,
        },
    )
    monkeypatch.setattr(
        rpr,
        "get_env",
        lambda name, default=None, cast=None, resolve_aliases=True: {
            "AI_TRADING_RUNTIME_PERF_RECONCILIATION_FALLBACK_TO_BROKER_ENABLED": True,
            "AI_TRADING_RUNTIME_PERF_RECONCILIATION_FALLBACK_TO_BROKER_ABS_DELTA_RATIO": 0.5,
            "AI_TRADING_RUNTIME_PERF_RECONCILIATION_FALLBACK_TO_BROKER_MISMATCH_COUNT": 1,
        }.get(name, default),
    )
    fills = _write_jsonl(
        tmp_path / "fills.jsonl",
        [
            {
                "symbol": "AMZN",
                "side": "buy",
                "qty": 1,
                "fill_price": 272.43,
                "ts": "2026-05-11T16:21:24Z",
                "source": "live",
            },
            {
                "symbol": "AMZN",
                "side": "buy",
                "qty": 1,
                "fill_price": 265.41,
                "ts": "2026-05-12T13:43:42Z",
                "source": "live",
            },
            {
                "symbol": "AMZN",
                "side": "sell",
                "qty": 1,
                "fill_price": 265.22,
                "ts": "2026-05-12T13:45:18Z",
                "source": "live",
            },
        ],
    )

    summary = rpr.summarize_trade_history(fills, fill_events_path=fills)

    assert summary["open_positions_basis"] == "broker_open_positions"
    assert summary["open_positions"] == {}
    same_day_rows = {
        row["date"]: row
        for row in summary["same_day_fill_pair_stats"]["daily_trade_stats"]
    }
    assert same_day_rows["2026-05-12"]["net_pnl"] == pytest.approx(-0.19)
    reconciliation = {
        row["date"]: row
        for row in summary["daily_pnl_reconciliation"]
    }
    assert reconciliation["2026-05-12"]["status"] == "mismatch"
    assert reconciliation["2026-05-12"]["accounting_net_pnl"] == pytest.approx(-7.21)
    assert reconciliation["2026-05-12"]["same_day_fill_gross_pnl"] == pytest.approx(-0.19)


def test_build_report_flattens_execution_fields_and_handles_invalid_optional_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_SYMBOLS", "")
    monkeypatch.setenv("AI_TRADING_CANARY_SYMBOLS", "")
    monkeypatch.setattr(
        rpr,
        "_summarize_broker_open_positions",
        lambda: rpr._broker_open_positions_unavailable("unit-test"),
    )
    monkeypatch.setattr(rpr, "summarize_oms_event_tca", lambda: {"enabled": False, "available": False})
    monkeypatch.setattr(rpr, "summarize_oms_invariants", lambda: {"enabled": False, "available": False})
    monkeypatch.setattr(rpr, "summarize_oms_lifecycle_parity", lambda: {"enabled": False, "available": False})
    monkeypatch.setattr(
        rpr,
        "summarize_replay_live_parity_gate",
        lambda *, oms_lifecycle_parity: {"enabled": False, "available": False},
    )
    trades = _write_json(
        tmp_path / "trades.json",
        [
            {
                "symbol": "MSFT",
                "side": "buy",
                "qty": 2,
                "entry_price": 50.0,
                "exit_price": 51.0,
                "timestamp": "2026-04-24T16:00:00Z",
                "pnl": 2.0,
            }
        ],
    )
    gate = _write_json(
        tmp_path / "gate.json",
        {
            "total_records": 4,
            "total_accepted_records": 2,
            "total_rejected_records": 2,
            "total_expected_net_edge_bps": 4.0,
        },
    )
    invalid_state = _write_json(tmp_path / "state.json", ["bad"])

    report = rpr.build_report(
        trade_history_path=trades,
        gate_summary_path=gate,
        edge_realism_state_path=invalid_state,
        policy_ablation_state_path=invalid_state,
        policy_runtime_toggles_path=invalid_state,
        uncertainty_capital_state_path=invalid_state,
        counterfactual_state_path=invalid_state,
    )

    assert report["report_schema_version"] == 3
    assert report["trade_history"]["pnl_available"] is True
    assert report["execution_vs_alpha"]["expected_edge_for_realism_bps"] == pytest.approx(2.0)
    assert report["expected_edge_for_realism_bps"] == pytest.approx(2.0)
    assert report["expected_edge_per_filled_trade_bps"] == pytest.approx(2.0)
    assert report["edge_realism"]["valid"] is False
    assert report["counterfactual_learning"]["valid"] is False


def test_evaluate_go_no_go_rolls_up_live_window_and_event_tca_parent_failures() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "daily_trade_stats_by_fill_source": {
                "live": [
                    {
                        "date": "2026-04-22",
                        "trades": 2,
                        "wins": 2,
                        "losses": 0,
                        "gross_win_pnl": 10.0,
                        "gross_loss_pnl": 0.0,
                        "net_pnl": 10.0,
                    },
                    {
                        "date": "2026-04-24",
                        "trades": 1,
                        "wins": 0,
                        "losses": 1,
                        "gross_win_pnl": 0.0,
                        "gross_loss_pnl": 8.0,
                        "net_pnl": -8.0,
                    },
                ],
                "reconcile_backfill": [
                    {
                        "date": "2026-04-24",
                        "trades": 99,
                        "wins": 99,
                        "losses": 0,
                        "gross_win_pnl": 99.0,
                        "gross_loss_pnl": 0.0,
                        "net_pnl": 99.0,
                    }
                ],
            },
            "open_position_reconciliation": {
                "available": True,
                "abs_delta_ratio": 0.0,
                "max_abs_delta_qty": 0.0,
                "symbol_mismatch_count": 0,
            },
            "slippage_drag_bps": 3.0,
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.5,
            "accepted_records": 2,
            "daily_gate_stats": [
                {
                    "date": "2026-04-22",
                    "total_records": 4,
                    "accepted_records": 2,
                    "total_expected_net_edge_bps": 10.0,
                },
                {
                    "date": "2026-04-24",
                    "total_records": 2,
                    "accepted_records": 1,
                    "total_expected_net_edge_bps": -2.0,
                },
            ],
        },
        "execution_vs_alpha": {
            "execution_capture_ratio": 0.5,
            "expected_edge_for_realism_bps": 4.0,
        },
        "oms_event_tca": {
            "enabled": True,
            "available": True,
            "filled_events": 1,
            "submit_reject_rate_pct": 8.0,
            "p90_slippage_bps": 30.0,
            "parent_execution_kpis_by_scope": [
                {
                    "scope": "AAPL",
                    "parent_orders": 2,
                    "retry_count": 8,
                    "failed_slices": 4,
                    "avg_success_ratio": 0.25,
                    "avg_arrival_slippage_bps": 40.0,
                }
            ],
        },
    }

    decision = rpr.evaluate_go_no_go(
        report,
        thresholds={
            "trade_fill_source": "live",
            "lookback_days": 2,
            "min_used_days": 2,
            "min_closed_trades": 2,
            "min_win_rate": 0.75,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.4,
            "min_expected_net_edge_bps": 0.0,
            "require_open_position_reconciliation": True,
            "require_oms_event_tca": True,
            "min_event_tca_filled_events": 2,
            "max_event_tca_submit_reject_rate_pct": 5.0,
            "max_event_tca_p90_slippage_bps": 20.0,
            "min_event_tca_parent_summary_events": 1,
            "max_event_tca_parent_retry_per_order": 2.0,
            "max_event_tca_parent_failed_slices_per_order": 1.0,
            "min_event_tca_parent_avg_success_ratio": 0.7,
            "max_event_tca_parent_avg_arrival_slippage_bps": 25.0,
        },
    )

    assert decision["gate_passed"] is False
    assert decision["observed"]["trade_metric_scope"]["mode"] == "rolling_days"
    assert decision["observed"]["closed_trades"] == 3
    assert decision["observed"]["net_pnl"] == pytest.approx(2.0)
    assert decision["observed"]["gate_metric_scope"]["window_accepted_records"] == 3
    assert decision["checks"]["oms_event_tca_consistent"] is False
    assert decision["checks"]["oms_event_tca_parent_execution_consistent"] is False
    assert "win_rate" in decision["failed_checks"]
    assert "oms_event_tca_consistent" in decision["failed_checks"]
    assert decision["observed"]["event_tca_parent_retry_per_order"] == pytest.approx(4.0)
    assert decision["observed"]["event_tca_parent_scope_threshold_breach_count"] == 1


def test_main_json_output_and_fail_on_no_go_uses_cli_threshold_overrides(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(rpr, "ensure_dotenv_loaded", lambda: None)
    monkeypatch.setattr(
        rpr,
        "resolve_runtime_report_paths",
        lambda **kwargs: {
            "trade_history": tmp_path / "trades.json",
            "gate_summary": tmp_path / "gate.json",
            "gate_log": None,
            "tca": None,
            "fill_events": None,
            "edge_realism_state": None,
            "policy_ablation_state": None,
            "policy_runtime_toggles": None,
            "uncertainty_capital_state": None,
            "counterfactual_state": None,
        },
    )

    def fake_build_report(**kwargs: object) -> dict[str, object]:
        captured["build_kwargs"] = kwargs
        return {"trade_history": {"path": "x"}, "gate_effectiveness": {"path": "y"}}

    def fake_thresholds() -> dict[str, object]:
        return {
            "min_closed_trades": 10,
            "require_pnl_available": True,
            "require_gate_valid": False,
            "require_open_position_reconciliation": False,
        }

    def fake_evaluate(report: dict[str, object], *, thresholds: dict[str, object]) -> dict[str, object]:
        captured["thresholds"] = dict(thresholds)
        assert report["trade_history"] == {"path": "x"}
        return {"gate_passed": False, "failed_checks": ["closed_trades"]}

    monkeypatch.setattr(rpr, "build_report", fake_build_report)
    monkeypatch.setattr(rpr, "resolve_runtime_gonogo_thresholds", fake_thresholds)
    monkeypatch.setattr(rpr, "evaluate_go_no_go", fake_evaluate)

    code = rpr.main(
        [
            "--json",
            "--fail-on-no-go",
            "--allow-missing-pnl",
            "--require-gate-valid",
            "--require-open-position-reconciliation",
            "--trade-fill-source",
            "broker_reconcile",
            "--min-closed-trades",
            "3",
            "--lookback-days",
            "2",
            "--max-slippage-drag-bps",
            "7.5",
        ]
    )

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["go_no_go"] == {
        "gate_passed": False,
        "failed_checks": ["closed_trades"],
    }
    assert captured["thresholds"] == {
        "min_closed_trades": 3,
        "require_pnl_available": False,
        "require_gate_valid": True,
        "require_open_position_reconciliation": True,
        "trade_fill_source": "broker_reconcile",
        "lookback_days": 2,
        "max_slippage_drag_bps": 7.5,
    }
    assert captured["build_kwargs"]["trade_history_path"] == tmp_path / "trades.json"
