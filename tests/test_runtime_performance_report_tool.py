from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_trading.tools import runtime_performance_report as rpt


def test_default_trade_history_path_uses_runtime_trade_history() -> None:
    assert rpt._DEFAULT_TRADE_HISTORY_PATH == "runtime/trade_history.parquet"


def test_build_report_summarizes_trade_and_gate_data(tmp_path: Path) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"

    trade_history_path.write_text(
        json.dumps(
            [
                {"symbol": "AAPL", "side": "buy", "pnl": 10},
                {"symbol": "MSFT", "side": "sell", "pnl": -4},
                {"symbol": "NVDA", "side": "buy", "pnl": 6},
            ]
        ),
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps(
            {
                "total_records": 20,
                "total_accepted_records": 5,
                "total_rejected_records": 15,
                "total_expected_net_edge_bps": -12.5,
                "gate_totals": {"COST_GATE": 11, "STOP_LOCK": 3, "OK_TRADE": 5},
            }
        ),
        encoding="utf-8",
    )

    report = rpt.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
    )

    trade = report["trade_history"]
    gate = report["gate_effectiveness"]
    assert trade["pnl_available"] is True
    assert trade["pnl_sum"] == 12
    assert trade["win_rate"] == 2 / 3
    assert gate["acceptance_rate"] == 0.25
    assert gate["top_gates"][0]["gate"] == "COST_GATE"
    assert gate["top_rejection_reasons_by_count"] == gate["top_gates"]
    assert gate["top_rejection_concentration_gate"] == "COST_GATE"
    assert gate["top_rejection_concentration_ratio"] == pytest.approx(11 / 15)


def test_build_report_treats_reward_as_realized_pnl(tmp_path: Path) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"
    trade_history_path.write_text(
        json.dumps(
            [
                {"symbol": "AAPL", "side": "buy", "reward": 2.5},
                {"symbol": "MSFT", "side": "sell", "reward": -1.0},
            ]
        ),
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps({"total_records": 0, "total_accepted_records": 0, "total_rejected_records": 0}),
        encoding="utf-8",
    )

    report = rpt.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
    )

    trade = report["trade_history"]
    assert trade["pnl_available"] is True
    assert trade["pnl_sum"] == pytest.approx(1.5)
    assert trade["closed_trades"] == 2


def test_build_report_uses_blocked_gate_counts_for_rejection_concentration(
    tmp_path: Path,
) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"
    trade_history_path.write_text("[]", encoding="utf-8")
    gate_summary_path.write_text(
        json.dumps(
            {
                "total_records": 100,
                "total_accepted_records": 30,
                "total_rejected_records": 70,
                "gate_totals": {
                    "SAFETY_TIER_ATTACK_SCALE": 60,
                    "RISK_PORTFOLIO_HARD_BLOCK": 8,
                    "OK_TRADE": 30,
                },
                "gate_attribution": {
                    "SAFETY_TIER_ATTACK_SCALE": {
                        "count": 60,
                        "accepted_records": 30,
                        "blocked_records": 2,
                    },
                    "RISK_PORTFOLIO_HARD_BLOCK": {
                        "count": 8,
                        "accepted_records": 0,
                        "blocked_records": 8,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = rpt.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
    )

    gate = report["gate_effectiveness"]
    assert gate["top_gates"][0]["gate"] == "SAFETY_TIER_ATTACK_SCALE"
    assert gate["top_rejection_reasons_by_count"][0]["gate"] == "RISK_PORTFOLIO_HARD_BLOCK"
    assert gate["top_rejection_concentration_gate"] == "RISK_PORTFOLIO_HARD_BLOCK"
    assert gate["top_rejection_concentration_ratio"] == pytest.approx(8 / 70)


def test_build_report_includes_execution_vs_alpha_attribution(tmp_path: Path) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"
    trade_history_path.write_text(
        json.dumps(
            [
                {
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": 5,
                    "entry_price": 100.0,
                    "entry_time": "2026-03-20T14:30:00+00:00",
                    "fee_bps": 0.0,
                    "slippage_bps": 4.0,
                },
                {
                    "symbol": "AAPL",
                    "side": "sell",
                    "qty": 5,
                    "entry_price": 103.0,
                    "entry_time": "2026-03-20T15:30:00+00:00",
                    "fee_bps": 0.0,
                    "slippage_bps": 2.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps(
            {
                "total_records": 100,
                "total_accepted_records": 20,
                "total_rejected_records": 80,
                "total_expected_net_edge_bps": 200.0,
            }
        ),
        encoding="utf-8",
    )

    report = rpt.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
    )

    attribution = report["execution_vs_alpha"]
    assert attribution["available"] is True
    assert attribution["realized_net_edge_bps"] is not None
    assert attribution["slippage_drag_bps"] is not None
    assert attribution["expected_edge_per_accept_bps"] == pytest.approx(10.0)
    assert attribution["expected_edge_per_accept_bps_raw"] == pytest.approx(10.0)
    assert attribution["edge_realism_gap_ratio"] is not None
    assert isinstance(attribution["daily"], list)


def test_summarize_execution_vs_alpha_clips_outlier_expected_edge_and_uses_notional_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_RUNTIME_PERF_EXPECTED_EDGE_ABS_CAP_BPS", "20.0")
    monkeypatch.setenv("AI_TRADING_RUNTIME_PERF_EXPECTED_EDGE_CLIP_MIN_SAMPLES", "99")

    attribution = rpt.summarize_execution_vs_alpha(
        trade_summary={
            "total_entry_notional": 200_000.0,
            "pnl_sum": 200.0,
            "slippage_drag_bps": 4.0,
            "daily_expectancy": [
                {
                    "date": "2026-04-01",
                    "trades": 12,
                    "entry_notional": 150_000.0,
                    "net_edge_bps": 8.0,
                },
                {
                    "date": "2026-04-02",
                    "trades": 3,
                    "entry_notional": 50_000.0,
                    "net_edge_bps": 16.0,
                },
            ],
        },
        gate_summary={
            "total_expected_net_edge_bps": 1_000.0,
            "accepted_records": 20,
            "daily_gate_stats": [
                {
                    "date": "2026-04-01",
                    "accepted_records": 10,
                    "total_expected_net_edge_bps": 30.0,
                },
                {
                    "date": "2026-04-02",
                    "accepted_records": 10,
                    "total_expected_net_edge_bps": 970.0,
                },
            ],
        },
    )

    assert attribution["realized_net_edge_bps"] == pytest.approx(10.0)
    assert attribution["expected_edge_per_accept_bps_raw"] == pytest.approx(50.0)
    assert attribution["expected_edge_per_accept_bps"] == pytest.approx(11.5)
    assert attribution["expected_edge_per_traded_notional_bps"] == pytest.approx(7.25)
    assert attribution["expected_edge_for_realism_bps"] == pytest.approx(7.25)
    assert attribution["edge_realism_gap_ratio"] == pytest.approx(10.0 / 7.25)
    clip = attribution["expected_edge_clip"]
    assert clip["applied"] is True
    assert clip["changed_samples"] == 1
    assert clip["abs_cap_bps"] == pytest.approx(20.0)
    assert attribution["daily"][1]["expected_edge_per_accept_bps_raw"] == pytest.approx(97.0)
    assert attribution["daily"][1]["expected_edge_per_accept_bps"] == pytest.approx(20.0)


def test_build_report_includes_edge_realism_snapshot(tmp_path: Path) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"
    edge_realism_state_path = tmp_path / "edge_realism_state.json"
    trade_history_path.write_text(
        json.dumps(
            [
                {"symbol": "AAPL", "side": "buy", "pnl": 5.0},
                {"symbol": "AAPL", "side": "sell", "pnl": -2.0},
            ]
        ),
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps({"total_records": 1, "total_accepted_records": 1, "total_rejected_records": 0}),
        encoding="utf-8",
    )
    edge_realism_state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "updated_at": "2026-04-01T00:00:00+00:00",
                "global": {
                    "samples": 12,
                    "mean_realized_to_expected_ratio": 0.71,
                    "mean_realized_net_edge_bps": 1.4,
                    "mean_expected_net_edge_bps": 2.0,
                },
                "buckets": {
                    "aapl:buy:midday:11:trend:normal": {
                        "samples": 6,
                        "mean_realized_to_expected_ratio": 0.66,
                        "mean_realized_net_edge_bps": 1.2,
                        "mean_expected_net_edge_bps": 1.8,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = rpt.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
        edge_realism_state_path=edge_realism_state_path,
    )

    edge_realism = report["edge_realism"]
    assert edge_realism["valid"] is True
    assert edge_realism["global_samples"] == 12
    assert edge_realism["global_mean_realized_to_expected_ratio"] == pytest.approx(0.71)
    assert edge_realism["bucket_count"] == 1


def test_format_text_report_handles_missing_inputs(tmp_path: Path) -> None:
    report = rpt.build_report(
        trade_history_path=tmp_path / "missing_trade_history.json",
        gate_summary_path=tmp_path / "missing_gate_summary.json",
    )

    output = rpt.format_text_report(report)

    assert "exists=False" in output
    assert "Realized pnl: unavailable" in output


def test_build_report_reconstructs_fifo_realized_pnl(tmp_path: Path) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"

    trade_history_path.write_text(
        json.dumps(
            [
                {
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": 10,
                    "entry_price": 100.0,
                    "entry_time": "2026-01-02T14:30:00+00:00",
                    "fee_bps": 10,
                    "slippage_bps": 2,
                    "strategy": "alpha",
                },
                {
                    "symbol": "AAPL",
                    "side": "sell",
                    "qty": 10,
                    "entry_price": 103.0,
                    "entry_time": "2026-01-02T20:30:00+00:00",
                    "fee_bps": 10,
                    "slippage_bps": 1,
                    "strategy": "alpha",
                },
                {
                    "symbol": "MSFT",
                    "side": "sell",
                    "qty": 5,
                    "entry_price": 200.0,
                    "entry_time": "2026-01-02T15:00:00+00:00",
                    "fee_bps": 10,
                    "slippage_bps": 3,
                    "strategy": "beta",
                },
                {
                    "symbol": "MSFT",
                    "side": "buy",
                    "qty": 5,
                    "entry_price": 210.0,
                    "entry_time": "2026-01-03T16:00:00+00:00",
                    "fee_bps": 10,
                    "slippage_bps": 2,
                    "strategy": "beta",
                },
            ]
        ),
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps({"total_records": 0, "total_accepted_records": 0, "total_rejected_records": 0}),
        encoding="utf-8",
    )

    report = rpt.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
    )
    trade = report["trade_history"]

    assert trade["pnl_available"] is True
    assert trade["pnl_source"] == "fifo_reconstructed_from_fills"
    assert trade["pnl_records"] == 2
    assert trade["open_lot_count"] == 0
    assert trade["pnl_sum"] == pytest.approx(-24.893, rel=1e-6)
    assert trade["top_loss_drivers"]["symbols"][0]["name"] == "MSFT"
    assert trade["daily_expectancy"][-1]["date"] == "2026-01-03"


def test_build_report_separates_live_and_reconcile_fill_expectancy(tmp_path: Path) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"
    order_events_path = tmp_path / "order_events.jsonl"

    trade_history_path.write_text(
        json.dumps(
            [
                {
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": 1,
                    "entry_price": 100.0,
                    "entry_time": "2026-02-02T14:30:00+00:00",
                    "order_id": "live-1",
                },
                {
                    "symbol": "AAPL",
                    "side": "sell",
                    "qty": 1,
                    "entry_price": 101.0,
                    "entry_time": "2026-02-02T15:30:00+00:00",
                    "order_id": "live-2",
                },
                {
                    "symbol": "MSFT",
                    "side": "buy",
                    "qty": 1,
                    "entry_price": 200.0,
                    "entry_time": "2026-02-02T16:00:00+00:00",
                    "order_id": "reconcile-1",
                },
                {
                    "symbol": "MSFT",
                    "side": "sell",
                    "qty": 1,
                    "entry_price": 190.0,
                    "entry_time": "2026-02-02T17:00:00+00:00",
                    "order_id": "reconcile-2",
                },
            ]
        ),
        encoding="utf-8",
    )
    order_events_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "ts": "2026-02-02T14:30:10+00:00",
                        "order_id": "live-1",
                        "source": "initial",
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-02-02T15:30:10+00:00",
                        "order_id": "live-2",
                        "source": "final",
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-02-02T16:00:10+00:00",
                        "order_id": "reconcile-1",
                        "source": "broker_reconcile",
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-02-02T17:00:10+00:00",
                        "order_id": "reconcile-2",
                        "source": "broker_reconcile",
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps({"total_records": 0, "total_accepted_records": 0, "total_rejected_records": 0}),
        encoding="utf-8",
    )

    report = rpt.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
    )
    trade = report["trade_history"]

    assert trade["pnl_available"] is True
    assert trade["closed_trades"] == 2
    assert trade["closed_trades_by_fill_source"]["live"] == 1
    assert trade["closed_trades_by_fill_source"]["reconcile_backfill"] == 1
    assert trade["daily_expectancy_live"][0]["net_pnl"] == pytest.approx(1.0)
    assert trade["daily_expectancy_reconcile_backfill"][0]["net_pnl"] == pytest.approx(-10.0)


def test_build_order_source_lookup_prefers_live_source_over_reconcile(tmp_path: Path) -> None:
    order_events_path = tmp_path / "order_events.jsonl"
    order_events_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "ts": "2026-02-02T14:30:00+00:00",
                        "order_id": "mixed-order",
                        "source": "initial",
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-02-02T14:31:00+00:00",
                        "order_id": "mixed-order",
                        "source": "broker_reconcile",
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-02-02T14:32:00+00:00",
                        "order_id": "mixed-order",
                        "source": None,
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-02-02T14:30:00+00:00",
                        "order_id": "reconcile-only-order",
                        "source": "broker_reconcile",
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    lookup = rpt._build_order_source_lookup(order_events_path)

    assert lookup["mixed-order"] == "live"
    assert lookup["reconcile-only-order"] == "reconcile_backfill"


def test_build_report_prefers_live_lookup_when_row_source_is_reconcile(
    tmp_path: Path,
) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"
    order_events_path = tmp_path / "order_events.jsonl"

    trade_history_path.write_text(
        json.dumps(
            [
                {
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": 1,
                    "entry_price": 100.0,
                    "entry_time": "2026-02-03T14:30:00+00:00",
                    "order_id": "mix-live-1",
                    "source": "broker_reconcile",
                },
                {
                    "symbol": "AAPL",
                    "side": "sell",
                    "qty": 1,
                    "entry_price": 101.0,
                    "entry_time": "2026-02-03T15:30:00+00:00",
                    "order_id": "mix-live-2",
                    "source": "broker_reconcile",
                },
            ]
        ),
        encoding="utf-8",
    )
    order_events_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "ts": "2026-02-03T14:30:10+00:00",
                        "order_id": "mix-live-1",
                        "source": "initial",
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-02-03T14:31:10+00:00",
                        "order_id": "mix-live-1",
                        "source": "broker_reconcile",
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-02-03T15:30:10+00:00",
                        "order_id": "mix-live-2",
                        "source": "initial",
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-02-03T15:31:10+00:00",
                        "order_id": "mix-live-2",
                        "source": "broker_reconcile",
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps({"total_records": 0, "total_accepted_records": 0, "total_rejected_records": 0}),
        encoding="utf-8",
    )

    report = rpt.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
    )
    trade = report["trade_history"]

    assert trade["closed_trades"] == 1
    assert trade["closed_trades_by_fill_source"].get("live", 0) == 1
    assert trade["closed_trades_by_fill_source"].get("reconcile_backfill", 0) == 0


def test_build_report_enriches_direct_rows_with_tca_costs(tmp_path: Path) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"
    tca_path = tmp_path / "tca_records.jsonl"

    trade_history_path.write_text(
        json.dumps(
            [
                {
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": 10,
                    "entry_price": 100.0,
                    "exit_price": 101.0,
                    "entry_time": "2026-02-01T14:30:00+00:00",
                    "exit_time": "2026-02-01T15:00:00+00:00",
                    "reward": 10.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps({"total_records": 1, "total_accepted_records": 1, "total_rejected_records": 0}),
        encoding="utf-8",
    )
    tca_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "symbol": "AAPL",
                        "side": "buy",
                        "qty": 10,
                        "fill_price": 100.0,
                        "ts": "2026-02-01T14:30:00+00:00",
                        "fees": 1.0,
                        "is_bps": 5.0,
                    }
                ),
                json.dumps(
                    {
                        "symbol": "AAPL",
                        "side": "sell",
                        "qty": 10,
                        "fill_price": 101.0,
                        "ts": "2026-02-01T15:00:00+00:00",
                        "fees": 1.0,
                        "is_bps": 6.0,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = rpt.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
        tca_path=tca_path,
    )
    trade = report["trade_history"]

    assert trade["pnl_available"] is True
    assert trade["closed_trades"] == 1
    assert trade["total_fee_cost"] == pytest.approx(2.0)
    assert trade["total_slippage_cost"] == pytest.approx(1.106, rel=1e-6)
    assert trade["pnl_sum"] == pytest.approx(6.894, rel=1e-6)
    assert trade["cost_enrichment"]["matched_legs"] == 2
    assert trade["cost_enrichment"]["enriched_trades"] == 1
    assert trade["cost_attribution"]["fee_sources"]["tca_matched"] == 1
    assert trade["cost_attribution"]["slippage_sources"]["tca_matched"] == 1
    assert trade["daily_expectancy"][0]["fee_cost"] == pytest.approx(2.0)
    assert trade["daily_expectancy"][0]["slippage_cost"] == pytest.approx(1.106, rel=1e-6)


def test_resolve_fee_amount_uses_env_fallback_bps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRADING_RUNTIME_PERF_FEE_BPS_FALLBACK", "12")
    rpt._runtime_fee_bps_fallback.cache_clear()
    amount, source = rpt._resolve_fee_amount_with_source({}, qty=10.0, price=100.0)
    assert amount == pytest.approx(1.2)
    assert source == "env_fee_bps_fallback"
    rpt._runtime_fee_bps_fallback.cache_clear()


def test_build_report_includes_broker_open_position_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"
    trade_history_path.write_text(
        json.dumps([{"symbol": "AAPL", "side": "buy", "pnl": 10.0}]),
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps(
            {
                "total_records": 10,
                "total_accepted_records": 4,
                "total_rejected_records": 6,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        rpt,
        "_summarize_broker_open_positions",
        lambda: {
            "broker_open_positions_available": True,
            "broker_open_position_count": 2,
            "broker_open_positions": {"AAPL": 5.0, "MSFT": -3.0},
            "broker_open_positions_error": None,
        },
    )

    report = rpt.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
    )

    trade = report["trade_history"]
    assert trade["broker_open_positions_available"] is True
    assert trade["broker_open_position_count"] == 2
    assert trade["broker_open_positions"] == {"AAPL": 5.0, "MSFT": -3.0}


def test_build_report_reports_reconstructed_open_position_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"
    trade_history_path.write_text(
        json.dumps(
            [
                {
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": 10,
                    "entry_price": 100.0,
                    "entry_time": "2026-03-01T15:00:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps(
            {
                "total_records": 0,
                "total_accepted_records": 0,
                "total_rejected_records": 0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        rpt,
        "_summarize_broker_open_positions",
        lambda: {
            "broker_open_positions_available": True,
            "broker_open_position_count": 1,
            "broker_open_positions": {"AAPL": 10.0},
            "broker_open_positions_error": None,
        },
    )

    report = rpt.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
    )

    trade = report["trade_history"]
    assert trade["pnl_available"] is False
    assert trade["reconstructed_open_lot_count"] == 1
    assert trade["reconstructed_open_position_count"] == 1
    assert trade["reconstructed_open_positions"] == {"AAPL": 10.0}
    assert trade["open_lot_count"] == 1
    assert trade["open_positions"] == {"AAPL": 10.0}
    reconciliation = trade["open_position_reconciliation"]
    assert reconciliation["available"] is True
    assert reconciliation["symbol_mismatch_count"] == 0


def test_build_report_flags_broker_vs_reconstructed_position_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"
    trade_history_path.write_text(
        json.dumps(
            [
                {
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": 10,
                    "entry_price": 100.0,
                    "entry_time": "2026-03-01T15:00:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps(
            {
                "total_records": 0,
                "total_accepted_records": 0,
                "total_rejected_records": 0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        rpt,
        "_summarize_broker_open_positions",
        lambda: {
            "broker_open_positions_available": True,
            "broker_open_position_count": 1,
            "broker_open_positions": {"AAPL": 7.0},
            "broker_open_positions_error": None,
        },
    )

    report = rpt.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
    )

    reconciliation = report["trade_history"]["open_position_reconciliation"]
    assert reconciliation["available"] is True
    assert reconciliation["symbol_mismatch_count"] == 1
    assert reconciliation["top_mismatches"][0]["symbol"] == "AAPL"
    assert reconciliation["top_mismatches"][0]["delta_qty"] == pytest.approx(3.0)


def test_build_report_prefers_fill_events_for_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    fill_events_path = tmp_path / "fill_events.jsonl"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"
    trade_history_path.write_text(
        json.dumps(
            [
                {
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": 10,
                    "entry_price": 100.0,
                    "entry_time": "2026-03-01T15:00:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )
    fill_events_path.write_text(
        json.dumps(
            {
                "symbol": "AAPL",
                "side": "buy",
                "qty": 7,
                "entry_price": 100.0,
                "entry_time": "2026-03-01T15:00:00+00:00",
                "order_id": "ord-1",
                "fill_id": "fill-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps(
            {
                "total_records": 0,
                "total_accepted_records": 0,
                "total_rejected_records": 0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        rpt,
        "_summarize_broker_open_positions",
        lambda: {
            "broker_open_positions_available": True,
            "broker_open_position_count": 1,
            "broker_open_positions": {"AAPL": 7.0},
            "broker_open_positions_error": None,
        },
    )

    report = rpt.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
        fill_events_path=fill_events_path,
    )

    trade = report["trade_history"]
    assert trade["open_positions"] == {"AAPL": 10.0}
    assert trade["reconciliation_open_positions_source"] == "fill_events"
    reconciliation = trade["open_position_reconciliation"]
    assert reconciliation["available"] is True
    assert reconciliation["source"] == "fill_events"
    assert reconciliation["symbol_mismatch_count"] == 0


def test_build_report_ignores_non_fill_tca_status_rows(tmp_path: Path) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"
    tca_path = tmp_path / "tca_records.jsonl"

    trade_history_path.write_text(
        json.dumps(
            [
                {
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": 10,
                    "entry_price": 100.0,
                    "exit_price": 101.0,
                    "entry_time": "2026-02-01T14:30:00+00:00",
                    "exit_time": "2026-02-01T15:00:00+00:00",
                    "reward": 10.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps({"total_records": 1, "total_accepted_records": 1, "total_rejected_records": 0}),
        encoding="utf-8",
    )
    tca_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "symbol": "AAPL",
                        "side": "buy",
                        "qty": 10,
                        "fill_price": 100.0,
                        "ts": "2026-02-01T14:30:00+00:00",
                        "status": "OrderStatus.PENDING_NEW",
                        "fees": 0.0,
                        "is_bps": 0.0,
                    }
                ),
                json.dumps(
                    {
                        "symbol": "AAPL",
                        "side": "buy",
                        "qty": 10,
                        "fill_price": 100.0,
                        "ts": "2026-02-01T14:30:00+00:00",
                        "status": "filled",
                        "fees": 1.0,
                        "is_bps": 5.0,
                    }
                ),
                json.dumps(
                    {
                        "symbol": "AAPL",
                        "side": "sell",
                        "qty": 10,
                        "fill_price": 101.0,
                        "ts": "2026-02-01T15:00:00+00:00",
                        "status": "submitted",
                        "fees": 0.0,
                        "is_bps": 0.0,
                    }
                ),
                json.dumps(
                    {
                        "symbol": "AAPL",
                        "side": "sell",
                        "qty": 10,
                        "fill_price": 101.0,
                        "ts": "2026-02-01T15:00:00+00:00",
                        "status": "partially_filled",
                        "fees": 1.0,
                        "is_bps": 6.0,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = rpt.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
        tca_path=tca_path,
    )
    trade = report["trade_history"]

    assert trade["closed_trades"] == 1
    assert trade["total_fee_cost"] == pytest.approx(2.0)
    assert trade["cost_enrichment"]["matched_legs"] == 2
    assert trade["cost_attribution"]["fee_sources"]["tca_matched"] == 1


def test_evaluate_go_no_go_uses_thresholds() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "closed_trades": 40,
            "profit_factor": 1.35,
            "win_rate": 0.58,
            "pnl_sum": 245.0,
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.18,
            "total_expected_net_edge_bps": 22.0,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "min_closed_trades": 20,
            "min_profit_factor": 1.1,
            "min_win_rate": 0.5,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.1,
            "min_expected_net_edge_bps": 0.0,
            "trade_fill_source": "all",
            "auto_live_fail_closed": False,
            "require_open_position_reconciliation": False,
            "require_pnl_available": True,
            "require_gate_valid": True,
        },
    )

    assert decision["gate_passed"] is True
    assert decision["failed_checks"] == []


def test_evaluate_go_no_go_expected_edge_uses_per_fill_metric_when_available() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "closed_trades": 40,
            "profit_factor": 1.35,
            "win_rate": 0.58,
            "pnl_sum": 245.0,
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.18,
            "accepted_records": 1_000,
            "total_expected_net_edge_bps": 200_000.0,
        },
        "expected_edge_per_filled_trade_bps": 4.5,
        "execution_vs_alpha": {
            "expected_edge_per_traded_notional_bps": 5.0,
            "expected_edge_for_realism_bps": 5.2,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "min_closed_trades": 20,
            "min_profit_factor": 1.1,
            "min_win_rate": 0.5,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.1,
            "min_expected_net_edge_bps": 10.0,
            "trade_fill_source": "all",
            "auto_live_fail_closed": False,
            "require_open_position_reconciliation": False,
            "require_pnl_available": True,
            "require_gate_valid": True,
        },
    )

    assert decision["gate_passed"] is False
    assert "expected_net_edge_bps" in decision["failed_checks"]
    assert decision["observed"]["expected_net_edge_bps"] == pytest.approx(4.5)
    assert decision["observed"]["expected_net_edge_bps_source"] == "execution_vs_alpha_per_fill"
    assert decision["observed"]["expected_net_edge_bps_raw_sum"] == pytest.approx(200_000.0)


def test_evaluate_go_no_go_expected_edge_falls_back_to_gate_per_accept() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "closed_trades": 40,
            "profit_factor": 1.35,
            "win_rate": 0.58,
            "pnl_sum": 245.0,
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.18,
            "accepted_records": 50,
            "total_expected_net_edge_bps": 250.0,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "min_closed_trades": 20,
            "min_profit_factor": 1.1,
            "min_win_rate": 0.5,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.1,
            "min_expected_net_edge_bps": 10.0,
            "trade_fill_source": "all",
            "auto_live_fail_closed": False,
            "require_open_position_reconciliation": False,
            "require_pnl_available": True,
            "require_gate_valid": True,
        },
    )

    assert decision["gate_passed"] is False
    assert "expected_net_edge_bps" in decision["failed_checks"]
    assert decision["observed"]["expected_net_edge_bps"] == pytest.approx(5.0)
    assert decision["observed"]["expected_net_edge_bps_source"] == "gate_per_accept"
    assert decision["observed"]["expected_net_edge_per_accept_bps"] == pytest.approx(5.0)
    assert decision["observed"]["expected_net_edge_bps_raw_sum"] == pytest.approx(250.0)


def test_evaluate_go_no_go_fails_when_metrics_are_weak() -> None:
    report = {
        "trade_history": {
            "pnl_available": False,
            "closed_trades": 3,
            "profit_factor": 0.7,
            "win_rate": 0.2,
            "pnl_sum": -50.0,
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.01,
            "total_expected_net_edge_bps": -120.0,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "min_closed_trades": 20,
            "min_profit_factor": 1.1,
            "min_win_rate": 0.5,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.05,
            "min_expected_net_edge_bps": -10.0,
            "require_pnl_available": True,
            "require_gate_valid": True,
        },
    )

    assert decision["gate_passed"] is False
    assert "pnl_available" in decision["failed_checks"]
    assert "profit_factor" in decision["failed_checks"]


def test_evaluate_go_no_go_applies_win_rate_confidence_gate() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "closed_trades": 20,
            "wins": 12,
            "profit_factor": 1.2,
            "win_rate": 0.6,
            "pnl_sum": 40.0,
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.2,
            "total_expected_net_edge_bps": 10.0,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "min_closed_trades": 10,
            "min_profit_factor": 1.0,
            "min_win_rate": 0.5,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.05,
            "min_expected_net_edge_bps": 0.0,
            "min_win_rate_confidence_floor": 0.5,
            "win_rate_confidence_z": 1.96,
            "win_rate_confidence_min_trades": 10,
            "require_pnl_available": True,
            "require_gate_valid": True,
            "require_open_position_reconciliation": False,
        },
    )

    assert decision["gate_passed"] is False
    assert "win_rate_confidence" in decision["failed_checks"]
    assert decision["observed"]["win_rate_confidence_enabled"] is True
    assert decision["observed"]["win_rate_confidence_reason"] == "win_rate_confidence_gate"
    assert (
        decision["observed"]["win_rate_confidence_lower_bound"]
        < decision["thresholds"]["min_win_rate_confidence_floor"]
    )


def test_build_report_includes_daily_gate_stats_from_log(tmp_path: Path) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"
    gate_log_path = tmp_path / "gate_effectiveness.jsonl"
    trade_history_path.write_text("[]", encoding="utf-8")
    gate_summary_path.write_text(
        json.dumps({"total_records": 2, "total_accepted_records": 1, "total_rejected_records": 1}),
        encoding="utf-8",
    )
    gate_log_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "ts": "2026-03-10T18:00:00+00:00",
                        "records_total": 10,
                        "accepted_records": 2,
                        "rejected_records": 8,
                        "total_expected_net_edge_bps": -5.0,
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-03-11T18:00:00+00:00",
                        "records_total": 20,
                        "accepted_records": 4,
                        "rejected_records": 16,
                        "total_expected_net_edge_bps": 7.5,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = rpt.build_report(
        trade_history_path=trade_history_path,
        gate_summary_path=gate_summary_path,
    )

    gate = report["gate_effectiveness"]
    daily = gate["daily_gate_stats"]
    assert gate["gate_log_exists"] is True
    assert len(daily) == 2
    assert daily[0]["date"] == "2026-03-10"
    assert daily[0]["acceptance_rate"] == pytest.approx(0.2)
    assert daily[1]["date"] == "2026-03-11"
    assert daily[1]["total_expected_net_edge_bps"] == pytest.approx(7.5)


def test_evaluate_go_no_go_lookback_days_uses_recent_window_metrics() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "closed_trades": 100,
            "profit_factor": 1.8,
            "win_rate": 0.7,
            "pnl_sum": 400.0,
            "daily_trade_stats": [
                {
                    "date": "2026-03-10",
                    "trades": 50,
                    "wins": 35,
                    "losses": 15,
                    "gross_win_pnl": 350.0,
                    "gross_loss_pnl": 100.0,
                    "net_pnl": 250.0,
                },
                {
                    "date": "2026-03-11",
                    "trades": 50,
                    "wins": 20,
                    "losses": 30,
                    "gross_win_pnl": 150.0,
                    "gross_loss_pnl": 300.0,
                    "net_pnl": -150.0,
                },
            ],
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.2,
            "total_expected_net_edge_bps": 20.0,
            "daily_gate_stats": [
                {
                    "date": "2026-03-10",
                    "total_records": 100,
                    "accepted_records": 20,
                    "rejected_records": 80,
                    "total_expected_net_edge_bps": 15.0,
                },
                {
                    "date": "2026-03-11",
                    "total_records": 100,
                    "accepted_records": 2,
                    "rejected_records": 98,
                    "total_expected_net_edge_bps": -3.0,
                },
            ],
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "min_closed_trades": 10,
            "min_profit_factor": 1.1,
            "min_win_rate": 0.55,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.05,
            "min_expected_net_edge_bps": -50.0,
            "lookback_days": 1,
            "trade_fill_source": "all",
            "auto_live_fail_closed": False,
            "require_open_position_reconciliation": False,
            "require_pnl_available": True,
            "require_gate_valid": True,
        },
    )

    assert decision["gate_passed"] is False
    assert "profit_factor" in decision["failed_checks"]
    assert "win_rate" in decision["failed_checks"]
    assert "net_pnl" in decision["failed_checks"]
    assert "acceptance_rate" in decision["failed_checks"]
    observed = decision["observed"]
    assert observed["trade_metric_scope"]["mode"] == "rolling_days"
    assert observed["gate_metric_scope"]["mode"] == "rolling_days"
    assert observed["closed_trades"] == 50
    assert observed["acceptance_rate"] == pytest.approx(0.02)


def test_evaluate_go_no_go_lookback_days_enforces_min_used_days() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "closed_trades": 6,
            "profit_factor": 1.5,
            "win_rate": 0.66,
            "pnl_sum": 12.0,
            "daily_trade_stats": [
                {
                    "date": "2026-03-11",
                    "trades": 3,
                    "wins": 2,
                    "losses": 1,
                    "gross_win_pnl": 6.0,
                    "gross_loss_pnl": 2.0,
                    "net_pnl": 4.0,
                },
                {
                    "date": "2026-03-12",
                    "trades": 3,
                    "wins": 2,
                    "losses": 1,
                    "gross_win_pnl": 6.0,
                    "gross_loss_pnl": 2.0,
                    "net_pnl": 4.0,
                },
            ],
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.3,
            "total_expected_net_edge_bps": 10.0,
            "daily_gate_stats": [
                {
                    "date": "2026-03-10",
                    "total_records": 10,
                    "accepted_records": 3,
                    "rejected_records": 7,
                    "total_expected_net_edge_bps": 2.0,
                },
                {
                    "date": "2026-03-11",
                    "total_records": 10,
                    "accepted_records": 3,
                    "rejected_records": 7,
                    "total_expected_net_edge_bps": 3.0,
                },
                {
                    "date": "2026-03-12",
                    "total_records": 10,
                    "accepted_records": 3,
                    "rejected_records": 7,
                    "total_expected_net_edge_bps": 4.0,
                },
            ],
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "min_closed_trades": 2,
            "min_profit_factor": 1.0,
            "min_win_rate": 0.5,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.05,
            "min_expected_net_edge_bps": -50.0,
            "lookback_days": 5,
            "min_used_days": 3,
            "trade_fill_source": "all",
            "auto_live_fail_closed": False,
            "require_open_position_reconciliation": False,
            "require_pnl_available": True,
            "require_gate_valid": True,
        },
    )

    assert decision["gate_passed"] is False
    assert "trade_used_days" in decision["failed_checks"]
    assert "gate_used_days" not in decision["failed_checks"]
    observed = decision["observed"]
    assert observed["trade_used_days"] == 2
    assert observed["gate_used_days"] == 3


def test_evaluate_go_no_go_trade_fill_source_uses_source_specific_rows() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "closed_trades": 100,
            "profit_factor": 2.0,
            "win_rate": 0.75,
            "pnl_sum": 500.0,
            "daily_trade_stats_by_fill_source": {
                "live": [
                    {
                        "date": "2026-03-11",
                        "trades": 4,
                        "wins": 1,
                        "losses": 3,
                        "gross_win_pnl": 10.0,
                        "gross_loss_pnl": 20.0,
                        "net_pnl": -10.0,
                    }
                ],
                "reconcile_backfill": [
                    {
                        "date": "2026-03-11",
                        "trades": 20,
                        "wins": 15,
                        "losses": 5,
                        "gross_win_pnl": 120.0,
                        "gross_loss_pnl": 30.0,
                        "net_pnl": 90.0,
                    }
                ],
            },
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.2,
            "total_expected_net_edge_bps": 10.0,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "trade_fill_source": "live",
            "min_closed_trades": 1,
            "min_profit_factor": 1.1,
            "min_win_rate": 0.5,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.05,
            "min_expected_net_edge_bps": -50.0,
            "require_pnl_available": True,
            "require_gate_valid": True,
        },
    )

    assert decision["gate_passed"] is False
    assert "profit_factor" in decision["failed_checks"]
    assert "win_rate" in decision["failed_checks"]
    assert "net_pnl" in decision["failed_checks"]
    observed = decision["observed"]
    assert observed["trade_fill_source"] == "live"
    assert observed["closed_trades"] == 4
    assert observed["trade_metric_scope"]["fill_source"] == "live"


def test_evaluate_go_no_go_all_excludes_reconcile_backfill_by_default() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "closed_trades": 24,
            "profit_factor": 2.6,
            "win_rate": 16 / 24,
            "pnl_sum": 80.0,
            "daily_trade_stats": [
                {
                    "date": "2026-03-11",
                    "trades": 24,
                    "wins": 16,
                    "losses": 8,
                    "gross_win_pnl": 130.0,
                    "gross_loss_pnl": 50.0,
                    "net_pnl": 80.0,
                }
            ],
            "daily_trade_stats_by_fill_source": {
                "live": [
                    {
                        "date": "2026-03-11",
                        "trades": 4,
                        "wins": 1,
                        "losses": 3,
                        "gross_win_pnl": 10.0,
                        "gross_loss_pnl": 20.0,
                        "net_pnl": -10.0,
                    }
                ],
                "reconcile_backfill": [
                    {
                        "date": "2026-03-11",
                        "trades": 20,
                        "wins": 15,
                        "losses": 5,
                        "gross_win_pnl": 120.0,
                        "gross_loss_pnl": 30.0,
                        "net_pnl": 90.0,
                    }
                ],
            },
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.2,
            "total_expected_net_edge_bps": 10.0,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "trade_fill_source": "all",
            "min_closed_trades": 1,
            "min_profit_factor": 1.5,
            "min_win_rate": 0.6,
            "min_net_pnl": 10.0,
            "min_acceptance_rate": 0.05,
            "min_expected_net_edge_bps": -50.0,
            "require_open_position_reconciliation": False,
            "require_pnl_available": True,
            "require_gate_valid": True,
        },
    )

    assert decision["gate_passed"] is False
    assert "profit_factor" in decision["failed_checks"]
    assert "win_rate" in decision["failed_checks"]
    assert "net_pnl" in decision["failed_checks"]
    observed = decision["observed"]
    assert observed["trade_fill_source"] == "all"
    assert observed["closed_trades"] == 4
    assert observed["trade_metric_scope"]["excluded_fill_sources"] == [
        "reconcile_backfill"
    ]


def test_evaluate_go_no_go_all_can_include_reconcile_backfill_when_enabled() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "closed_trades": 24,
            "profit_factor": 2.6,
            "win_rate": 16 / 24,
            "pnl_sum": 80.0,
            "daily_trade_stats": [
                {
                    "date": "2026-03-11",
                    "trades": 24,
                    "wins": 16,
                    "losses": 8,
                    "gross_win_pnl": 130.0,
                    "gross_loss_pnl": 50.0,
                    "net_pnl": 80.0,
                }
            ],
            "daily_trade_stats_by_fill_source": {
                "live": [
                    {
                        "date": "2026-03-11",
                        "trades": 4,
                        "wins": 1,
                        "losses": 3,
                        "gross_win_pnl": 10.0,
                        "gross_loss_pnl": 20.0,
                        "net_pnl": -10.0,
                    }
                ],
                "reconcile_backfill": [
                    {
                        "date": "2026-03-11",
                        "trades": 20,
                        "wins": 15,
                        "losses": 5,
                        "gross_win_pnl": 120.0,
                        "gross_loss_pnl": 30.0,
                        "net_pnl": 90.0,
                    }
                ],
            },
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.2,
            "total_expected_net_edge_bps": 10.0,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "trade_fill_source": "all",
            "exclude_reconcile_backfill_from_metrics": False,
            "min_closed_trades": 1,
            "min_profit_factor": 1.5,
            "min_win_rate": 0.6,
            "min_net_pnl": 10.0,
            "min_acceptance_rate": 0.05,
            "min_expected_net_edge_bps": -50.0,
            "require_open_position_reconciliation": False,
            "require_pnl_available": True,
            "require_gate_valid": True,
        },
    )

    assert decision["gate_passed"] is True
    observed = decision["observed"]
    assert observed["trade_fill_source"] == "all"
    assert observed["closed_trades"] == 24


def test_evaluate_go_no_go_trade_fill_source_normalises_backfill_alias() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "daily_trade_stats_by_fill_source": {
                "reconcile_backfill": [
                    {
                        "date": "2026-03-11",
                        "trades": 8,
                        "wins": 5,
                        "losses": 3,
                        "gross_win_pnl": 30.0,
                        "gross_loss_pnl": 10.0,
                        "net_pnl": 20.0,
                    }
                ]
            },
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.2,
            "total_expected_net_edge_bps": 10.0,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "trade_fill_source": "backfill",
            "min_closed_trades": 1,
            "min_profit_factor": 1.1,
            "min_win_rate": 0.5,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.05,
            "min_expected_net_edge_bps": -50.0,
            "require_open_position_reconciliation": False,
            "require_pnl_available": True,
            "require_gate_valid": True,
        },
    )

    assert decision["gate_passed"] is True
    assert decision["thresholds"]["trade_fill_source"] == "reconcile_backfill"
    assert decision["observed"]["trade_fill_source"] == "reconcile_backfill"
    assert (
        decision["observed"]["trade_metric_scope"]["fill_source"]
        == "reconcile_backfill"
    )


def test_evaluate_go_no_go_auto_live_prefers_live_when_sufficient() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "daily_trade_stats_by_fill_source": {
                "live": [
                    {
                        "date": "2026-03-10",
                        "trades": 40,
                        "wins": 24,
                        "losses": 16,
                        "gross_win_pnl": 150.0,
                        "gross_loss_pnl": 100.0,
                        "net_pnl": 50.0,
                    },
                    {
                        "date": "2026-03-11",
                        "trades": 35,
                        "wins": 20,
                        "losses": 15,
                        "gross_win_pnl": 120.0,
                        "gross_loss_pnl": 80.0,
                        "net_pnl": 40.0,
                    },
                ],
                "all": [
                    {
                        "date": "2026-03-10",
                        "trades": 100,
                        "wins": 55,
                        "losses": 45,
                        "gross_win_pnl": 260.0,
                        "gross_loss_pnl": 200.0,
                        "net_pnl": 60.0,
                    }
                ],
            },
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.2,
            "total_expected_net_edge_bps": 10.0,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "trade_fill_source": "auto_live",
            "auto_live_min_closed_trades": 20,
            "auto_live_min_used_days": 2,
            "auto_live_min_available_days": 2,
            "auto_live_fail_closed": False,
            "min_closed_trades": 10,
            "min_profit_factor": 1.0,
            "min_win_rate": 0.5,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.05,
            "min_expected_net_edge_bps": -50.0,
            "require_open_position_reconciliation": False,
            "require_pnl_available": True,
            "require_gate_valid": True,
        },
    )

    assert decision["thresholds"]["requested_trade_fill_source"] == "auto_live"
    assert decision["thresholds"]["trade_fill_source"] == "live"
    assert decision["observed"]["trade_fill_source"] == "live"
    assert decision["observed"]["auto_live_selection"]["selected"] == "live"
    assert decision["observed"]["auto_live_selection"]["used_live"] is True


def test_evaluate_go_no_go_auto_live_falls_back_to_all_when_live_insufficient() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "daily_trade_stats_by_fill_source": {
                "live": [
                    {
                        "date": "2026-03-11",
                        "trades": 4,
                        "wins": 2,
                        "losses": 2,
                        "gross_win_pnl": 10.0,
                        "gross_loss_pnl": 8.0,
                        "net_pnl": 2.0,
                    }
                ],
                "all": [
                    {
                        "date": "2026-03-10",
                        "trades": 50,
                        "wins": 30,
                        "losses": 20,
                        "gross_win_pnl": 100.0,
                        "gross_loss_pnl": 70.0,
                        "net_pnl": 30.0,
                    },
                    {
                        "date": "2026-03-11",
                        "trades": 55,
                        "wins": 31,
                        "losses": 24,
                        "gross_win_pnl": 120.0,
                        "gross_loss_pnl": 80.0,
                        "net_pnl": 40.0,
                    },
                ],
            },
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.2,
            "total_expected_net_edge_bps": 10.0,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "trade_fill_source": "auto_live",
            "auto_live_min_closed_trades": 20,
            "auto_live_min_used_days": 2,
            "auto_live_min_available_days": 2,
            "auto_live_fail_closed": False,
            "min_closed_trades": 10,
            "min_profit_factor": 1.0,
            "min_win_rate": 0.5,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.05,
            "min_expected_net_edge_bps": -50.0,
            "require_open_position_reconciliation": False,
            "require_pnl_available": True,
            "require_gate_valid": True,
        },
    )

    assert decision["thresholds"]["requested_trade_fill_source"] == "auto_live"
    assert decision["thresholds"]["trade_fill_source"] == "all"
    assert decision["observed"]["trade_fill_source"] == "all"
    assert decision["observed"]["auto_live_selection"]["selected"] == "all"
    assert decision["observed"]["auto_live_selection"]["used_live"] is False


def test_evaluate_go_no_go_auto_live_fail_closed_blocks_when_live_insufficient() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "daily_trade_stats_by_fill_source": {
                "live": [
                    {
                        "date": "2026-03-11",
                        "trades": 4,
                        "wins": 2,
                        "losses": 2,
                        "gross_win_pnl": 10.0,
                        "gross_loss_pnl": 8.0,
                        "net_pnl": 2.0,
                    }
                ],
                "all": [
                    {
                        "date": "2026-03-10",
                        "trades": 50,
                        "wins": 30,
                        "losses": 20,
                        "gross_win_pnl": 100.0,
                        "gross_loss_pnl": 70.0,
                        "net_pnl": 30.0,
                    }
                ],
            },
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.2,
            "total_expected_net_edge_bps": 10.0,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "trade_fill_source": "auto_live",
            "auto_live_min_closed_trades": 20,
            "auto_live_min_used_days": 2,
            "auto_live_min_available_days": 2,
            "auto_live_fail_closed": True,
            "min_closed_trades": 1,
            "min_profit_factor": 0.1,
            "min_win_rate": 0.0,
            "min_net_pnl": -1_000.0,
            "min_acceptance_rate": 0.0,
            "min_expected_net_edge_bps": -1_000.0,
            "require_pnl_available": True,
            "require_gate_valid": True,
        },
    )

    assert decision["gate_passed"] is False
    assert "live_samples_sufficient" in decision["failed_checks"]
    assert decision["thresholds"]["requested_trade_fill_source"] == "auto_live"
    assert decision["thresholds"]["trade_fill_source"] == "live"
    assert decision["thresholds"]["auto_live_fail_closed"] is True
    assert decision["observed"]["auto_live_selection"]["selected"] == "live"
    assert decision["observed"]["auto_live_selection"]["used_live"] is False
    assert (
        decision["observed"]["auto_live_selection"]["reason"]
        == "live_insufficient_fail_closed"
    )


def test_evaluate_go_no_go_fails_when_slippage_drag_bps_exceeds_threshold() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "closed_trades": 100,
            "profit_factor": 1.4,
            "win_rate": 0.6,
            "pnl_sum": 120.0,
            "slippage_drag_bps": 26.0,
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.2,
            "total_expected_net_edge_bps": 15.0,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "min_closed_trades": 10,
            "min_profit_factor": 1.0,
            "min_win_rate": 0.5,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.01,
            "min_expected_net_edge_bps": -50.0,
            "max_slippage_drag_bps": 10.0,
            "require_pnl_available": True,
            "require_gate_valid": True,
        },
    )

    assert decision["gate_passed"] is False
    assert "slippage_drag_bps" in decision["failed_checks"]


def test_evaluate_go_no_go_allows_ratio_soft_breach_when_abs_and_mismatch_are_small() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "closed_trades": 100,
            "profit_factor": 1.4,
            "win_rate": 0.6,
            "pnl_sum": 120.0,
            "open_position_reconciliation": {
                "available": True,
                "abs_delta_ratio": 0.45,
                "max_abs_delta_qty": 10.0,
                "symbol_mismatch_count": 1,
            },
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.2,
            "total_expected_net_edge_bps": 15.0,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "min_closed_trades": 10,
            "min_profit_factor": 1.0,
            "min_win_rate": 0.5,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.01,
            "min_expected_net_edge_bps": -50.0,
            "require_gate_valid": True,
            "require_pnl_available": True,
            "trade_fill_source": "all",
            "require_open_position_reconciliation": True,
            "max_open_position_delta_ratio": 0.2,
            "max_open_position_delta_ratio_hard": 0.5,
            "max_open_position_mismatch_count": 25,
            "max_open_position_abs_delta_qty": 50.0,
        },
    )

    assert decision["gate_passed"] is True
    assert "open_position_reconciliation_consistent" not in decision["failed_checks"]
    observed = decision["observed"]
    assert observed["open_position_reconciliation_ratio_soft_ok"] is False
    assert observed["open_position_reconciliation_ratio_hard_ok"] is True


def test_evaluate_go_no_go_fails_when_open_position_ratio_breaches_hard_limit() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "closed_trades": 100,
            "profit_factor": 1.4,
            "win_rate": 0.6,
            "pnl_sum": 120.0,
            "open_position_reconciliation": {
                "available": True,
                "abs_delta_ratio": 0.55,
                "max_abs_delta_qty": 10.0,
                "symbol_mismatch_count": 1,
            },
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.2,
            "total_expected_net_edge_bps": 15.0,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "min_closed_trades": 10,
            "min_profit_factor": 1.0,
            "min_win_rate": 0.5,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.01,
            "min_expected_net_edge_bps": -50.0,
            "require_gate_valid": True,
            "require_pnl_available": True,
            "trade_fill_source": "all",
            "require_open_position_reconciliation": True,
            "max_open_position_delta_ratio": 0.2,
            "max_open_position_delta_ratio_hard": 0.5,
            "max_open_position_mismatch_count": 25,
            "max_open_position_abs_delta_qty": 50.0,
        },
    )

    assert decision["gate_passed"] is False
    assert "open_position_reconciliation_consistent" in decision["failed_checks"]
    observed = decision["observed"]
    assert observed["open_position_reconciliation_ratio_hard_ok"] is False


def test_evaluate_go_no_go_fails_on_open_position_reconciliation_when_required() -> None:
    report = {
        "trade_history": {
            "pnl_available": True,
            "closed_trades": 100,
            "profit_factor": 1.4,
            "win_rate": 0.6,
            "pnl_sum": 120.0,
            "open_position_reconciliation": {
                "available": True,
                "abs_delta_ratio": 0.45,
                "max_abs_delta_qty": 120.0,
                "symbol_mismatch_count": 30,
            },
        },
        "gate_effectiveness": {
            "valid": True,
            "acceptance_rate": 0.2,
            "total_expected_net_edge_bps": 15.0,
        },
    }

    decision = rpt.evaluate_go_no_go(
        report,
        thresholds={
            "min_closed_trades": 10,
            "min_profit_factor": 1.0,
            "min_win_rate": 0.5,
            "min_net_pnl": 0.0,
            "min_acceptance_rate": 0.01,
            "min_expected_net_edge_bps": -50.0,
            "require_gate_valid": True,
            "require_pnl_available": True,
            "require_open_position_reconciliation": True,
            "max_open_position_delta_ratio": 0.2,
            "max_open_position_mismatch_count": 25,
            "max_open_position_abs_delta_qty": 50.0,
        },
    )

    assert decision["gate_passed"] is False
    assert "open_position_reconciliation_consistent" in decision["failed_checks"]


def test_main_resolves_runtime_paths_from_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    runtime_dir = data_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    trade_history_path = runtime_dir / "trade_history.json"
    gate_summary_path = runtime_dir / "gate_effectiveness_summary.json"
    trade_history_path.write_text(
        json.dumps([{"symbol": "AAPL", "side": "buy", "pnl": 1.0}]),
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps(
            {
                "total_records": 10,
                "total_accepted_records": 2,
                "total_rejected_records": 8,
                "total_expected_net_edge_bps": 1.0,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_dir))
    monkeypatch.setenv(
        "AI_TRADING_RUNTIME_PERF_TRADE_HISTORY_PATH",
        "runtime/trade_history.json",
    )
    monkeypatch.setenv(
        "AI_TRADING_RUNTIME_PERF_GATE_SUMMARY_PATH",
        "runtime/gate_effectiveness_summary.json",
    )
    monkeypatch.setenv("AI_TRADING_RUNTIME_PERF_GATE_LOG_PATH", "")

    exit_code = rpt.main(["--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["trade_history"]["path"] == str(trade_history_path.resolve())
    assert payload["gate_effectiveness"]["path"] == str(gate_summary_path.resolve())
    assert payload["trade_history"]["exists"] is True
    assert payload["gate_effectiveness"]["exists"] is True


def test_main_go_no_go_uses_env_threshold_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    runtime_dir = data_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    trade_history_path = runtime_dir / "trade_history.json"
    gate_summary_path = runtime_dir / "gate_effectiveness_summary.json"
    trade_history_path.write_text(
        json.dumps(
            [
                {"symbol": "AAPL", "side": "buy", "pnl": 1.0},
                {"symbol": "MSFT", "side": "buy", "pnl": -1.0},
                {"symbol": "NVDA", "side": "buy", "pnl": 1.0},
                {"symbol": "GOOGL", "side": "buy", "pnl": -1.0},
                {"symbol": "AMZN", "side": "buy", "pnl": 1.0},
                {"symbol": "META", "side": "buy", "pnl": -1.0},
            ]
        ),
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps(
            {
                "total_records": 100,
                "total_accepted_records": 2,
                "total_rejected_records": 98,
                "total_expected_net_edge_bps": 0.0,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_dir))
    monkeypatch.setenv(
        "AI_TRADING_RUNTIME_PERF_TRADE_HISTORY_PATH",
        "runtime/trade_history.json",
    )
    monkeypatch.setenv(
        "AI_TRADING_RUNTIME_PERF_GATE_SUMMARY_PATH",
        "runtime/gate_effectiveness_summary.json",
    )
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_CLOSED_TRADES", "5")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_PROFIT_FACTOR", "0.8")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_WIN_RATE", "0.4")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_NET_PNL", "-10")
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_ACCEPTANCE_RATE",
        "0.01",
    )
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_EXPECTED_NET_EDGE_BPS",
        "-5",
    )
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_LOOKBACK_DAYS", "7")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_MIN_USED_DAYS", "0")
    monkeypatch.setenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_TRADE_FILL_SOURCE", "all")
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_RUNTIME_GONOGO_AUTO_LIVE_FAIL_CLOSED",
        "0",
    )
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_RUNTIME_GONOGO_REQUIRE_OPEN_POSITION_RECONCILIATION",
        "0",
    )
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_RUNTIME_GONOGO_REQUIRE_PNL_AVAILABLE",
        "1",
    )
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_RUNTIME_GONOGO_REQUIRE_GATE_VALID",
        "1",
    )

    exit_code = rpt.main(["--json", "--go-no-go"])
    payload = json.loads(capsys.readouterr().out)
    decision = payload["go_no_go"]

    assert exit_code == 0
    assert decision["gate_passed"] is True
    assert decision["thresholds"]["min_closed_trades"] == 5
    assert decision["thresholds"]["min_profit_factor"] == pytest.approx(0.8)
    assert decision["thresholds"]["min_acceptance_rate"] == pytest.approx(0.01)
    assert decision["thresholds"]["lookback_days"] == 7


def test_main_fail_on_no_go_returns_nonzero(tmp_path: Path) -> None:
    trade_history_path = tmp_path / "trade_history.json"
    gate_summary_path = tmp_path / "gate_effectiveness_summary.json"
    trade_history_path.write_text(
        json.dumps([{"symbol": "AAPL", "side": "buy", "pnl": -5}]),
        encoding="utf-8",
    )
    gate_summary_path.write_text(
        json.dumps(
            {
                "total_records": 10,
                "total_accepted_records": 0,
                "total_rejected_records": 10,
                "total_expected_net_edge_bps": -100,
            }
        ),
        encoding="utf-8",
    )

    exit_code = rpt.main(
        [
            "--trade-history",
            str(trade_history_path),
            "--gate-summary",
            str(gate_summary_path),
            "--go-no-go",
            "--fail-on-no-go",
            "--min-closed-trades",
            "10",
            "--min-profit-factor",
            "1.2",
        ]
    )

    assert exit_code == 2
