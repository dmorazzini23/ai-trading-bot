from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
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


def test_gate_effectiveness_prefers_blocking_attribution_and_daily_log(tmp_path: Path) -> None:
    summary_path = _write_json(
        tmp_path / "gate_effectiveness_summary.json",
        {
            "total_records": 10,
            "total_accepted_records": 3,
            "total_rejected_records": 7,
            "total_expected_net_edge_bps": 42.5,
            "gate_root_totals": {
                "RISK_LIMIT": 6,
                "OK_TRADE": 99,
                "BANDIT_EXPLORE": 20,
                "MIN_EDGE": 2,
            },
            "gate_root_attribution": {
                "RISK_LIMIT": {
                    "blocked_records": 5,
                    "expected_net_edge_bps_sum": -17.0,
                    "count": 6,
                },
                "OK_TRADE": {"blocked_records": 100},
                "BANDIT_EXPLORE": {"blocked_records": 100},
                "MIN_EDGE": {
                    "blocked_records": 2,
                    "expected_net_edge_bps_sum": -5.0,
                    "count": 2,
                },
            },
            "symbol_attribution": {
                "MSFT": {"expected_net_edge_bps_sum": -8.0, "count": 3},
                "AAPL": {"expected_net_edge_bps_sum": 4.0, "count": 1},
            },
            "regime_attribution": {
                "bear:high": {"expected_net_edge_bps_sum": -12.0, "count": 4}
            },
        },
    )
    log_path = _write_jsonl(
        tmp_path / "gate_effectiveness.jsonl",
        [
            {
                "ts": "2026-04-25T14:00:00Z",
                "records_total": 4,
                "accepted_records": 1,
                "total_expected_net_edge_bps": 10.0,
            },
            "not-json",
            {
                "ts": "2026-04-25T15:00:00+00:00",
                "records_total": 2,
                "accepted_records": 2,
                "rejected_records": 0,
                "total_expected_net_edge_bps": 8.0,
            },
            {
                "ts": "2026-04-26T14:00:00Z",
                "records_total": 3,
                "accepted_records": 0,
                "rejected_records": 3,
                "total_expected_net_edge_bps": -2.0,
            },
        ],
    )

    summary = rpr.summarize_gate_effectiveness(summary_path, gate_log_path=log_path)

    assert summary["valid"] is True
    assert summary["acceptance_rate"] == pytest.approx(0.3)
    assert summary["top_gates_source"] == "gate_root_totals"
    assert summary["top_rejection_gates"] == [
        {"gate": "RISK_LIMIT", "count": 5},
        {"gate": "MIN_EDGE", "count": 2},
    ]
    assert summary["rejection_concentration"][0] == {
        "gate": "RISK_LIMIT",
        "count": 5,
        "ratio": pytest.approx(5 / 7),
    }
    assert summary["top_negative_gate_roots"] == [
        {"name": "RISK_LIMIT", "expected_net_edge_bps_sum": -17.0, "count": 6},
        {"name": "MIN_EDGE", "expected_net_edge_bps_sum": -5.0, "count": 2},
    ]
    assert summary["top_negative_symbols"] == [
        {"name": "MSFT", "expected_net_edge_bps_sum": -8.0, "count": 3}
    ]
    assert summary["daily_gate_stats"] == [
        {
            "date": "2026-04-25",
            "total_records": 6,
            "accepted_records": 3,
            "rejected_records": 3,
            "acceptance_rate": 0.5,
            "total_expected_net_edge_bps": 18.0,
        },
        {
            "date": "2026-04-26",
            "total_records": 3,
            "accepted_records": 0,
            "rejected_records": 3,
            "acceptance_rate": 0.0,
            "total_expected_net_edge_bps": -2.0,
        },
    ]


def test_trade_history_direct_rows_enriches_costs_and_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_SYMBOLS", "")
    monkeypatch.setenv("AI_TRADING_CANARY_SYMBOLS", "")
    monkeypatch.setattr(
        rpr,
        "_summarize_broker_open_positions",
        lambda: {
            "broker_open_positions_available": True,
            "broker_open_position_count": 1,
            "broker_open_positions": {"AAPL": 5.0},
            "broker_open_position_snapshots": {},
            "broker_open_positions_error": None,
        },
    )
    order_events = _write_jsonl(
        tmp_path / "order_events.jsonl",
        [
            {"order_id": "ord-1", "ts": "2026-04-25T13:29:00Z", "source": "live"},
            {
                "order_id": "ord-2",
                "ts": "2026-04-25T13:30:00Z",
                "source": "broker_reconcile",
            },
        ],
    )
    assert order_events.exists()
    trades_path = _write_json(
        tmp_path / "trade_history.json",
        {
            "trades": [
                {
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": 10,
                    "entry_price": 100.0,
                    "exit_price": 102.0,
                    "entry_time": "2026-04-25T13:30:00Z",
                    "exit_time": "2026-04-25T15:45:00Z",
                    "pnl": 20.0,
                    "order_id": "ord-1",
                    "strategy": "mean-reversion",
                    "session_regime": "",
                    "market_regime": "bull",
                    "volatility_regime": "low",
                    "venue": "arcx",
                },
                {
                    "symbol": "MSFT",
                    "side": "sell",
                    "qty": 5,
                    "entry_price": 50.0,
                    "exit_price": 49.0,
                    "timestamp": "2026-04-26T19:30:00Z",
                    "pnl": -7.0,
                    "fee": 0.25,
                    "slippage_cost": 0.75,
                    "order_id": "ord-2",
                    "strategy": "breakout",
                    "session_regime": "closing",
                    "market_regime": "bear",
                    "volatility_regime": "high",
                },
                {"symbol": "IGNORED", "side": "buy"},
            ]
        },
    )
    tca_path = _write_jsonl(
        tmp_path / "tca_records.jsonl",
        [
            {
                "symbol": "AAPL",
                "side": "buy",
                "qty": 10,
                "fill_price": 100.0,
                "first_fill_ts": "2026-04-25T13:30:02Z",
                "status": "filled",
                "fee_amount": 1.0,
                "slippage_cost": 2.0,
            },
            {
                "symbol": "AAPL",
                "side": "sell",
                "qty": 10,
                "fill_price": 102.0,
                "first_fill_ts": "2026-04-25T15:45:00Z",
                "status": "filled",
                "fee_amount": 1.5,
                "slippage_cost": 3.0,
            },
            {
                "symbol": "MSFT",
                "side": "buy",
                "qty": 5,
                "fill_price": 49.0,
                "first_fill_ts": "2026-04-26T19:30:00Z",
                "status": "canceled",
                "fee_amount": 99.0,
            },
        ],
    )

    summary = rpr.summarize_trade_history(trades_path, tca_path=tca_path)

    assert summary["pnl_available"] is True
    assert summary["pnl_source"] == "direct_pnl_rows"
    assert summary["records"] == 3
    assert summary["closed_trades"] == 2
    assert summary["pnl_sum"] == pytest.approx(4.5)
    assert summary["total_fee_cost"] == pytest.approx(2.75)
    assert summary["total_slippage_cost"] == pytest.approx(5.75)
    assert summary["cost_enrichment"] == {
        "enabled": True,
        "source": "tca",
        "records": 3,
        "events_considered": 2,
        "matched_entry_legs": 1,
        "matched_exit_legs": 1,
        "matched_legs": 2,
        "enriched_trades": 1,
        "trades_with_nonzero_fee": 1,
        "trades_with_nonzero_slippage": 1,
    }
    assert summary["closed_trades_by_fill_source"] == {
        "live": 1,
        "reconcile_backfill": 1,
    }
    assert summary["daily_expectancy_by_fill_source"]["live"][0]["date"] == "2026-04-25"
    assert summary["daily_expectancy_by_fill_source"]["reconcile_backfill"][0][
        "date"
    ] == "2026-04-26"
    assert summary["top_loss_drivers"]["symbols"] == [
        {"name": "MSFT", "net_pnl": -8.0}
    ]
    reconciliation = summary["open_position_reconciliation"]
    assert reconciliation["available"] is True
    assert reconciliation["symbol_mismatch_count"] == 2
    assert {
        item["reason"] for item in reconciliation["top_mismatches"]
    } == {"missing_in_broker", "quantity_mismatch"}

    text = rpr.format_text_report(
        {
            "trade_history": summary,
            "gate_effectiveness": {"path": "gate.json", "exists": True, "valid": False},
            "execution_vs_alpha": {
                "available": True,
                "realized_net_edge_bps": 50.0,
                "pre_execution_edge_bps_est": 75.0,
                "slippage_drag_bps": 25.0,
                "execution_capture_ratio": 2 / 3,
                "edge_realism_gap_ratio": 0.5,
            },
            "post_trade_attribution_ledger": {
                "available": True,
                "alpha_error_bps": -5.0,
                "execution_error_bps": -25.0,
                "gate_overblocking_opportunity_bps": 7.0,
            },
        }
    )
    assert "- TCA enrichment: matched_legs=2 enriched_trades=1" in text
    assert "- Closed trades by fill source: {'live': 1, 'reconcile_backfill': 1}" in text
    assert "- Open-position reconciliation: available=True mismatches=2" in text
    assert "- Execution vs alpha: realized_edge_bps=50.0" in text
    assert "- Post-trade attribution: alpha_error_bps=-5.0" in text


def test_state_summaries_normalize_rank_and_render(tmp_path: Path) -> None:
    stale_at = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
    edge_path = _write_json(
        tmp_path / "edge_realism_state.json",
        {
            "version": 2,
            "updated_at": "2026-04-25T15:00:00Z",
            "global": {
                "samples": 12,
                "mean_realized_to_expected_ratio": 0.73,
                "mean_realized_net_edge_bps": 4.2,
                "mean_expected_net_edge_bps": 5.8,
            },
            "buckets": {
                "good": {"samples": 5, "mean_realized_to_expected_ratio": 0.9},
                "bad": {
                    "samples": 8,
                    "mean_realized_to_expected_ratio": 0.25,
                    "mean_realized_net_edge_bps": -2.0,
                    "mean_expected_net_edge_bps": 8.0,
                },
                "empty": {"samples": 0, "mean_realized_to_expected_ratio": 0.1},
                "missing_ratio": {"samples": 3},
            },
        },
    )
    ablation_path = _write_json(
        tmp_path / "policy_ablation_state.json",
        {
            "updated_at": "2026-04-25T16:00:00Z",
            "slices": {
                "late_day": {
                    "events": 7,
                    "accepted": 2,
                    "rejected": 5,
                    "mean_edge_proxy_bps": -4.0,
                },
                "open": {
                    "events": 10,
                    "accepted": 9,
                    "rejected": 1,
                    "mean_edge_proxy_bps": 2.0,
                },
            },
        },
    )
    toggles_path = _write_json(
        tmp_path / "policy_runtime_toggles.json",
        {
            "updated_at": "2026-04-25T16:30:00Z",
            "source_updated_at": "2026-04-25T16:29:00Z",
            "disabled_slices": ["late_day", "", "OPEN"],
            "toggles": {
                "rankers": {
                    "bandit_enabled": False,
                    "counterfactual_enabled": True,
                },
                "disabled_gate_roots": ["risk_limit", "MIN_EDGE", ""],
                "disabled_sleeves": ["Core", "satellite", ""],
            },
        },
    )
    uncertainty_path = _write_json(
        tmp_path / "uncertainty_capital_state.json",
        {
            "updated_at": "2026-04-25T17:00:00Z",
            "total_events": "9",
            "scaled_events": 4,
            "blocked_events": 2,
            "cycle_records": 3,
            "cycle_scaled": 1,
            "cycle_blocked": 1,
            "score_mean": "0.62",
            "scale_mean": 0.85,
            "high_score_threshold": 0.8,
            "bayesian_high_score_posterior": 0.31,
            "quantiles": {"score_p50": 0.5, "score_p80": 0.8, "score_p95": 0.95},
        },
    )
    counterfactual_path = _write_json(
        tmp_path / "counterfactual_learning_state.json",
        {
            "updated_at": stale_at,
            "global": {
                "events": 11,
                "accepted": 6,
                "rejected": 5,
                "accept_rate": 6 / 11,
                "dr_mean_bps": 1.25,
                "ips_mean_bps": -0.75,
                "missed_dr_sum_bps": 12.0,
            },
            "buckets": {"AAPL": {}, "MSFT": {}},
        },
    )

    edge = rpr.summarize_edge_realism_state(edge_path)
    ablation = rpr.summarize_policy_ablation_state(ablation_path)
    toggles = rpr.summarize_policy_runtime_toggles(toggles_path)
    uncertainty = rpr.summarize_uncertainty_capital_state(uncertainty_path)
    counterfactual = rpr.summarize_counterfactual_learning_state(counterfactual_path)

    assert edge["worst_buckets"][0]["bucket"] == "bad"
    assert edge["bucket_count"] == 2
    assert ablation["worst_slices"][0] == {
        "slice": "late_day",
        "events": 7,
        "accepted": 2,
        "rejected": 5,
        "mean_edge_proxy_bps": -4.0,
    }
    assert toggles["disabled_slices"] == ["LATE_DAY", "OPEN"]
    assert toggles["ranker_toggles"] == {
        "bandit_enabled": False,
        "counterfactual_enabled": True,
        "geometric_enabled": True,
        "portfolio_log_growth_enabled": True,
    }
    assert toggles["disabled_gate_roots"] == ["MIN_EDGE", "RISK_LIMIT"]
    assert toggles["disabled_sleeves"] == ["core", "satellite"]
    assert uncertainty["score_p95"] == 0.95
    assert counterfactual["active_learning"] is True
    assert counterfactual["warning"] is None
    assert counterfactual["bucket_count"] == 2
    assert counterfactual["stale_hours"] >= 0.0

    text = rpr.format_text_report(
        {
            "trade_history": {"path": "trades.json", "exists": True},
            "gate_effectiveness": {"path": "gate.json", "exists": True},
            "edge_realism": edge,
            "policy_ablation": ablation,
            "policy_runtime_toggles": toggles,
            "uncertainty_capital": uncertainty,
        }
    )
    assert "- Realized pnl: unavailable" in text
    assert "- Edge realism calibration: samples=12 ratio=0.73" in text
    assert "- Policy ablation diagnostics: slices=2" in text
    assert "- Policy runtime toggles: disabled_slices=2" in text
    assert "- Uncertainty capital diagnostics: score_mean=0.62" in text


def test_edge_helpers_cover_fallbacks_and_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rpr,
        "get_env",
        lambda name, default=None, cast=None: {
            "AI_TRADING_RUNTIME_PERF_FEE_BPS_FALLBACK": "nan",
            "AI_TRADING_ESTIMATED_FEE_BPS": "1.5",
            "AI_TRADING_POLICY_FEE_BPS": "2.5",
            "AI_TRADING_RUNTIME_PERF_EXPECTED_EDGE_CLIP_LOWER_Q": "bad",
            "AI_TRADING_RUNTIME_PERF_EXPECTED_EDGE_CLIP_UPPER_Q": "0.20",
            "AI_TRADING_RUNTIME_PERF_EXPECTED_EDGE_CLIP_MIN_SAMPLES": "2",
            "AI_TRADING_RUNTIME_PERF_EXPECTED_EDGE_ABS_CAP_BPS": "2",
        }.get(name, default),
    )
    rpr._runtime_fee_bps_fallback.cache_clear()

    assert rpr._as_bool("yes") is True
    assert rpr._as_bool("off") is False
    assert rpr._as_bool("maybe") is None
    assert rpr._confidence_z_from_level(None) is None
    assert rpr._confidence_z_from_level(float("inf")) is None
    assert rpr._confidence_z_from_level(0.0) == 0.0
    assert rpr._confidence_z_from_level(0.997) == 2.807
    assert rpr._confidence_z_from_level(0.50) == 0.126
    assert rpr._wilson_lower_bound(5, 0, 1.96) is None
    assert rpr._wilson_lower_bound(5, 10, 0.0) is None
    assert rpr._wilson_lower_bound(20, 10, 1.96) == pytest.approx(0.7224598312)
    assert rpr._percentile([1.0, float("nan"), 5.0, 9.0], 0.25) == pytest.approx(3.0)
    assert rpr._percentile([], 0.5) is None
    assert rpr._runtime_fee_bps_fallback() == 1.5
    assert rpr._expected_edge_clip_config() == {
        "lower_q": 0.05,
        "upper_q": 0.2,
        "min_samples": 3.0,
        "abs_cap_bps": 5.0,
    }
    assert rpr._normalise_fill_source("broker_reconcile_final") == "reconcile_backfill"
    assert rpr._normalise_fill_source("manual_probe") == "live"
    assert rpr._normalise_fill_source("nan") == "unknown"
    assert rpr._normalise_side("short") == "sell"
    assert rpr._normalise_status_token("OrderStatus.PARTIAL_FILL") == "partially_filled"
    assert rpr._session_bucket_from_timestamp(None) == "unknown"
    assert rpr._trade_session_bucket({"exit_time": "bad", "entry_time": "bad"}) == "unknown"
    assert rpr._trade_regime_bucket({"market_regime": "trend", "volatility_regime": "low"}) == "trend:low"

    rpr._runtime_fee_bps_fallback.cache_clear()
