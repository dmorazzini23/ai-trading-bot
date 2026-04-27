from __future__ import annotations

import logging
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import ai_trading.data.fetch as data_fetch
from ai_trading.core import bot_engine


class _BadStr:
    def __str__(self) -> str:
        raise ValueError("no text")


def _patch_get_env(monkeypatch: pytest.MonkeyPatch, values: dict[str, Any]) -> None:
    def _fake_get_env(key: str, default: Any = None, *, cast: Any = None, **_: Any) -> Any:
        value = values.get(key, default)
        if isinstance(value, BaseException):
            raise value
        if cast is not None and value is not None:
            return cast(value)
        return value

    monkeypatch.setattr(bot_engine, "get_env", _fake_get_env)


def test_tca_stats_and_sequential_significance_helper_edges() -> None:
    assert bot_engine._mean_std([1.0, 3.0, float("nan"), "skip"]) == (2.0, 1.0, 2)
    assert bot_engine._mean_std([]) == (0.0, 0.0, 0)
    assert bot_engine._percentile_linear([10.0, 20.0, 30.0], 0.25) == pytest.approx(15.0)
    assert bot_engine._percentile_linear([7.0], 99.0) == 7.0
    assert bot_engine._percentile_linear([float("inf"), "bad"], 0.5) is None

    assert bot_engine._infer_tca_liquidity_role({"liquidity_role": " Maker "}) == "maker"
    assert bot_engine._infer_tca_liquidity_role({"order_type": "market"}) == "taker"
    assert bot_engine._infer_tca_liquidity_role({"spread_paid_bps": 0.01}) == "maker"
    assert bot_engine._infer_tca_liquidity_role({"spread_paid_bps": 1.2}) == "taker"
    assert bot_engine._infer_tca_liquidity_role({"order_type": "stop_limit"}) == "maker"
    assert bot_engine._infer_tca_liquidity_role({}) == "mixed"

    assert bot_engine._extract_tca_fill_success_ratio({"pending_event": True}) is None
    assert bot_engine._extract_tca_fill_success_ratio({"status": "rejected"}) == 0.0
    assert (
        bot_engine._extract_tca_fill_success_ratio(
            {"status": "partially_filled", "resolved_fill_qty": 3, "requested_qty": 4}
        )
        == 0.75
    )
    assert (
        bot_engine._extract_tca_fill_success_ratio(
            {"status": "filled", "qty": 9, "original_qty": 3}
        )
        == 1.0
    )
    assert (
        bot_engine._extract_tca_fill_success_ratio(
            {"status": "partially_filled", "partial_fill": True}
        )
        == 0.5
    )
    assert bot_engine._extract_tca_fill_success_ratio({"status": "new"}) is None

    insufficient = bot_engine._sequential_significance_gate(
        mean_reward_bps=1.0,
        std_reward_bps=2.0,
        samples=1,
        min_samples=3,
        target_mean_bps=0.0,
        method="either",
        posterior_prob_min=0.9,
        sprt_alpha=0.05,
        sprt_beta=0.05,
        sprt_effect_bps=1.0,
    )
    assert insufficient["reason"] == "insufficient_samples"

    strong = bot_engine._sequential_significance_gate(
        mean_reward_bps=5.0,
        std_reward_bps=1.0,
        samples=40,
        min_samples=3,
        target_mean_bps=0.0,
        method="both",
        posterior_prob_min=0.9,
        sprt_alpha=0.05,
        sprt_beta=0.05,
        sprt_effect_bps=1.0,
    )
    assert strong["passed"] is True
    assert strong["sprt_state"] == "accept"

    weak = bot_engine._sequential_significance_gate(
        mean_reward_bps=-5.0,
        std_reward_bps=1.0,
        samples=40,
        min_samples=3,
        target_mean_bps=0.0,
        method="sprt",
        posterior_prob_min=0.9,
        sprt_alpha=0.05,
        sprt_beta=0.05,
        sprt_effect_bps=1.0,
    )
    assert weak["passed"] is False
    assert weak["sprt_state"] == "reject"


def test_correlation_matrix_and_small_normalizers() -> None:
    matrix = bot_engine._build_symbol_return_correlation_matrix(
        {
            "AAPL": [1.0, 2.0, 3.0],
            "MSFT": [1.0, 2.0, 4.0],
            "FLAT": [5.0, 5.0, 5.0],
            "SHORT": [1.0],
        }
    )

    assert matrix["AAPL"]["AAPL"] == 1.0
    assert matrix["AAPL"]["MSFT"] == pytest.approx(0.9819805)
    assert matrix["AAPL"]["FLAT"] == 0.0
    assert matrix["AAPL"]["SHORT"] == 0.0
    assert bot_engine._portfolio_optimizer_decision_token(SimpleNamespace(value=" BLOCK ")) == "block"
    assert bot_engine._portfolio_optimizer_decision_token(None) == "unknown"
    assert bot_engine._normalize_price_source_label(_BadStr()) == ""
    assert bot_engine._normalize_price_source_label(" N/A ") == ""
    assert bot_engine._normalize_price_source_label(" Alpaca_IEX ") == "alpaca_iex"
    assert bot_engine._clamp_unit_interval("1.5", default=0.3) == 1.0
    assert bot_engine._clamp_unit_interval(float("nan"), default=0.3) == 0.3


def test_feed_sanitizers_and_reliability_adjustments(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(bot_engine, "_pytest_running", lambda: False)
    monkeypatch.setattr(
        bot_engine,
        "data_fetcher_module",
        SimpleNamespace(
            _state={"sip_unauthorized": True},
            _SIP_UNAUTHORIZED=False,
            _normalize_feed_value=lambda raw: "iex" if raw == "alpaca_iex" else None,
        ),
    )

    assert bot_engine._sanitize_alpaca_feed(None) is None
    assert bot_engine._sanitize_alpaca_feed("sip") == "iex"
    assert bot_engine._sanitize_alpaca_feed("bogus") is None
    assert bot_engine._canonicalize_fallback_feed("alpaca_sip") == "sip"
    assert bot_engine._canonicalize_fallback_feed("alpaca_iex") == "iex"
    assert bot_engine._canonicalize_fallback_feed(_BadStr()) is None

    monkeypatch.setattr(data_fetch, "_state", {"sip_unauthorized": True}, raising=False)
    monkeypatch.setattr(data_fetch, "_SIP_UNAUTHORIZED", False, raising=False)
    assert bot_engine._sanitized_alpaca_feed_for_quote("sip") is None
    assert bot_engine._sanitized_alpaca_feed_for_quote("iex") == "iex"
    assert bot_engine._sanitized_alpaca_feed_for_quote("bad") is None

    reliability = {
        "active": True,
        "size_multiplier": 0.4,
        "score": 0.4,
        "sample_count": 12,
        "as_of": "2026-04-27T12:00:00Z",
    }
    with caplog.at_level(logging.INFO):
        assert (
            bot_engine._apply_feed_reliability_size_adjustment(
                symbol="AAPL",
                side="buy",
                qty=10,
                reliability=reliability,
            )
            == 4
        )
    assert any(record.getMessage() == "ORDER_SIZE_SCALED_FEED_RELIABILITY" for record in caplog.records)

    annotations: dict[str, Any] = {}
    bot_engine._annotate_feed_reliability(annotations, reliability)
    assert annotations == {
        "feed_reliability_score": 0.4,
        "feed_reliability_sample_count": 12,
        "feed_reliability_size_multiplier": 0.4,
        "feed_reliability_as_of": "2026-04-27T12:00:00Z",
    }
    assert (
        bot_engine._apply_feed_reliability_size_adjustment(
            symbol="AAPL",
            side="buy",
            qty=3,
            reliability={"active": True, "size_multiplier": 0.01},
        )
        == 1
    )
    assert (
        bot_engine._apply_feed_reliability_size_adjustment(
            symbol="AAPL",
            side="buy",
            qty=3,
            reliability={"active": False},
        )
        == 3
    )


def test_feed_reliability_payload_clamps_and_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get_env(
        monkeypatch,
        {
            "AI_TRADING_FEED_RELIABILITY_MIN_SAMPLES": 3,
            "AI_TRADING_FEED_RELIABILITY_LIVE_MIN_SAMPLES": 4,
            "AI_TRADING_FEED_RELIABILITY_MIN_SCORE": 0.35,
            "AI_TRADING_FEED_RELIABILITY_SIZE_FLOOR": 0.25,
            "AI_TRADING_FEED_RELIABILITY_THRESHOLD_BONUS_MAX": 0.1,
        },
    )
    monkeypatch.setattr(bot_engine, "_feed_reliability_enabled", lambda: True)
    monkeypatch.setattr(
        bot_engine,
        "_load_feed_reliability_scores",
        lambda: (
            {
                "AAPL": {"reliability_score": 0.2, "sample_count": 5},
                "MSFT": {"reliability_score": 0.9, "sample_count": 1},
                "BAD": {"reliability_score": None, "sample_count": "bad"},
            },
            "2026-04-27",
        ),
    )

    aapl = bot_engine._get_symbol_feed_reliability("aapl")
    assert aapl["active"] is True
    assert aapl["blocked"] is True
    assert aapl["size_multiplier"] == 0.25
    assert aapl["threshold_bonus"] == pytest.approx(0.08)

    msft = bot_engine._get_symbol_feed_reliability("MSFT")
    assert msft["active"] is False
    assert msft["score"] == 0.9

    assert bot_engine._get_symbol_feed_reliability("")["active"] is False
    assert bot_engine._get_symbol_feed_reliability("BAD")["score"] is None


def test_learning_and_replay_helpers_parse_bounded_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_env(
        monkeypatch,
        {
            "AI_TRADING_ALLOCATION_ENABLED": True,
            "AI_TRADING_ALLOC_DAY_BASE_WEIGHT": 2.0,
            "AI_TRADING_ALLOC_SWING_BASE_WEIGHT": 1.0,
            "AI_TRADING_ALLOC_LONGSHORT_BASE_WEIGHT": 1.0,
            "AI_TRADING_ALLOCATION_OUTPUT_PATH": "ignored.json",
            "AI_TRADING_REPLAY_INITIAL_POSITIONS_JSON": '{"aapl": 2, "flat": 0, "bad": "x"}',
            "AI_TRADING_REPLAY_INITIAL_POSITIONS_PATH": "",
        },
    )
    monkeypatch.setattr(bot_engine, "load_allocation_state", lambda _path: {"swing": 3.0})

    weights = bot_engine._allocation_weights_for_sleeves(
        SimpleNamespace(),
        [SimpleNamespace(name="day"), SimpleNamespace(name="swing"), SimpleNamespace(name="")],
    )
    assert weights == {"day": pytest.approx(1.0 / 7.0), "swing": pytest.approx(6.0 / 7.0)}

    frame = pd.DataFrame({"volume": [0, 100, -5, 300]})
    assert bot_engine._rolling_volume_from_bars(frame, 3) == 200.0
    assert bot_engine._rolling_volume_from_bars(pd.DataFrame({"close": [1.0]}), 5) == 0.0

    records = [
        {"symbol": "aapl", "status": "filled", "side": "buy", "is_bps": "2.0"},
        {"symbol": "AAPL", "status": "filled", "side": "sell", "is_bps": "4.0"},
        {"symbol": "MSFT", "status": "rejected", "side": "buy", "is_bps": "99.0"},
        {"symbol": "", "status": "filled", "side": "buy", "is_bps": "1.0"},
    ]
    metrics = bot_engine._build_symbol_learning_metrics(records, min_samples=2)
    assert metrics == {"AAPL": {"is_bps": 3.0, "flip_rate": 0.5, "trades": 2.0}}

    assert bot_engine._parse_iso_timestamp("2026-04-27T12:00:00Z") == datetime(
        2026,
        4,
        27,
        12,
        0,
        tzinfo=UTC,
    )
    assert bot_engine._parse_iso_timestamp("bad") is None
    assert bot_engine._replay_initial_positions_from_env() == {"AAPL": 2.0}
    assert bot_engine._parse_replay_initial_positions_payload({"msft": "-3", "flat": 0}) == {
        "MSFT": -3.0
    }

    rows = [
        {"symbol": "AAPL", "ts": "2026-04-27T12:05:00Z", "position_before": 9},
        {"symbol": "AAPL", "ts": "2026-04-27T12:00:00Z", "position_before": 4},
        {"symbol": "MSFT", "timestamp": "2026-04-27T12:01:00Z", "side": "sell"},
        {"symbol": "TSLA", "timestamp": "2026-04-27T12:02:00Z", "side": "buy"},
    ]
    assert bot_engine._replay_initial_positions_from_rows(rows) == {"AAPL": 4.0}
    assert bot_engine._replay_non_flat_start_symbols(rows, seeded_symbols={"TSLA"}) == {"MSFT"}


def test_gate_analytics_maps_merge_and_bucket_helpers() -> None:
    assert bot_engine._gate_name_is_warmup_noise(" warmup_data_only ") is True
    assert bot_engine._gate_name_is_warmup_noise("ENTRY_BLOCK") is False
    assert bot_engine._gate_name_is_halt_noise("auth_halt") is True
    assert bot_engine._gate_name_is_halt_noise("pre_submit") is False
    assert bot_engine._gate_root_cause("LIQ_PARTICIPATION_BLOCK_BYPASSED") == "LIQUIDITY_PARTICIPATION"
    assert bot_engine._gate_root_cause("alpha_decay_entry_guard") == "ALPHA_DECAY"
    assert bot_engine._gate_root_cause("") == ""
    assert bot_engine._dedupe_gate_root_causes(
        ["PRE_SUBMIT_A", "PRE_SUBMIT_B", "LIQ_TEST", "", "OTHER"]
    ) == ["PRE_SUBMIT_A", "LIQ_TEST", "OTHER"]

    assert bot_engine._analytics_counter_map({"a": "2", "b": 0, "c": "bad"}) == {"a": 2}
    assert bot_engine._analytics_counter_map([]) == {}
    stats = bot_engine._analytics_stats_map(
        {
            "root": {
                "count": "2",
                "accepted_records": 1,
                "blocked_records": "bad",
                "expected_net_edge_bps_sum": 3.5,
                "edge_proxy_bps_sum": "",
            },
            "skip": object(),
        }
    )
    assert stats == {
        "root": {
            "count": 2.0,
            "accepted_records": 1.0,
            "blocked_records": 0.0,
            "expected_net_edge_bps_sum": 3.5,
            "edge_proxy_bps_sum": 0.0,
        }
    }

    bot_engine._bump_analytics_stats(
        stats,
        "root",
        accepted=False,
        expected_net_edge_bps=1.5,
        edge_proxy_bps=-0.5,
    )
    assert stats["root"]["count"] == 3.0
    assert stats["root"]["blocked_records"] == 1.0
    assert stats["root"]["expected_net_edge_bps_sum"] == 5.0

    bot_engine._merge_analytics_stats(
        stats,
        {
            "root": {"count": 2, "accepted_records": 2, "edge_proxy_bps_sum": 1},
            "new": {"blocked_records": 4},
            "bad": object(),
        },
    )
    assert stats["root"]["count"] == 5.0
    assert stats["root"]["accepted_records"] == 3.0
    assert stats["root"]["edge_proxy_bps_sum"] == 0.5
    assert stats["new"]["blocked_records"] == 4.0
    assert bot_engine._counterfactual_bucket_key_from_observation(
        {"symbol": " aapl ", "session_bucket": " Opening "}
    ) == "AAPL:opening"
    assert bot_engine._counterfactual_bucket_key_from_observation({}) == "UNKNOWN:offhours"


def test_jsonl_timestamp_and_replay_metric_helpers(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(
            [
                "",
                "{bad json",
                json.dumps({"ts": "2026-04-27T12:00:00Z", "value": 1}),
                json.dumps(
                    {
                        "benchmark": {
                            "submit_ts": "2026-04-27T12:05:00Z",
                            "first_fill_ts": "bad",
                        }
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    assert bot_engine._read_jsonl_records(str(tmp_path / "missing.jsonl")) == []
    assert bot_engine._read_jsonl_records(str(path), max_records=1) == [
        {"benchmark": {"submit_ts": "2026-04-27T12:05:00Z", "first_fill_ts": "bad"}}
    ]
    assert bot_engine._latest_tca_timestamp(str(path)) == datetime(
        2026,
        4,
        27,
        12,
        5,
        tzinfo=UTC,
    )

    result = {
        "orders": [
            {"id": "o1", "client_order_id": "c1", "symbol": "AAPL", "side": "buy", "limit_price": 100},
            {"id": "o2", "client_order_id": "c2", "symbol": "MSFT", "side": "sell", "price": 50},
            {"id": "", "symbol": "IGNORED"},
            object(),
        ],
        "events": [
            {
                "event_type": "fill",
                "order_id": "o1",
                "client_order_id": "c1",
                "fill_price": 99,
                "ts": "2026-04-27T13:40:00Z",
            },
            {
                "event_type": "fill",
                "order_id": "o2",
                "client_order_id": "c2",
                "fill_price": 51,
                "ts": "2026-04-27T20:00:00Z",
            },
            {"event_type": "fill", "order_id": "o1", "fill_price": -1},
            {"event_type": "cancel", "order_id": "o1", "fill_price": 101},
            object(),
        ],
    }

    summary = bot_engine._replay_summary_metrics(result)
    assert summary["sample_count"] == 2
    assert summary["net_edge_bps"] == pytest.approx(150.0)
    assert summary["max_drawdown_pct"] == 0.0
    assert bot_engine._replay_summary_metrics({"orders": [], "events": []}) == {
        "sample_count": 0,
        "net_edge_bps": 0.0,
        "max_drawdown_pct": 0.0,
    }

    metric_summary = bot_engine._replay_metric_summary(
        {"wins": [1.0, 2.0], "mixed": [4.0, -2.0], "empty": [float("nan")]}
    )
    assert metric_summary["wins"]["profit_factor"] == 0.0
    assert metric_summary["mixed"]["profit_factor"] == 2.0
    assert "empty" not in metric_summary

    symbol_summary = bot_engine._replay_symbol_summary_metrics(result)
    assert symbol_summary["AAPL"]["net_edge_bps"] == pytest.approx(100.0)
    assert symbol_summary["MSFT"]["net_edge_bps"] == pytest.approx(200.0)

    bucket_summary = bot_engine._replay_bucket_summary_metrics(
        result,
        order_context_by_client_id={"c1": {"session_token": "manual", "regime_token": "trend"}},
    )
    assert bucket_summary["by_symbol_session"]["AAPL:manual"]["sample_count"] == 1.0
    assert bucket_summary["by_symbol_session_regime"]["AAPL:manual:trend"]["net_edge_bps"] == pytest.approx(100.0)
    assert bucket_summary["by_symbol_session"]["MSFT:closing"]["net_edge_bps"] == pytest.approx(200.0)


def test_schedule_threshold_and_latest_replay_summary_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    now = datetime(2026, 4, 27, 20, 30, tzinfo=UTC)
    state = SimpleNamespace(
        last_learning_run_date=None,
        last_allocation_update_date=now.date(),
        last_replay_run_date=None,
        last_walk_forward_run_date=now.date(),
        last_runtime_truth_report_date=None,
    )

    _patch_get_env(
        monkeypatch,
        {
            "AI_TRADING_POST_TRADE_LEARNING_ENABLED": True,
            "AI_TRADING_LEARNING_RUN_SCHEDULE": "market_close",
            "AI_TRADING_ALLOCATION_ENABLED": True,
            "AI_TRADING_ALLOCATION_UPDATE_SCHEDULE": "daily_first_run",
            "AI_TRADING_REPLAY_ENABLED": True,
            "AI_TRADING_WALK_FORWARD_ENABLED": True,
            "AI_TRADING_RUNTIME_TRUTH_REPORT_ENABLED": True,
            "AI_TRADING_RUNTIME_GONOGO_TRADE_FILL_SOURCE": " broker ",
            "AI_TRADING_RUNTIME_GONOGO_MIN_CLOSED_TRADES": 7,
            "AI_TRADING_RUNTIME_GONOGO_MIN_PROFIT_FACTOR": 1.4,
            "AI_TRADING_RUNTIME_GONOGO_PROFIT_FACTOR_MIN_LOSSES": -3,
            "AI_TRADING_RUNTIME_GONOGO_PROFIT_FACTOR_MIN_GROSS_LOSS_PNL": -10.0,
            "AI_TRADING_RUNTIME_GONOGO_MIN_WIN_RATE": 0.55,
            "AI_TRADING_RUNTIME_GONOGO_MIN_NET_PNL": 12.0,
            "AI_TRADING_RUNTIME_GONOGO_MIN_ACCEPTANCE_RATE": 0.08,
            "AI_TRADING_RUNTIME_GONOGO_MIN_EXPECTED_NET_EDGE_BPS": -5.0,
            "AI_TRADING_RUNTIME_GONOGO_MIN_USED_DAYS": -1,
            "AI_TRADING_RUNTIME_GONOGO_LOOKBACK_DAYS": 14,
            "AI_TRADING_RUNTIME_GONOGO_REQUIRE_PNL_AVAILABLE": False,
            "AI_TRADING_RUNTIME_GONOGO_REQUIRE_GATE_VALID": True,
            "AI_TRADING_RUNTIME_GONOGO_REQUIRE_REPLAY_LIVE_PARITY_GATE": False,
        },
    )

    assert bot_engine._post_trade_learning_schedule_due(
        state,
        now=now,
        market_open_now=False,
    ) is True
    state.last_learning_run_date = now.date()
    assert bot_engine._post_trade_learning_schedule_due(
        state,
        now=now,
        market_open_now=False,
    ) is False
    assert bot_engine._allocation_schedule_due(state, now=now, market_open_now=True) is False
    state.last_allocation_update_date = None
    assert bot_engine._allocation_schedule_due(state, now=now, market_open_now=True) is True
    assert bot_engine._replay_schedule_due(state, now=now, market_open_now=False) is True
    assert bot_engine._replay_schedule_due(state, now=now, market_open_now=True) is False
    assert bot_engine._walk_forward_schedule_due(state, now=now, market_open_now=False) is False
    assert bot_engine._runtime_truth_report_schedule_due(
        state,
        now=now,
        market_open_now=False,
    ) is True

    thresholds = bot_engine._runtime_truth_report_thresholds()
    assert thresholds["min_closed_trades"] == 7
    assert thresholds["profit_factor_min_losses"] == 0
    assert thresholds["profit_factor_min_gross_loss_pnl"] == 0.0
    assert thresholds["min_used_days"] == 0
    assert thresholds["lookback_days"] == 14
    assert thresholds["trade_fill_source"] == "broker"
    assert thresholds["require_pnl_available"] is False
    assert thresholds["require_gate_valid"] is True
    assert thresholds["require_replay_live_parity_gate"] is False

    before_path = tmp_path / "replay_hash_current.json"
    before_path.write_text(json.dumps({"replay_summary": {"sample_count": 99}}), encoding="utf-8")
    stale_path = tmp_path / "replay_hash_stale.json"
    stale_path.write_text("{bad json", encoding="utf-8")
    older_path = tmp_path / "replay_hash_older.json"
    older_path.write_text(
        json.dumps(
            {
                "replay_summary": {
                    "sample_count": 3,
                    "net_edge_bps": "4.5",
                    "max_drawdown_pct": "0.01",
                }
            }
        ),
        encoding="utf-8",
    )
    assert bot_engine._load_latest_replay_summary(tmp_path, before_path=before_path) == {
        "sample_count": 3,
        "net_edge_bps": 4.5,
        "max_drawdown_pct": 0.01,
    }

    assert bot_engine._load_latest_replay_symbol_summary(
        tmp_path,
        max_age_hours=0,
    ) == ({}, {"reason": "disabled"})

    quality_path = tmp_path / "replay_hash_quality.json"
    quality_path.write_text(
        json.dumps(
            {
                "ts": datetime.now(UTC).isoformat(),
                "replay_symbol_summary": {
                    "AAPL": {
                        "sample_count": 2,
                        "net_edge_bps": "1.5",
                        "win_rate": "0.5",
                        "profit_factor": "2.0",
                    },
                    "BAD": {"sample_count": 0, "net_edge_bps": 99},
                },
                "replay_bucket_summary": {
                    "by_symbol_session": {
                        "AAPL:opening": {"sample_count": 1, "net_edge_bps": 3.0}
                    },
                    "by_symbol_session_regime": {
                        "AAPL:opening:trend": {"sample_count": 1, "net_edge_bps": 4.0}
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    symbol_summary, session_summary, regime_summary, context = (
        bot_engine._load_latest_replay_quality_summaries(tmp_path, max_age_hours=1)
    )
    assert symbol_summary == {
        "AAPL": {
            "sample_count": 2.0,
            "net_edge_bps": 1.5,
            "win_rate": 0.5,
            "profit_factor": 2.0,
        }
    }
    assert session_summary["AAPL:opening"]["net_edge_bps"] == 3.0
    assert regime_summary["AAPL:opening:trend"]["net_edge_bps"] == 4.0
    assert context["symbols"] == 1


def test_screen_env_prefilter_and_market_data_quality_helpers(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_get_env(
        monkeypatch,
        {
            "POSITIVE": 4,
            "ZERO": 0,
            "BAD": ValueError("bad env"),
            "FLOAT_LOW": -2.5,
            "FLAG": True,
        },
    )
    assert bot_engine._safe_env_int("POSITIVE", 1) == 4
    assert bot_engine._safe_env_int("ZERO", 9) == 9
    assert bot_engine._safe_env_non_negative_int("ZERO", 9) == 0
    assert bot_engine._safe_env_float("FLOAT_LOW", 1.0, min_value=0.25) == 0.25
    assert bot_engine._safe_env_bool("FLAG", False) is True
    assert bot_engine._safe_env_bool("BAD", True) is True

    monkeypatch.setattr(bot_engine, "_SCREEN_TOPN", 2)
    monkeypatch.setattr(bot_engine, "_SCREEN_BATCH_SIZE", 2)
    monkeypatch.setattr(bot_engine, "_SCREEN_MIN_REFETCH_SEC", 60)
    bot_engine._LAST_SCREEN_FETCH.clear()

    assert bot_engine._should_refetch_screen_symbol("AAPL", now_ts=100.0) is True
    assert bot_engine._should_refetch_screen_symbol("AAPL", now_ts=120.0) is False
    assert bot_engine._should_refetch_screen_symbol("AAPL", now_ts=161.0) is True
    filtered, throttled = bot_engine._prefilter_screen_symbols([" aapl ", "MSFT", "MSFT", "", "TSLA"])
    assert filtered == ["AAPL", "MSFT"]
    assert throttled == ["TSLA"]
    assert list(bot_engine._iter_screen_batches(["A", "B", "C"])) == [["A", "B"], ["C"]]

    assert bot_engine._validate_market_data_quality(None, "AAPL")["reason"] == "no_data"
    short = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [100]})
    assert bot_engine._validate_market_data_quality(short, "AAPL")["reason"].startswith(
        "insufficient_data_"
    )
    missing = pd.DataFrame({"close": [1.0] * 25})
    assert bot_engine._validate_market_data_quality(missing, "AAPL")["reason"] == "missing_columns"

    base = pd.DataFrame(
        {
            "open": [10.0] * 25,
            "high": [11.0] * 25,
            "low": [9.0] * 25,
            "close": [10.0] * 25,
            "volume": [20_000] * 25,
        }
    )
    excessive_nan = base.copy()
    excessive_nan.loc[:4, "open"] = None
    assert (
        bot_engine._validate_market_data_quality(excessive_nan, "AAPL")["reason"]
        == "excessive_nan_open"
    )
    invalid_price = base.copy()
    invalid_price.loc[5, "close"] = 0.0
    assert bot_engine._validate_market_data_quality(invalid_price, "AAPL")["reason"] == "invalid_prices"

    low_liquidity = base.copy()
    low_liquidity["volume"] = [9_999] * 25
    assert bot_engine._validate_market_data_quality(low_liquidity, "AAPL")["reason"] == "low_liquidity"

    zero_volume = base.copy()
    zero_volume.loc[:6, "volume"] = 0
    assert (
        bot_engine._validate_market_data_quality(zero_volume, "AAPL")["reason"]
        == "excessive_zero_volume"
    )

    volatile = base.copy()
    volatile["close"] = [10.0, 20.0, 9.0, 19.0, 8.0] * 5
    with caplog.at_level(logging.WARNING):
        valid = bot_engine._validate_market_data_quality(volatile, "AAPL")
    assert valid["valid"] is True
    assert any(record.getMessage() == "DATA_QUALITY_EXTREME_VOLATILITY" for record in caplog.records)

    passed = bot_engine._validate_market_data_quality(base, "AAPL")
    assert passed["valid"] is True
    assert passed["reason"] == "passed_validation"
