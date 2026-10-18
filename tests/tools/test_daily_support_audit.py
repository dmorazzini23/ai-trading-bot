from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ai_trading.tools.research_feasibility import (
    DailyGrid, daily_rebalance_support, regular_session_minutes, strategy_bucket,
)
from ai_trading.utils.market_calendar import is_trading_day

REPO_CONTRACT = Path(__file__).resolve().parents[2] / "config/cross_sectional_momentum_trial_campaign.json"


def protocol(**changes):
    contract = json.loads(REPO_CONTRACT.read_text())
    contract.update(changes)
    return contract


def full_index(contract):
    dates = [d.date() for d in pd.date_range(contract["development_start"], contract["development_end"]) if is_trading_day(d.date())]
    parts = regular_session_minutes(dates)
    return parts[0].append(parts[1:]), parts


def test_contract_grid_counts_periods_and_selections_per_fold_from_timestamps():
    contract = protocol()
    index, _ = full_index(contract)
    report = daily_rebalance_support(contract, {s: index for s in contract["symbols"]})
    assert report["status"] == "support_possible_not_strategy_eligible"
    assert report["development_sessions"] == 502
    # range(63, 502 - 1 - 5, 5): decisions at sessions 63, 68, ..., 493.
    assert report["candidate_periods"] == report["possible_periods"] == 87
    assert report["universe"]["buy_selections_per_period"] == 1
    assert report["possible_selections"] == 87
    assert [(f["fold"], f["possible_selections"]) for f in report["folds"]] == [
        ("2024Q2", 12), ("2024Q3", 13), ("2024Q4", 13), ("2025Q1", 12), ("2025Q2", 12), ("2025Q3", 13), ("2025Q4", 12)]
    assert report["supported_folds"] == 7
    assert report["grid"]["required_complete_sessions_per_period"] == 63 + 5 + 2
    assert report["prices_read_for_analysis"] is False and report["trial_claimed"] is False


def test_threshold_arithmetic_blocks_at_each_boundary():
    contract = protocol()
    index, _ = full_index(contract)
    indices = {s: index for s in contract["symbols"]}

    def run(**thresholds):
        return daily_rebalance_support(protocol(support_thresholds=contract["support_thresholds"] | thresholds), indices)

    assert run(minimum_decision_periods=87)["status"] == "support_possible_not_strategy_eligible"
    assert run(minimum_decision_periods=88)["blocking_reasons"] == ["decision_periods_below_minimum"]
    assert run(minimum_selections=88)["blocking_reasons"] == ["selections_below_minimum"]
    # Only the three 13-period quarters reach 13 selections.
    thirteen = run(minimum_selections_per_fold=13)
    assert thirteen["supported_folds"] == 3 and thirteen["blocking_reasons"] == ["supported_folds_below_minimum"]
    assert run(minimum_supported_folds=7)["status"] == "support_possible_not_strategy_eligible"
    assert run(minimum_supported_folds=8)["blocking_reasons"] == ["supported_folds_below_minimum"]


def test_bucket_mirrors_strategy_quantile_and_min_universe():
    assert strategy_bucket(protocol()) == 1
    ten = protocol(symbols=[f"S{i}" for i in range(10)])
    assert strategy_bucket(ten) == 2
    index, _ = full_index(ten)
    assert daily_rebalance_support(ten, {s: index for s in ten["symbols"]})["possible_selections"] == 174
    stocks = protocol(symbols=["AAPL", "AMZN", "MSFT"])
    report = daily_rebalance_support(stocks, {s: index for s in stocks["symbols"]})
    assert strategy_bucket(stocks) == 0
    assert report["possible_periods"] == 87 and report["possible_selections"] == 0
    assert "universe_below_strategy_min_universe" in report["blocking_reasons"]


def test_middle_minute_gap_blocks_only_every_minute_rule_but_open_gap_blocks_both():
    contract = protocol(development_end="2024-06-28", support_thresholds={"minimum_decision_periods": 1, "minimum_selections": 1, "minimum_selections_per_fold": 1, "minimum_supported_folds": 1})
    index, parts = full_index(contract)
    session = parts[70]  # outside the first window (sessions 0..69) but inside every later one
    middle_gap = {s: index for s in contract["symbols"]} | {"DIA": index.difference(session[100:104])}
    report = daily_rebalance_support(contract, middle_gap)
    sessions = report["development_sessions"]
    assert report["symbols"]["DIA"]["complete_sessions_by_rule"] == {"every_minute": sessions - 1, "open_and_close_minutes": sessions}
    assert report["by_completeness_rule"]["every_minute"]["possible_periods"] == 1
    assert report["by_completeness_rule"]["every_minute"]["possible_periods"] < report["possible_periods"]
    assert report["possible_periods"] == report["candidate_periods"]
    every = daily_rebalance_support(protocol(**{**contract, "session_completeness": {"rule": "every_minute"}}), middle_gap)
    assert every["symbols"]["DIA"]["incomplete_sessions"] == 1
    open_gap = middle_gap | {"DIA": index.difference(session[:1])}
    assert daily_rebalance_support(contract, open_gap)["possible_periods"] < report["possible_periods"]


def test_holdout_2026_and_out_of_window_timestamps_refused():
    with pytest.raises(ValueError, match="holdout"):
        daily_rebalance_support(protocol(development_end="2026-09-10"), {})
    with pytest.raises(ValueError, match="2026"):
        daily_rebalance_support(protocol(development_end="2026-01-05"), {})
    contract = protocol(development_end="2024-06-28")
    index, _ = full_index(contract)
    late = index.append(pd.DatetimeIndex([pd.Timestamp("2026-01-02 15:00", tz="UTC")]))
    report = daily_rebalance_support(contract, {s: index for s in contract["symbols"]} | {"SPY": late})
    assert report["status"] == "blocked_sampling_support"
    assert "timestamps_outside_development:SPY" in report["invalid_inputs"]


def test_grid_validation():
    assert DailyGrid.from_protocol(protocol()) == DailyGrid(63, 5, "open_and_close_minutes", 0.0)
    with pytest.raises(ValueError, match="positive"):
        DailyGrid.from_protocol(protocol(rebalance_grid={"lookback_sessions": 63, "hold_sessions": 0}))
    with pytest.raises(ValueError, match="completeness"):
        DailyGrid.from_protocol(protocol(session_completeness={"rule": "close_only"}))
    with pytest.raises(ValueError, match="full coverage"):
        DailyGrid.from_protocol(protocol(session_completeness={"rule": "every_minute", "minimum_minute_coverage": 0.9}))
    schedule = DailyGrid(63, 5).schedule(502)
    assert all(nxt[1] == prev[2] for prev, nxt in zip(schedule, schedule[1:]))
    assert schedule[-1][2] <= 501
