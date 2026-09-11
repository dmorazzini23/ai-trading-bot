from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_trading.tools.endpoint_feasibility import audit, check_required_bars, observation_grid


@pytest.fixture
def proposal():
    value = json.loads((Path(__file__).resolve().parents[2] / "config/endpoint_data_proposal.json").read_text())
    return {**value, "symbols": ["SPY", "QQQ"]}


def source_frame(grid):
    required = sorted({ts for row in grid for ts, _ in row["observations"].values()})
    return pd.DataFrame({"open": 100., "high": 102., "low": 99., "close": 101.}, index=pd.DatetimeIndex(required))


def acquisition(tmp_path, proposal):
    frame = source_frame(observation_grid(proposal))
    entries = []
    for symbol in proposal["symbols"]:
        path = tmp_path / f"{symbol}.csv"
        frame.to_csv(path, index_label="timestamp")
        entries.append({"symbol": symbol, "csv_path": str(path), "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"quality_passed": True, "dataset_identity": {"request_interval": {"start_date": proposal["development_start"], "end_date": proposal["development_end"]}, "symbols": proposal["symbols"], "feed": "sip", "adjustment": "split", "timeframe": "1Min"}, "symbols": entries}
    meta = tmp_path / "manifest.json"
    meta.write_text(json.dumps(manifest))
    pointer = tmp_path / "acquisition.json"
    pointer.write_text(json.dumps({"quality_passed": True, "manifest_path": str(meta)}))
    return pointer, meta


def test_exact_required_bars_suffice_without_interior_minutes(proposal):
    grid = observation_grid(proposal)
    frame = source_frame(grid)
    result = check_required_bars(frame, grid)
    assert result["possible_periods"] == len(grid) == 88
    assert result["rejection_counts"] == {}
    assert grid[0]["observations"]["previous_close"][0] < grid[0]["observations"]["entry_open"][0]
    assert grid[0]["observations"]["exit_open"][0] == grid[1]["observations"]["entry_open"][0]


@pytest.mark.parametrize("problem", ["missing", "duplicate", "nan", "negative", "incoherent"])
def test_bad_required_observation_excludes_period(proposal, problem):
    grid = observation_grid(proposal)
    frame = source_frame(grid)
    ts = grid[0]["observations"]["previous_close"][0]
    if problem == "missing":
        frame = frame.drop(ts)
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.loc[[ts]]])
    else:
        frame.loc[ts, "close"] = {"nan": np.nan, "negative": -1, "incoherent": 500}[problem]
    result = check_required_bars(frame, grid)
    assert grid[0]["session"] not in result["valid_entry_dates"]
    assert result["rejection_counts"]


def test_valid_price_changes_do_not_select_a_signal_or_change_support(proposal):
    grid = observation_grid(proposal)
    frame = source_frame(grid)
    expected = check_required_bars(frame, grid)
    frame.loc[::2, "close"] = 50
    frame["low"] = 40
    assert check_required_bars(frame, grid) == expected


def test_passed_support_does_not_approve_adjustments_or_registration(tmp_path, proposal):
    pointer, _ = acquisition(tmp_path, proposal)
    result = audit(proposal, pointer)
    assert result["observation_feasibility"] == "passes_upper_bound_screen"
    assert result["common_valid_periods"] == 88
    assert result["maximum_possible_selections"] == 176
    assert result["actual_selections"] is None
    assert result["recommendation"] == "stop_before_registration"
    assert "proposed_adjustment_contract_not_met" in result["blockers"]
    assert result["signals_computed"] is result["returns_computed"] is result["trial_claimed"] is False
    assert result["baselines"]["common_mask_shared"] is True


def test_source_hash_mismatch_blocks_analysis(tmp_path, proposal):
    pointer, _ = acquisition(tmp_path, proposal)
    (tmp_path / "SPY.csv").write_text("modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit(proposal, pointer)


def test_holdout_metadata_rejected_before_reading_prices(tmp_path, proposal, monkeypatch):
    pointer, meta = acquisition(tmp_path, proposal)
    manifest = json.loads(meta.read_text())
    manifest["dataset_identity"]["request_interval"] = {"start_date": "2026-09-09", "end_date": "2026-12-08"}
    meta.write_text(json.dumps(manifest))
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: pytest.fail("must not read holdout prices"))
    with pytest.raises(ValueError, match="development audit source"):
        audit(proposal, pointer)
