from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ai_trading.models.contracts import DAY_SLEEVE_ML_FEATURE_COLUMNS
from ai_trading.tools import model_replacement_trial as trial
from ai_trading.tools.research_feasibility import audit_sampling_acquisition, intraday_support, sampling_support
from ai_trading.utils.market_calendar import is_trading_day, session_info


def protocol(**changes):
    contract = {"development_start": "2024-01-01", "development_end": "2024-12-31", "holdout_start": "2026-09-09", "holdout_end": "2026-12-08", "symbols": ["A", "B", "C"], "feed": "sip", "adjustment": "split", "exit_horizon_minutes": 60, "flatten_cutoff_et": "15:55"}
    contract.update(changes)
    return contract


def minutes(contract):
    parts = []
    for day in pd.date_range(contract["development_start"], contract["development_end"]):
        if is_trading_day(day.date()):
            session = session_info(day.date())
            parts.append(pd.date_range(session.start_utc, session.end_utc, freq="min", inclusive="left"))
    return parts[0].append(parts[1:])


def test_sixty_minute_grid_counts_folds_from_timestamps_only():
    contract = protocol()
    index = minutes(contract)
    report = intraday_support(contract, {s: index for s in contract["symbols"]})
    assert report["status"] == "support_possible_not_strategy_eligible"
    assert report["audit"] == "intraday_grid_timestamp_only"
    assert len(report["folds"]) == 5
    assert all(f["possible_test_opportunities"] >= 250 and f["possible_train_opportunities"] > 0 for f in report["folds"])
    per_symbol = report["symbols"]["A"]["possible_opportunities"]
    # Six hourly entries per full session; early closes and warm-up reduce the total.
    assert 6 * 245 < per_symbol < 6 * 252
    assert report["possible_opportunities"] == 3 * per_symbol
    assert report["prices_read_for_analysis"] is False and report["trial_claimed"] is False
    tested = sum(f["possible_test_opportunities"] for f in report["folds"])
    assert tested == sum(sum(f["test_opportunities_by_symbol"].values()) for f in report["folds"])


def test_audit_counts_match_trial_labels(monkeypatch):
    monkeypatch.setattr(trial, "build_day_sleeve_features", lambda window: pd.DataFrame([{c: 1.0 for c in DAY_SLEEVE_ML_FEATURE_COLUMNS}]))
    contract = protocol(development_end="2024-01-31", symbols=["A"])
    days = [str(d.date()) for d in pd.date_range("2024-01-01", "2024-01-31") if is_trading_day(d.date())]
    index = minutes(contract)
    frame = pd.DataFrame({"open": np.full(len(index), 100.0), "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000}, index=index)
    for horizon, cutoff in ((60, "15:55"), (5, None)):
        grid = {"exit_horizon_minutes": horizon, "flatten_cutoff_et": cutoff}
        grid = {k: v for k, v in grid.items() if v is not None}
        audit_contract = {k: v for k, v in contract.items() if k != "flatten_cutoff_et"} | grid
        rows = trial.labeled_rows(frame, trial.regular_bars(frame, days), symbol="A", spec=trial.TrialSpec(grid=trial.IntradayGrid.from_protocol(audit_contract)))
        report = intraday_support(audit_contract, {"A": index})
        assert report["symbols"]["A"]["possible_opportunities"] == len(rows)


def test_incomplete_sessions_or_bad_inputs_block():
    contract = protocol()
    index = minutes(contract)
    gapped = index.delete(1000)
    report = intraday_support(contract, {"A": index, "B": gapped, "C": index})
    assert report["status"] == "blocked_sampling_support"
    assert report["invalid_inputs"] == ["incomplete_sessions:B"]
    assert report["symbols"]["B"]["incomplete_sessions"] == 1
    assert intraday_support(contract, {"A": index, "B": index})["invalid_inputs"] == ["symbol_missing:C"]
    naive = intraday_support(contract, {s: index.tz_localize(None) for s in contract["symbols"]})
    assert naive["status"] == "blocked_sampling_support"
    short = intraday_support(protocol(development_end="2024-01-31"), {s: minutes(protocol(development_end="2024-01-31")) for s in "ABC"})
    assert short["status"] == "blocked_sampling_support" and short["folds"] == []


def test_holdout_overlap_and_bad_grid_rejected():
    with pytest.raises(ValueError, match="holdout"):
        intraday_support(protocol(development_end="2026-09-10"), {})
    with pytest.raises(ValueError, match="flatten_cutoff_et"):
        intraday_support(protocol(flatten_cutoff_et="3:55pm"), {})
    with pytest.raises(ValueError, match="exit_horizon_minutes"):
        intraday_support(protocol(exit_horizon_minutes=True), {})
    session_protocol = {k: v for k, v in protocol().items() if k != "exit_horizon_minutes"}
    with pytest.raises(ValueError, match="unsupported sampling contract"):
        sampling_support(session_protocol, {})


def acquisition(tmp_path, contract, raw):
    paths = {}
    for symbol in contract["symbols"]:
        paths[symbol] = tmp_path / f"{symbol}.csv"
        paths[symbol].write_text(raw)
    manifest = {"quality_passed": True, "dataset_identity": {"request_interval": {"start_date": contract["development_start"], "end_date": contract["development_end"]}, "symbols": contract["symbols"], "feed": "sip", "adjustment": "split"}, "symbols": [{"symbol": s, "csv_path": str(p), "content_sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for s, p in paths.items()]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    pointer = tmp_path / "acquisition.json"
    pointer.write_text(json.dumps({"manifest_path": str(tmp_path / "manifest.json"), "quality_passed": True}))
    return pointer


def test_acquisition_audit_selects_intraday_grid_and_reads_timestamps_only(tmp_path, monkeypatch):
    contract = protocol(symbols=["A"])
    pointer = acquisition(tmp_path, contract, "timestamp,open\n2024-01-02T14:30:00Z,not-a-price\n")
    original = pd.read_csv

    def read(*args, **kwargs):
        assert kwargs["usecols"] == ["timestamp"]
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read)
    report = audit_sampling_acquisition(contract, pointer)
    assert report["audit"] == "intraday_grid_timestamp_only"
    assert report["status"] == "blocked_sampling_support"
    assert report["sources"]["A"]["columns_used"] == ["timestamp"]


def test_malformed_grid_rejected_before_source_read(tmp_path, monkeypatch):
    contract = protocol(symbols=["A"], exit_horizon_minutes=-5)
    pointer = acquisition(tmp_path, contract, "timestamp\n2024-01-02T14:30:00Z\n")
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: pytest.fail("source must not be read"))
    with pytest.raises(ValueError, match="exit_horizon_minutes"):
        audit_sampling_acquisition(contract, pointer)
