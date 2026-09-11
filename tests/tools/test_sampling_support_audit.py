from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from ai_trading.tools.research_feasibility import audit_sampling_acquisition, sampling_support
from ai_trading.utils.market_calendar import is_trading_day, session_info


def protocol():
    return {"lookback_sessions": 60, "holding_sessions": 5, "development_start": "2024-01-01", "development_end": "2025-12-31", "holdout_start": "2026-09-09", "holdout_end": "2026-12-08", "symbols": ["SPY"], "feed": "sip", "adjustment": "split"}


def sessions(contract):
    parts = []
    for day in pd.date_range(contract["development_start"], contract["development_end"]):
        if is_trading_day(day.date()):
            session = session_info(day.date())
            parts.append(pd.date_range(session.start_utc, session.end_utc, freq="min", inclusive="left"))
    return parts


def test_sparse_gaps_collapse_support_without_reading_prices():
    contract = protocol()
    contract["symbols"] = ["A", "B"]
    parts = sessions(contract)
    index = parts[0].append(parts[1:])
    full = sampling_support(contract, {"A": index, "B": index})
    assert full["status"] == "support_possible_not_strategy_eligible"
    # A single missing minute every 30 sessions prevents any 67-session window.
    sparse = index.difference(pd.DatetimeIndex([part[1] for part in parts[::30]]))
    result = sampling_support(contract, {"A": index, "B": sparse})
    assert result["possible_common_periods"] == 0
    assert result["symbols"]["B"]["missing_ratio"] < .001
    assert result["status"] == "blocked_sampling_support"
    assert result["strategy_outcomes_computed"] is False
    assert result["trial_claimed"] is False


def test_duplicate_or_missing_symbols_cannot_pass():
    contract = protocol()
    parts = sessions(contract)
    index = parts[0].append(parts[1:])
    result = sampling_support(contract, {"SPY": index.append(index[:1])})
    assert result["status"] == "blocked_sampling_support"
    assert result["invalid_inputs"] == ["invalid_timestamp_grain:SPY"]
    assert sampling_support(contract, {})["possible_common_periods"] == 0


def acquisition(tmp_path, contract, raw):
    csv = tmp_path / "SPY.csv"
    csv.write_text(raw)
    manifest = {"quality_passed": True, "dataset_identity": {"request_interval": {"start_date": contract["development_start"], "end_date": contract["development_end"]}, "symbols": ["SPY"], "feed": "sip", "adjustment": "split"}, "symbols": [{"symbol": "SPY", "csv_path": str(csv), "content_sha256": hashlib.sha256(csv.read_bytes()).hexdigest()}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    pointer = tmp_path / "acquisition.json"
    pointer.write_text(json.dumps({"manifest_path": str(path), "quality_passed": True}))
    return pointer, csv


def test_audit_uses_timestamp_column_only_and_checks_hash(tmp_path, monkeypatch):
    contract = protocol()
    pointer, csv = acquisition(tmp_path, contract, "timestamp,close\n2024-01-02T14:30:00Z,not-a-price\n")
    original = pd.read_csv
    def read(*args, **kwargs):
        assert kwargs["usecols"] == ["timestamp"]
        return original(*args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", read)
    report = audit_sampling_acquisition(contract, pointer)
    assert report["prices_read_for_analysis"] is False
    assert report["sources"]["SPY"]["sha256"]
    csv.write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_sampling_acquisition(contract, pointer)


def test_holdout_rejected_before_source_read(tmp_path, monkeypatch):
    contract = protocol()
    other = {**contract, "development_start": "2026-09-09", "development_end": "2026-12-08"}
    pointer, _ = acquisition(tmp_path, other, "timestamp\n2026-09-09T13:30:00Z\n")
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: pytest.fail("holdout source must not be read"))
    with pytest.raises(ValueError, match="frozen development"):
        audit_sampling_acquisition(contract, pointer)


def test_naive_source_timestamps_are_not_assumed_utc(tmp_path):
    contract = protocol()
    pointer, _ = acquisition(tmp_path, contract, "timestamp\n2024-01-02T14:30:00\n")
    with pytest.raises(ValueError, match="timezone missing"):
        audit_sampling_acquisition(contract, pointer)
