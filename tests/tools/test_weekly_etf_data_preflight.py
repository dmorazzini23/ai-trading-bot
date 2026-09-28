from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from ai_trading.tools.weekly_etf_data_preflight import (
    SYMBOLS,
    audit,
    development_actions,
    weekly_observation_grid,
)
from ai_trading.utils.market_calendar import is_trading_day, session_info


def _days() -> list[date]:
    return [day.date() for day in pd.date_range("2024-01-01", "2025-12-31") if is_trading_day(day.date())]


def _capture() -> dict:
    return {
        "status": "captured",
        "pagination_complete": True,
        "request": {"symbols": ",".join(SYMBOLS), "start": "2023-12-01", "end": "2026-03-31", "data_quality": "complete"},
        "pages": [{"next_page_token": None, "corporate_actions": {
            "cash_dividends": [{"id": "dividend-1", "symbol": "DIA", "ex_date": "2024-03-01", "process_date": "2024-03-11", "rate": 0.5}],
            "forward_splits": [{"id": "split-1", "symbol": "XLE", "ex_date": "2025-12-05", "process_date": "2025-12-05", "old_rate": 1, "new_rate": 2}],
        }}],
    }


def _acquisition(tmp_path: Path, days: list[date]) -> Path:
    required = {timestamp for row in weekly_observation_grid(days) for timestamp, _ in row["observations"].values()}
    for ex_day in (date(2024, 3, 1), date(2025, 12, 5)):
        index = days.index(ex_day)
        previous, current = session_info(days[index - 1]), session_info(ex_day)
        required.update({pd.Timestamp(previous.end_utc) - pd.Timedelta(minutes=1), pd.Timestamp(current.start_utc), pd.Timestamp(current.end_utc) - pd.Timedelta(minutes=1)})
    frame = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0}, index=pd.DatetimeIndex(sorted(required)))
    rows = []
    for symbol in SYMBOLS:
        path = tmp_path / f"{symbol}.csv"
        frame.to_csv(path, index_label="timestamp")
        rows.append({"symbol": symbol, "csv_path": str(path), "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = tmp_path / "dataset.provenance.json"
    manifest.write_text(json.dumps({"quality_passed": True, "dataset_identity": {
        "request_interval": {"start_date": "2024-01-01", "end_date": "2025-12-31"},
        "symbols": SYMBOLS, "feed": "sip", "adjustment": "all", "timeframe": "1Min",
    }, "symbols": rows}))
    pointer = tmp_path / "acquisition.json"
    pointer.write_text(json.dumps({"quality_passed": True, "manifest_path": str(manifest)}))
    return pointer


def test_friday_grid_needs_every_20_session_close_and_real_entry_exit_opens():
    days = _days()
    grid = weekly_observation_grid(days)
    assert len(grid) == 96
    assert all(date.fromisoformat(row["session"]).weekday() == 4 for row in grid)
    first = grid[0]
    signal_index = days.index(date.fromisoformat(first["session"]))
    assert len(first["observations"]) == 23
    assert first["observations"]["close_20"][0] == pd.Timestamp(session_info(days[signal_index - 20]).end_utc) - pd.Timedelta(minutes=1)
    assert first["observations"]["entry_open"][0] == pd.Timestamp(session_info(days[signal_index + 1]).start_utc)
    assert first["observations"]["exit_open"][0] == pd.Timestamp(session_info(days[signal_index + 6]).start_utc)


@pytest.mark.parametrize("change,reason", [
    ({"pagination_complete": False}, "incomplete_or_wrong_scope"),
    ({"pages": [{"next_page_token": "more", "corporate_actions": {}}]}, "pagination_unverified"),
])
def test_incomplete_action_capture_fails_closed(change, reason):
    capture = {**_capture(), **change}
    with pytest.raises(ValueError, match=reason):
        development_actions(capture, _days())


def test_action_ids_and_ex_dates_are_validated():
    capture = _capture()
    actions = development_actions(capture, _days())
    assert len(actions["DIA"]) == len(actions["XLE"]) == 1
    capture["pages"][0]["corporate_actions"]["forward_splits"][0]["id"] = "dividend-1"
    with pytest.raises(ValueError, match="invalid_corporate_action_identity"):
        development_actions(capture, _days())


def test_malformed_process_date_fails_action_capture():
    capture = _capture()
    capture["pages"][0]["corporate_actions"]["cash_dividends"][0]["process_date"] = "2024-99-99"
    with pytest.raises(ValueError, match="month must be in 1..12"):
        development_actions(capture, _days())


def test_audit_counts_support_without_claiming_strategy_or_holdout(tmp_path):
    days = _days()
    pointer = _acquisition(tmp_path, days)
    action_path = tmp_path / "actions.json"
    action_path.write_text(json.dumps(_capture()))
    report = audit(pointer, action_path)
    assert report["candidate_friday_sessions"] == report["common_valid_weeks"] == 96
    assert report["symbols"]["DIA"]["captured_actions_with_valid_boundary_bars"] == 1
    assert report["symbols"]["XLE"]["captured_actions_with_valid_boundary_bars"] == 1
    assert report["status"] == "observation_support_measured_action_completeness_unverified"
    assert report["signals_computed"] is report["returns_computed"] is report["trial_claimed"] is report["holdout_evaluated"] is False


def test_changed_bar_source_is_rejected_before_analysis(tmp_path):
    pointer = _acquisition(tmp_path, _days())
    (tmp_path / "DIA.csv").write_text("changed\n")
    action_path = tmp_path / "actions.json"
    action_path.write_text(json.dumps(_capture()))
    with pytest.raises(ValueError, match="source hash mismatch"):
        audit(pointer, action_path)
