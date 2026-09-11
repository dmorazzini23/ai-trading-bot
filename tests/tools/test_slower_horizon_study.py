from __future__ import annotations

import json

import pandas as pd
import pytest

from ai_trading.tools import slower_horizon_study as study
from ai_trading.utils.market_calendar import is_trading_day, session_info


def test_signal_excludes_future_and_missing_sessions_do_not_shift_horizon():
    frames = []
    for i, day in enumerate(day.date() for day in pd.date_range("2024-01-01", "2024-05-01") if is_trading_day(day.date())):
        session = session_info(day)
        index = pd.date_range(session.start_utc, session.end_utc, freq="min", inclusive="left")
        frames.append(pd.DataFrame({"open": 100. + i, "high": 102. + i, "low": 99. + i, "close": 101. + i}, index=index))
    data = pd.concat(frames)
    rows, _ = study.build_opportunities(data, "SPY", "2024-01-01", "2024-05-01")
    assert rows[0]["selected"] is True
    assert rows[0]["entry"] == frames[61].index[0].isoformat()
    assert rows[0]["exit"] == frames[66].index[0].isoformat()
    assert rows[1]["entry"] == rows[0]["exit"]
    data.loc[frames[61].index[0]:, "close"] *= 2
    changed, _ = study.build_opportunities(data, "SPY", "2024-01-01", "2024-05-01")
    assert changed[0]["signal_return"] == rows[0]["signal_return"]
    assert changed[0]["selected"] == rows[0]["selected"]
    data = data.drop(frames[63].index[0])
    missing, rejected = study.build_opportunities(data, "SPY", "2024-01-01", "2024-05-01")
    assert all(row["entry"] != rows[0]["entry"] for row in missing)
    assert rejected["opportunity_has_incomplete_session"] > 0


def test_positive_strategy_still_fails_without_incremental_baseline_value():
    rows = [{"session": str(i), "fold": str(i // 12), "symbol": symbol, "selected": True, "gross_bps": 100.} for i in range(96) for symbol in ["A", "B", "C", "D", "E", "F"]]
    result = study.evaluate(pd.DataFrame(rows), ["A", "B", "C", "D", "E", "F"])
    assert result["aggregate"]["net_per_opportunity_bps"] == 90
    assert result["block_bootstrap_95_bps"]["vs_always_long"] == [0, 0]
    assert result["decision"] == "retired"
    assert result["aggregate"]["stress_net_bps"]["20"] == 80


def test_missing_symbol_and_duplicate_grain_cannot_pass():
    data = pd.DataFrame([{"session": "2024-01-01", "fold": "2024Q1", "symbol": "A", "selected": True, "gross_bps": 999}])
    assert study.evaluate(data, ["A", "B"])["reason"] == "no_common_opportunities"
    with pytest.raises(ValueError, match="duplicate opportunity"):
        study.evaluate(pd.concat([data, data]), ["A"])


def test_one_bootstrap_block_does_not_imply_zero_uncertainty():
    data = pd.DataFrame([{"session": str(i), "fold": "2024Q1", "symbol": "A", "selected": True, "gross_bps": float(i)} for i in range(4)])
    result = study.evaluate(data, ["A"])
    assert result["decision"] == "inconclusive_budget_consumed"
    assert result["block_bootstrap_95_bps"] is None
    assert result["uncertainty_status"] == "unavailable_fewer_than_two_blocks"


def test_holdout_is_rejected_before_any_price_loader_or_budget_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "STATE_PATH", tmp_path / "state.json")
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"dataset_identity": {"request_interval": {"start_date": "2026-09-09", "end_date": "2026-12-08"}, "feed": "sip", "adjustment": "split"}}))
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(json.dumps({"manifest_path": str(meta)}))
    def forbidden(*args, **kwargs):
        pytest.fail("must not read prices or claim budget for holdout")
    monkeypatch.setattr(study, "_resolve_training_input", forbidden)
    monkeypatch.setattr(study, "claim_campaign_trial", forbidden)
    with pytest.raises(ValueError, match="frozen development"):
        study.main(["--acquisition", str(acquisition)])
    assert json.loads(study.STATE_PATH.read_text())["trials"] == []


def test_structurally_insufficient_support_blocks_before_price_loading_and_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "STATE_PATH", tmp_path / "state.json")
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"dataset_identity": {"request_interval": {"start_date": "2024-01-01", "end_date": "2025-12-31"}, "feed": "sip", "adjustment": "split"}}))
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(json.dumps({"manifest_path": str(meta)}))
    monkeypatch.setattr(study, "audit_sampling_acquisition", lambda *args: {"status": "blocked_sampling_support", "possible_common_periods": 4})
    def forbidden(*args, **kwargs):
        pytest.fail("insufficient timestamp support must block before outcomes or claim")
    monkeypatch.setattr(study, "_resolve_training_input", forbidden)
    monkeypatch.setattr(study, "claim_campaign_trial", forbidden)
    assert study.main(["--acquisition", str(acquisition), "--output-dir", str(tmp_path / "out")]) == 2
    assert json.loads(study.STATE_PATH.read_text())["trials"] == []
    assert json.loads((tmp_path / "out/report.json").read_text())["budget_consumed"] is False
