import hashlib
import json

import pandas as pd
import pytest

from ai_trading.tools.research_feasibility import require_feasibility, screen_development


def _protocol():
    return {"hypothesis": "fixed mechanism", "stopping_rule": "one trial", "horizons_minutes": [15], "retrospective_intervals": [{"name": "development", "start": "2025-07-21", "end": "2025-12-31"}, {"name": "validation", "start": "2026-01-01", "end": "2026-06-30"}], "acceptance": {"primary_one_way_cost_bps": 6, "minimum_completed_validation_trades": 250}}


def test_feasibility_rejects_cost_and_support_without_using_validation_returns():
    protocol = _protocol()
    days = pd.bdate_range("2025-07-21", "2025-12-31")
    ledger = pd.DataFrame([{"session": str(day.date()), "selected": True, "gross_return_bps": 50.0} for day in days for _ in range(10)])
    report = screen_development(protocol, {15: ledger})
    assert report["status"] == "eligible_for_fixed_study"
    polluted = pd.concat([ledger, pd.DataFrame([{"session": "2026-02-02", "selected": True, "gross_return_bps": -1e9}])], ignore_index=True)
    assert screen_development(protocol, {15: polluted}) == report
    costs = screen_development(protocol, {15: ledger.assign(gross_return_bps=1.0)})
    assert "optimistic_gross_edge_does_not_cover_costs" in costs["horizons"][0]["reasons"]
    support = screen_development(protocol, {15: ledger.iloc[:10]})
    assert "expected_trade_support_below_requirement" in support["horizons"][0]["reasons"]


def test_feasibility_review_is_bound_to_protocol_and_artifact(tmp_path):
    protocol = _protocol()
    with pytest.raises(ValueError, match="required"):
        require_feasibility(protocol)
    days = pd.bdate_range("2025-07-21", "2025-12-31")
    ledger = pd.DataFrame([{"session": str(day.date()), "selected": True, "gross_return_bps": 50.0} for day in days for _ in range(10)])
    path = tmp_path / "review.json"
    path.write_text(json.dumps(screen_development(protocol, {15: ledger})))
    protocol["feasibility_review"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    require_feasibility(protocol)
    protocol["hypothesis"] = "changed after review"
    with pytest.raises(ValueError, match="mismatch"):
        require_feasibility(protocol)
